import os
import time
import multiprocessing as mp
import sys
from loguru import logger

# 1. 显式设置 spawn 模式
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# --- 模拟外部模块或深层函数 ---
def external_function():
    # 它直接调用全局 logger，不需要任何修改
    logger.info("  >> 正在执行深层任务...")

class MyProcessWorker(mp.Process):
    def __init__(self, task_id, parent_logger):
        super().__init__()
        self.task_id = task_id
        # 1. 使用【实例属性】接收。
        # 在 p.start() 被调用时，这个实例会被安全地带到子进程中。
        self._logger = parent_logger

    def run(self):
        """ 这里是子进程的独立世界 """
        
        # 2. 全局覆盖（推荐做法）
        # 将传入的带有安全队列的 logger，覆盖当前子进程的全局 logger
        global logger
        logger = self._logger

        # 你现在既可以使用 self._logger，也可以直接使用全局的 logger
        logger.info(f"[子进程 {self.task_id} - PID: {os.getpid()}] 开始工作")
        
        # 调用其他不接受 logger 参数的函数也能正常记录
        external_function()
        time.sleep(0.5)

        logger.success(f"[子进程 {self.task_id}] 工作完成")

        # 3. 极其重要：在子进程退出前，确保异步队列清空！
        # 官方文档指出：enqueue=True 会在后台开启消费线程。
        # 子进程结束时，必须调用 complete() 确保所有日志发送回主进程。
        self._logger.complete()


if __name__ == "__main__":
    # 统一入口配置
    context = mp.get_context('spawn')
    logger.remove()
    logger.add(sys.stderr, enqueue=True, context=context)
    logger.add(
        "process_best_practice.log", 
        mode="w", 
        enqueue=True, 
        format="{time:HH:mm:ss.SSS} | {level: <8} | {process.id} | {message}"
    )

    logger.info(f"主进程 (PID: {os.getpid()}) 启动")

    processes = []
    for i in range(3):
        p = MyProcessWorker(i, logger)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    logger.info("主进程结束")
    # 主进程结束前，也建议调用一次 complete() 确保主进程的日志也 flush 干净
    logger.complete()