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

# 2. 配置 Logger (必须放在 if __name__ == "__main__" 之外)
# 这样子进程被 spawn 出来重新 import 这个文件时，也会执行这段配置
def configure_logging():
    # 先移除默认的控制台输出
    logger.remove()

    # 添加控制台输出
    logger.add(sys.stderr, enqueue=True)

    # 添加文件输出
    # 注意点1: 必须用 mode="a" (追加模式)，否则子进程启动会清空文件
    # 注意点2: enqueue=True 保证了多进程写入时的原子性和顺序
    logger.add(
        "spawn_test.log", 
        mode="a", 
        enqueue=True, 
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {process.id} | {message}"
    )
if __name__ == "__main__":
    # 清空旧文件(仅在主进程开始前清空一次)
    if os.path.exists("spawn_test.log"):
        os.remove("spawn_test.log")
# 立即执行配置
configure_logging()

def worker_task(task_id):
    # 在 spawn 的子进程中，这里的 logger 已经由上面的 configure_logging() 配置好了
    logger.info(f"子进程启动 | 任务ID: {task_id}")
    time.sleep(0.5)
    logger.success(f"子进程完成 | 任务ID: {task_id}")

if __name__ == "__main__":

    logger.info(f"主进程 PID: {os.getpid()} - 开始运行")

    processes = []
    for i in range(3):
        p = mp.Process(target=worker_task, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    logger.info("主进程结束")
    
    # 显式等待日志队列刷新 (在 spawn 模式下建议加上)
    logger.complete()