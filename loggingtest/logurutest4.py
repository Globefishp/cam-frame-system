import sys
import multiprocessing as mp
from loguru import logger

# Loguru的logger可以直接在mp环境里序列化, 但是不能标准序列化
# 下面两个方式都可以

class WorkerClass:
    def __init__(self):
        # 将全局 loguru logger 赋值给实例属性
        self._logger = logger
        # 可选：为子进程配置独立的日志 sink（如输出到文件），便于观察
        # 此处使用默认的 stderr sink，子进程会继承标准输出，一般可用
        self._logger.info("WorkerClass 初始化完成（主进程）")

    def _worker(self):
        """子进程目标函数，尝试使用 self._logger"""
        try:
            # 尝试记录日志
            self._logger.info("子进程 _worker 开始执行")
            self._logger.debug(f"子进程 PID: {mp.current_process().pid}")
            self._logger.warning("子进程日志测试成功")
            # 再记录一条错误级别日志
            self._logger.error("如果看到这条消息，说明 logger 在子进程中完全可用")
        except Exception as e:
            # 若出现异常，捕获并打印到 stderr（因为此时 logger 可能不可用）
            print(f"子进程中访问 self._logger 失败: {type(e).__name__}: {e}", file=sys.stderr)
            raise

    def start(self):
        """启动子进程执行 _worker 方法"""
        self._logger.info("主进程准备启动子进程")
        # 注意：target=self._worker 会隐式传递 self 对象，导致序列化
        p = mp.Process(target=self._worker, name="TestWorkerProcess")
        p.start()
        p.join()
        self._logger.info("子进程已结束，主进程退出")

class WorkerClass2:
    def __init__(self):
        # 将全局 loguru logger 赋值给实例属性
        self._logger = logger
        # 可选：为子进程配置独立的日志 sink（如输出到文件），便于观察
        # 此处使用默认的 stderr sink，子进程会继承标准输出，一般可用
        self._logger.info("WorkerClass 初始化完成（主进程）")
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_logger"]
        return state

    def _worker(self, logger):
        """子进程目标函数，尝试使用 self._logger"""
        self._logger = logger
        try:
            # 尝试记录日志
            self._logger.info("子进程 _worker 开始执行")
            self._logger.debug(f"子进程 PID: {mp.current_process().pid}")
            self._logger.warning("子进程日志测试成功")
            # 再记录一条错误级别日志
            self._logger.error("如果看到这条消息，说明 logger 在子进程中完全可用")
        except Exception as e:
            # 若出现异常，捕获并打印到 stderr（因为此时 logger 可能不可用）
            print(f"子进程中访问 self._logger 失败: {type(e).__name__}: {e}", file=sys.stderr)
            raise

    def start(self):
        """启动子进程执行 _worker 方法"""
        self._logger.info("主进程准备启动子进程")
        # 注意：target=self._worker 会隐式传递 self 对象，导致序列化
        p = mp.Process(target=self._worker, args=(self._logger,), name="TestWorkerProcess2")
        p.start()
        p.join()
        self._logger.info("子进程已结束，主进程退出")


if __name__ == "__main__":
    # 必须使用 if __name__ 保护，尤其对 Windows/macOS spawn 方式
    mp.set_start_method('spawn', force=True)  # 强制使用 spawn 方式，暴露序列化问题
    logger.remove()
    logger.add(sys.stderr, enqueue=True)
    logger.add("test4.log", enqueue=True)
    obj2 = WorkerClass2()
    obj2.start()
    print("Test1 finished")
    input()
    obj = WorkerClass()
    obj.start()
