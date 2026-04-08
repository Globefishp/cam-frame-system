import os
import time
import multiprocessing as mp
import sys
from loguru import logger

# ==========================================
# 0. 准备测试环境：动态生成一个外部模块文件
# ==========================================
def create_dummy_module():
    with open("logurutest_dummy.py", "w", encoding="utf-8") as f:
        f.write("""\
from loguru import logger

def do_work_default():
    # 错误示范：依赖模块顶部的全局 logger
    logger.info("    ❌ [Dummy] 全局导入的 logger 被调用了！")

def do_work_injected(passed_logger):
    # 正确示范：使用显式传递进来的 logger
    passed_logger.success("    ✅ [Dummy] 显式注入的 logger 被调用了！")
""")

# ==========================================
# 1. 全局导入外部模块（模拟日常开发中的习惯）
# ==========================================
create_dummy_module()
import logurutest_dummy  # 提前导入，模拟实际工程中不可避免的依赖链

# ==========================================
# 2. 定义多进程类
# ==========================================
class TestWorker(mp.Process):
    def __init__(self, passed_logger):
        super().__init__()
        # 保存父进程传来的实例
        self._logger = passed_logger

    def run(self):
        # 尝试 1：全局覆盖法
        global logger
        logger = self._logger
        logger.info(f"子进程 (PID: {os.getpid()}) 开始运行")

        # --- 猜想 1 验证：外部模块调用 ---
        # 结果：由于 dummy_module 在最顶部被 import，它内部的 logger 依然是默认状态。
        # 这里的日志只会输出到控制台，不会进入父进程的日志文件！
        logurutest_dummy.do_work_default()

        # --- 猜想 2 验证：推迟导入（脆弱性测试） ---
        # 结果：由于 spawn 机制和顶部的 import，sys.modules 里已经有 dummy_module。
        # 这里的 import 不会重新执行模块代码，覆盖依然无效！
        import logurutest_dummy as deferred_dummy
        deferred_dummy.do_work_default()

        # --- 猜想 3 验证：依赖注入（最佳实践） ---
        # 结果：完美输出到文件和控制台，架构最稳健！
        logurutest_dummy.do_work_injected(self._logger)

        self._logger.complete()


# ==========================================
# 3. 主进程逻辑
# ==========================================
if __name__ == "__main__":
    # 显式设置 spawn
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    log_file = "architecture_test.log"
    if os.path.exists(log_file):
        os.remove(log_file)

    # 主进程配置
    logger.remove()
    logger.add(sys.stderr, enqueue=True)
    logger.add(log_file, enqueue=True, mode="w", format="{level} | {message}")

    logger.info("主进程开始启动子进程...")

    # 启动子进程并传递 logger
    p = TestWorker(passed_logger=logger)
    p.start()
    p.join()

    logger.info("主进程结束。下面是日志文件内容：\n")

    # 打印文件内容验证
    print("=" * 40)
    print(f"📄 文件 {log_file} 的内容：")
    print("-" * 40)
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            print(f.read().strip())
    except FileNotFoundError:
        print("日志文件未生成！")
    print("=" * 40)
    
    # 清理临时文件
    if os.path.exists("dummy_module.py"):
        os.remove("dummy_module.py")