from loguru import logger

def do_work_default():
    # 错误示范：依赖模块顶部的全局 logger
    logger.info("    ❌ [Dummy] 全局导入的 logger 被调用了！")

def do_work_injected(passed_logger):
    # 正确示范：使用显式传递进来的 logger
    passed_logger.success("    ✅ [Dummy] 显式注入的 logger 被调用了！")
