import sys
from loguru import logger

# 1. 定义一个 patcher 函数
def patch_message(record):
    extra = record["extra"]
    # 获取 extra 中的字段（如果存在）
    pid = extra.get("pid")
    name = extra.get("name")
    
    # 根据字段是否存在，构建前缀
    prefix_parts = []
    if pid:
        prefix_parts.append(f"PID:{pid}")
    if name:
        prefix_parts.append(f"Worker:{name}")
    
    if prefix_parts:
        prefix = f"[{' '.join(prefix_parts)}] "
        # 修改 record["message"]，这会影响最终打印的内容
        record["message"] = f"{prefix}{record['message']}"

# 2. 配置 logger
logger.remove()  # 移除默认 handler
# 使用 patch 将函数应用到所有日志记录中
logger = logger.patch(patch_message)
# 添加 handler，这里依然可以使用默认格式或自定义格式
logger.add(sys.stderr)
# logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# 3. 测试
# 情况 A: 含有 pid 和 name
logger.bind(pid=1234, name="Process-1").info("这是一条普通消息")

# 情况 B: 只有 pid
logger.bind(pid=5678).warning("只有PID的消息")

# 情况 C: 什么都没有（保持原始状态）
logger.info("没有任何额外字段的消息")