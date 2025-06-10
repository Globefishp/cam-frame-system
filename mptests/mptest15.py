from multiprocessing import Queue
from queue import Empty  # 明确导入 Empty 异常

q = Queue()

try:
    item = q.get(timeout=1)  # 如果队列为空，1秒后抛出 Empty 异常
except Empty:  # 直接捕获 Empty 异常
    print("队列为空")