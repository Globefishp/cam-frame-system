import multiprocessing
import time

def worker(q):
    time.sleep(2)
    print("Getting item:", q.get())

if __name__ == "__main__":
    q = multiprocessing.Queue(maxsize=2)  # 缓冲区大小为 2

    # 启动消费者进程
    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()

    # 生产者尝试放入数据
    q.put(1)  # 成功
    q.put(2)  # 成功
    print("Queue is full, blocking...")
    q.put(3, block=True)  # 阻塞，直到消费者取出一个元素

    p.join()