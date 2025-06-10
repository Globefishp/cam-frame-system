import threading
import time

# 创建一个锁
lock = threading.Lock()

def worker(thread_id):
    # 线程1 先获取锁并持有 2 秒
    if thread_id == 1:
        lock.acquire()
        print(f"Thread {thread_id} 获取了锁，并持有 2 秒")
        time.sleep(2)
        lock.release()
        print(f"Thread {thread_id} 释放了锁")
    # 线程2 尝试获取锁（blocking=True, timeout=0）
    else:
        acquired = lock.acquire(blocking=True, timeout=0)
        if acquired:
            print(f"Thread {thread_id} 成功获取了锁")
            lock.release()
        else:
            print(f"Thread {thread_id} 未能获取锁（锁已被占用）")

# 创建并启动两个线程
thread1 = threading.Thread(target=worker, args=(1,))
thread2 = threading.Thread(target=worker, args=(2,))

thread1.start()
thread2.start()

thread1.join()
thread2.join()

print("程序结束")