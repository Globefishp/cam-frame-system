from multiprocessing import Queue, Process

def worker(q):
    data = q.get()  # 取出 list
    data.append(100)  # 修改它
    print("Worker:", data)  # [1, 2, 100]

if __name__ == '__main__':
    q = Queue()
    a = [1, 2]
    q.put(a)  # 放入 list

    p = Process(target=worker, args=(q,))
    p.start()
    p.join()

    # 主进程的队列数据不受影响
    print("Main:", a)

import numpy as np
from multiprocessing import Queue, Process

def worker(q):
    arr = q.get()  # 取出 numpy 数组
    arr[0] = 100   # 修改它
    print("Worker:", arr)  # [100, 2, 3]

if __name__ == '__main__':
    q = Queue()
    arr = np.array([1, 2, 3])
    q.put(arr)  # 放入 numpy 数组

    p = Process(target=worker, args=(q,))
    p.start()
    p.join()

    # 主进程的数组不受影响
    print("Main:", arr)  # [1, 2, 3]