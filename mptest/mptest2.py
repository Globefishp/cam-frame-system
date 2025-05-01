import multiprocessing
import multiprocessing.shared_memory as shared_memory
import numpy as np

def worker(shm):
    # ❌ 错误：直接修改传入的 SharedMemory 对象（可能不安全）
    arr = np.ndarray((10,), dtype=np.int32, buffer=shm.buf)
    arr[0] = 100  # 修改数据是有效的，但 shm 对象本身不能直接共享
    print(f"Worker: arr[0] = {arr[0]}")

if __name__ == '__main__':
    # 创建共享内存
    shm = shared_memory.SharedMemory(create=True, size=10 * 4)  # 10个int32
    arr = np.ndarray((10,), dtype=np.int32, buffer=shm.buf)
    arr[:] = 0  # 初始化为0
    print(f"Main: arr[0] = {arr[0]}")

    p = multiprocessing.Process(target=worker, args=(shm,))
    p.start()
    p.join()

    # 主进程读取数据
    arr = np.ndarray((10,), dtype=np.int32, buffer=shm.buf)
    print(f"Main: arr[0] = {arr[0]}")  

    shm.close()
    shm.unlink()