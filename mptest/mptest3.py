import multiprocessing
import multiprocessing.shared_memory as shared_memory
import numpy as np

def worker(shm_name):
    # ✅ 正确：子进程通过名称重新连接共享内存
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray((10,), dtype=np.int32, buffer=shm.buf)
    arr[0] = 100  # 修改数据
    print(f"Worker: arr[0] = {arr[0]}")

if __name__ == '__main__':
    # 创建共享内存
    shm = shared_memory.SharedMemory(create=True, size=10 * 4)  # 10个int32
    arr = np.ndarray((10,), dtype=np.int32, buffer=shm.buf)
    arr[:] = 0  # 初始化为0
    print(f"Main: arr[0] = {arr[0]}")

    # 启动子进程，并传递共享内存名称（而不是 SharedMemory 对象）
    p = multiprocessing.Process(target=worker, args=(shm.name,))
    p.start()
    p.join()

    arr2 = np.ndarray((10,), dtype=np.int32, buffer=shm.buf)
    # 只要shm.buf作为缓存，就会同步（底层数据相同）
    print(f"Main: arr[0] = {arr[0]}, arr2[0] = {arr2[0]}")  # 应该输出 100

    shm.close()
    shm.unlink()