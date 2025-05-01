import numpy as np
from multiprocessing.shared_memory import SharedMemory

# 写入端
a = np.random.rand(4, 4).astype('float32')
b = np.random.rand(4, 4).astype('float32')
combined = np.array([a, b])  # shape=(2, 32, 32)
print(combined)

shm = SharedMemory(create=True, size=combined.nbytes)
shm_arr = np.ndarray(combined.shape, dtype=combined.dtype, buffer=shm.buf)
shm_arr[:] = combined[:]  # 写入数据

# 读取端
existing_shm = SharedMemory(name=shm.name)
loaded = np.ndarray((2, 4, 4), dtype='float32', buffer=existing_shm.buf)
a_loaded, b_loaded = loaded[0], loaded[1]  # 直接切片获取
print(loaded)

existing_shm.close()  # 关闭共享内存
shm.close()  # 关闭共享内存
shm.unlink()  # 可选：删除共享内存

# 写入端（手动分开写入）
shm = SharedMemory(create=True, size=a.nbytes + b.nbytes)
a_shm = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf[0:])
b_shm = np.ndarray(b.shape, dtype=b.dtype, buffer=shm.buf[a.nbytes:])
a_shm[:] = a[:]
b_shm[:] = b[:]

# 读取端
existing_shm = SharedMemory(name=shm.name)
a_loaded = np.ndarray((4, 4), dtype='float32', buffer=existing_shm.buf[0:])
b_loaded = np.ndarray((4, 4), dtype='float32', buffer=existing_shm.buf[a_loaded.nbytes:])
print(a_loaded)
print(b_loaded)