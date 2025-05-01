import numpy as np
a = np.array((1, 2))*2
print(a)
print(*a)


b=slice(1,3)
c=[1,2,3]
print(c[b])

arr = np.arange(10)
view = arr[::2]  # 非连续视图（步长为2）
print(view.flags['C_CONTIGUOUS'])  # 输出 False
print(view.flags['F_CONTIGUOUS'])  # 输出 False

import multiprocessing.shared_memory as shared_memory
import numpy as np

shm = shared_memory.SharedMemory(create=True, size=16 * np.dtype('int').itemsize)
arr = np.ndarray((16,), dtype='int', buffer=shm.buf)
arr[:] = np.arange(16)
print(arr)

ashape = (2, 4)
shm_view = np.ndarray((2,) + ashape, dtype='int', buffer=shm.buf)
print(shm_view)

print((2,) + ashape)

dtype = np.dtype('uint8')
print(np.dtype('uint8').itemsize)

def myfunc(a = np.dtype('uint8')):
    print(type(a))
    b=np.dtype('uint8')
    print(type(b))

myfunc()

c = np.dtype(np.uint8)
d = np.dtype(c)
print(type(c))
print(type(d))