import multiprocessing
from multiprocessing import shared_memory
import os

def write_test_data(shm, size):
    """主进程写入测试数据到共享内存"""
    # 在头部写入标记 "HEAD"
    head_marker = b"HEAD"
    shm.buf[:len(head_marker)] = head_marker

    # 在尾部写入标记 "TAIL"（假设实际大小是 4096）
    tail_offset = 4096 - len(b"TAIL")  # 假设实际大小是 4096
    if tail_offset >= size:  # 确保不越界
        tail_offset = size - len(b"TAIL")
    shm.buf[tail_offset:tail_offset + len(b"TAIL")] = b"TAIL"

    print(f"主进程写入数据: 头部标记 '{head_marker.decode()}', 尾部标记 'TAIL' (假设实际大小 4096)")

def check_test_data(shm, size):
    """子进程检查共享内存中的标记"""
    # 检查头部是否有 "HEAD"
    head_marker = b"HEAD"
    if shm.buf[:len(head_marker)] == head_marker:
        print(f"子进程: 在共享内存头部找到标记 '{head_marker.decode()}'")
    else:
        print(f"子进程: 共享内存头部未找到标记 'HEAD'")

    # 检查尾部是否有 "TAIL"（假设实际大小是 4096）
    tail_marker = b"TAIL"
    tail_offset = 4096 - len(tail_marker)  # 假设实际大小是 4096
    if tail_offset >= size:  # 确保不越界
        tail_offset = size - len(tail_marker)
    if shm.buf[tail_offset:tail_offset + len(tail_marker)] == tail_marker:
        print(f"子进程: 在共享内存尾部找到标记 'TAIL'")
    else:
        print(f"子进程: 共享内存尾部未找到标记 'TAIL'")

    # 打印共享内存的实际大小（通过系统 API 获取会更准确，但 Python 的 shared_memory 不提供）
    print(f"子进程确认共享内存大小: {shm.size} 字节 (但实际可能更大)")

def main():
    shm_size = 1024  # 请求的大小
    shm_name = "test_shared_memory_4096_test"

    # 主进程创建共享内存
    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=shm_size)
    try:
        print(f"主进程 {os.getpid()} 创建共享内存 '{shm_name}'，请求大小: {shm_size} 字节")

        # 写入测试数据
        write_test_data(shm, shm_size)

        # 子进程验证
        p = multiprocessing.Process(target=check_test_data, args=(shm, shm_size))
        p.start()
        p.join()

    finally:
        shm.close()
        shm.unlink()  # 清理共享内存

if __name__ == "__main__":
    main()