import multiprocessing
from multiprocessing import shared_memory
import os

def test_shared_memory_size(shm_name, size):
    """在子进程中测试共享内存的大小"""
    print(f"\n子进程 {os.getpid()} 尝试连接共享内存 '{shm_name}'")
    
    # 连接到已存在的共享内存
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    
    try:
        # 检查共享内存大小
        print(f"子进程确认共享内存大小: {existing_shm.size} 字节")
        assert existing_shm.size == size, "子进程中共享内存大小与创建时不一致"
    finally:
        # 关闭连接但不释放内存（由主进程释放）
        existing_shm.close()

def main():
    # 设置共享内存大小（以字节为单位）
    shm_size = 1024  # 1KB
    shm_name = "test_shared_memory"
    
    print(f"主进程 {os.getpid()} 创建共享内存 '{shm_name}'，大小: {shm_size} 字节")
    
    # 创建共享内存
    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=shm_size)
    
    try:
        # 检查主进程中的共享内存大小
        print(f"主进程确认共享内存大小: {shm.size} 字节")
        assert shm.size == shm_size, "主进程中共享内存大小创建不正确"
        
        # 创建并启动子进程来测试
        p = multiprocessing.Process(target=test_shared_memory_size, args=(shm_name, shm_size))
        p.start()
        p.join()
        
    finally:
        # 清理共享内存
        print(f"主进程释放共享内存 '{shm_name}'")
        shm.close()
        shm.unlink()

if __name__ == "__main__":
    main()