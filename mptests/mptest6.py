import multiprocessing
import os
import time

def worker_process(task_id):
    # 每个子进程都会执行这个函数
    print(f"子进程 {task_id} (PID: {os.getpid()}) 开始执行")
    time.sleep(1)  # 模拟工作
    print(f"子进程 {task_id} (PID: {os.getpid()}) 完成任务")

if __name__ == '__main__':
    print(f"主进程 (PID: {os.getpid()}) 启动")
    
    processes = []
    for i in range(3):  # 创建3个子进程
        p = multiprocessing.Process(target=worker_process, args=(i,))
        processes.append(p)
        p.start()
    
    # 等待所有子进程完成
    for p in processes:
        p.join()
    
    print("主进程结束")