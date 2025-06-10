import multiprocessing
import time

class MyClass:
    def __init__(self):
        self.a = 0  # 普通变量，不会被共享
    
    def worker(self):
        print(f"Worker: 初始值 a = {self.a}")
        self.a = 100  # 修改普通变量
        print(f"Worker: 修改后 a = {self.a}")
        time.sleep(1)  # 确保主进程有时间尝试读取
    
    def run_demo(self):
        print(f"主进程: 初始值 a = {self.a}")
        
        p = multiprocessing.Process(target=self.worker)
        p.start()
        p.join()  # 等待子进程结束
        
        print(f"主进程: 子进程结束后 a = {self.a}")  # 仍然保持原值

if __name__ == '__main__':
    obj = MyClass()
    obj.run_demo()