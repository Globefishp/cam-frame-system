from threading import Lock, Thread, Condition
import time
import random

class LockTest:
    def __init__(self):
        self.lockA = Lock()  # 创建锁A
        self.lockB = Lock()  # 创建锁B
        self.lockC = Lock()  # 创建锁C
    
    def writer(self, num):
        self.lockA.acquire()
        self.lockB.acquire()
        # do something
        print("writer", num)
        time.sleep(0.2)
        self.lockA.release()
        # wait reader to read
        self.lockB.release()
        # let read_and_delete to do something

    def reader(self, num):
        self.lockA.acquire() # 假定lockA是快过程（也就是writer总能很快的释放）
        if self.lockC.locked(): # if deleter is working
            self.lockA.release() # 释放权限(后续deleter转交给writer)
            return None # 不读取/或者也可以等待writer的信号量。
        else: # if deleter is not working
            self.lockC.acquire()
        # do something
        print("reader", num)
        time.sleep(0.2)
        self.lockC.release()
        # let read_and_delete to do something
        self.lockA.release()
        pass

    def read_and_delete(self, num):
        self.lockB.acquire()
        self.lockC.acquire() # will block here until reader finished
        # do something
        print("read_and_delete")
        time.sleep(0.2)
        self.lockB.release()
        self.lockC.release()
        pass

if __name__ == "__main__":
    lock_test = LockTest()
    def writer_worker(LockTest: LockTest):
        while True:
            LockTest.writer()
            time.sleep(0.1)
    def reader_worker(LockTest: LockTest):
        while True:
            LockTest.reader()
            time.sleep(0.1)
    def read_and_delete_worker(LockTest: LockTest):
        while True:
            LockTest.read_and_delete()
            time.sleep(0.1)
    tw = Thread(target=lock_test.writer).start()
    tr = Thread(target=lock_test.reader).start()
    td = Thread(target=lock_test.read_and_delete).start()