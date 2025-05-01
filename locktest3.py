import threading
from collections import deque

# class PriorityLock:
#     '''
#     实现了一个优先级锁，用于控制对共享资源的访问。
#     该锁支持多个线程同时请求访问，但会按照预定的顺序进行释放。
#     这一顺序是环形的，优先级0>1>2>0...
#     首先实现了3个优先级。
#     '''
#     def __init__(self, level: int = 3):
#         self.lock = threading.Lock()
#         self._internal_lock = threading.Lock()
#         self.condition = threading.Condition(self.lock)
#         self.waiting_priority = [0 for i in range(level)]
#         self.current_priority = None
#         # 用于决定控制权将要移交的下一个层级，为环形结构。先不考虑一个优先级有多次请求。
    
#     def acquire(self, priority: int, timeout: int) -> bool:
#         # 先不考虑锁
#         self.waiting_priority[priority] = 1 # 此优先级有等待。
#         if not self.lock.locked():
#             self.lock.acquire()
#             self.current_priority = priority
#             return True
#         else:
#             # 等待到超时后获取，但要注意internallock的控制问题。
#             pass
#         self.waiting_priority[priority] = 0 # 此优先级无等待。
#         return False
            
    
#     def release(self):

        

class OrderedRingBuffer:
    def __init__(self, capacity):
        self.buffer = [None] * capacity
        self.capacity = capacity
        self.head = 0
        self.tail = 0
        self.size = 0
        
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        
        # 等待队列和优先级管理
        self.waiting_queues = {i: deque() for i in range(3)}
        self.priority_order = [i for i in range(3)]  # 优先级顺序
        self.waiting_num = 0
        self.current_holder = None  # 当前持有锁的线程优先级
    
    def _enter_queue(self, caller):
        """进入等待队列并返回是否需要等待"""
        with self.condition:
            if self.current_holder is None:
                # 没有线程持有锁，直接获取
                self.current_holder = caller
                # print('no lock, execute immediately.')
                return False
            else:
                # 加入等待队列
                event = threading.Event()
                self.waiting_num += 1
                self.waiting_queues[caller].append(event)
                # print('Locked, add to queue, caller:', caller)
                return True
    
    def _exit_and_schedule(self, caller_type):
        """释放锁并选择下一个线程"""
        with self.condition:
            # 检查所有优先级队列，按优先级顺序循环选择下一个线程
            # print('now releasing from caller: ', caller_type)
            if self.waiting_num:
                for priority in [(1 + i + self.current_holder) % 3 for i in range(3)]:
                    if self.waiting_queues[priority]:
                        self.waiting_num -= 1
                        next_event = self.waiting_queues[priority].popleft()
                        self.current_holder = priority
                        # print('now notifying: ', self.current_holder)
                        next_event.set() # 通知正在等待的线程
                        return
            
            # 没有等待的线程，完全释放锁
            self.current_holder = None
            self.condition.notify_all()
    
    def write(self, item):
        # 尝试获取执行权
        need_wait = self._enter_queue(0)
        
        if need_wait:
            event = self.waiting_queues[0][-1]
            event.wait()
        
        try:
            # 实际的写入逻辑
            with self.lock:
                if self.size == self.capacity:
                    return False
                
                self.buffer[self.tail] = item
                self.tail = (self.tail + 1) % self.capacity
                self.size += 1
                return True
        finally:
            self._exit_and_schedule(0)
    
    def read(self):
        # 尝试获取执行权
        need_wait = self._enter_queue(1)
        
        if need_wait:
            event = self.waiting_queues[1][-1]
            event.wait()
        
        try:
            # 实际的读取逻辑
            with self.lock:
                if self.size == 0:
                    return None
                
                item = self.buffer[self.head]
                return item
        finally:
            self._exit_and_schedule(1)
    
    def delete(self):
        # 尝试获取执行权
        need_wait = self._enter_queue(2)
        
        if need_wait:
            event = self.waiting_queues[2][-1]
            event.wait()
        
        try:
            # 实际的删除逻辑
            with self.lock:
                if self.size == 0:
                    return False
                
                self.buffer[self.head] = None
                self.head = (self.head + 1) % self.capacity
                self.size -= 1
                return True
        finally:
            self._exit_and_schedule(2)

import time
import random

def test_concurrent_access():
    buffer = OrderedRingBuffer(10)
    
    def writer_thread():
        for i in range(2):
            # time.sleep(random.uniform(0.5, 0.7))
            buffer.write(i)
            print(f"Writer wrote: {i}")
    
    def reader_thread():
        for _ in range(1):
            # time.sleep(random.uniform(0.5, 0.7))
            item = buffer.read()
            print(f"Reader read: {item}")
    
    def deleter_thread():
        for _ in range(2):
            # time.sleep(random.uniform(0.5, 0.7))
            success = buffer.delete()
            print(f"Deleter {'succeeded' if success else 'failed'}")
    
    threads = [
        threading.Thread(target=deleter_thread),
        threading.Thread(target=reader_thread),
        threading.Thread(target=writer_thread),
    ]
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()

# test_concurrent_access()

import time
import threading
import statistics
from collections import defaultdict

class SchedulerBenchmark:
    def __init__(self, buffer_class):
        self.buffer_class = buffer_class
        self.results = defaultdict(dict)
    
    def run_benchmarks(self):
        self.test_no_contention()
        self.test_low_contention()
        self.test_high_contention()
        self.test_extreme_contention()
        self.print_results()
    
    def timed_op(self, func, *args):
        start = time.perf_counter_ns()
        func(*args)
        return time.perf_counter_ns() - start
    
    def test_no_contention(self):
        """单线程基准测试"""
        buffer = self.buffer_class(1000)
        times = []
        
        # 测试写操作
        for i in range(1000):
            times.append(self.timed_op(buffer.write, i))
        
        # 测试读操作
        for _ in range(1000):
            times.append(self.timed_op(buffer.read))
        
        # 测试删除操作
        for _ in range(1000):
            times.append(self.timed_op(buffer.delete))
        
        self.results['no_contention'] = {
            'avg': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times),
            'unit': 'ns'
        }
    
    def test_low_contention(self):
        """低竞争测试(3个线程)"""
        buffer = self.buffer_class(1000)
        results = {'write': [], 'read': [], 'delete': []}
        
        def worker(op, count):
            for i in range(count):
                start = time.perf_counter_ns()
                op(i) if op == buffer.write else op()
                duration = time.perf_counter_ns() - start
                results[op.__name__].append(duration)
        
        threads = [
            threading.Thread(target=worker, args=(buffer.write, 300)),
            threading.Thread(target=worker, args=(buffer.read, 300)),
            threading.Thread(target=worker, args=(buffer.delete, 300))
        ]
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_time = time.time() - start
        
        self.results['low_contention'] = {
            'total_time': total_time,
            'writer': statistics.mean(results['write']),
            'reader': statistics.mean(results['read']),
            'deleter': statistics.mean(results['delete']),
            'unit': 'ns'
        }
    
    def test_high_contention(self):
        """高竞争测试(9个线程)"""
        buffer = self.buffer_class(1000)
        results = defaultdict(list)
        
        def worker(op, count):
            for i in range(count):
                start = time.perf_counter_ns()
                op(i) if op == buffer.write else op()
                duration = time.perf_counter_ns() - start
                results[op.__name__].append(duration)
        
        threads = []
        for _ in range(3):  # 每种操作3个线程
            threads.append(threading.Thread(target=worker, args=(buffer.write, 100)))
            threads.append(threading.Thread(target=worker, args=(buffer.read, 100)))
            threads.append(threading.Thread(target=worker, args=(buffer.delete, 100)))
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_time = time.time() - start
        
        self.results['high_contention'] = {
            'total_time': total_time,
            'writer': statistics.mean(results['write']),
            'reader': statistics.mean(results['read']),
            'deleter': statistics.mean(results['delete']),
            'unit': 'ns'
        }
    
    def test_extreme_contention(self):
        """极端竞争测试(30个线程)"""
        buffer = self.buffer_class(1000)
        results = defaultdict(list)
        lock = threading.Lock()
        
        def worker(op, count):
            for i in range(count):
                start = time.perf_counter_ns()
                op(i) if op == buffer.write else op()
                duration = time.perf_counter_ns() - start
                with lock:
                    results[op.__name__].append(duration)
        
        threads = []
        for _ in range(10):  # 每种操作10个线程
            threads.append(threading.Thread(target=worker, args=(buffer.write, 30)))
            threads.append(threading.Thread(target=worker, args=(buffer.read, 30)))
            threads.append(threading.Thread(target=worker, args=(buffer.delete, 30)))
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_time = time.time() - start
        
        self.results['extreme_contention'] = {
            'total_time': total_time,
            'writer': statistics.mean(results['write']),
            'reader': statistics.mean(results['read']),
            'deleter': statistics.mean(results['delete']),
            'unit': 'ns'
        }
    
    def print_results(self):
        print("\n=== Scheduler Benchmark Results ===")
        for test, data in self.results.items():
            print(f"\nTest: {test.replace('_', ' ').title()}")
            if 'avg' in data:
                print(f"Average operation time: {data['avg']:.2f} {data['unit']}")
                print(f"Median operation time: {data['median']:.2f} {data['unit']}")
                print(f"Standard deviation: {data['stdev']:.2f} {data['unit']}")
            else:
                print(f"Total execution time: {data['total_time']:.4f} sec")
                print(f"Average write time: {data['writer']:.2f} {data['unit']}")
                print(f"Average read time: {data['reader']:.2f} {data['unit']}")
                print(f"Average delete time: {data['deleter']:.2f} {data['unit']}")

# 对比实现：简单锁版本
class SimpleLockRingBuffer:
    def __init__(self, capacity):
        self.buffer = [None] * capacity
        self.capacity = capacity
        self.head = 0
        self.tail = 0
        self.size = 0
        self.lock = threading.Lock()
    
    def write(self, item):
        with self.lock:
            if self.size == self.capacity:
                return False
            self.buffer[self.tail] = item
            self.tail = (self.tail + 1) % self.capacity
            self.size += 1
            return True
    
    def read(self):
        with self.lock:
            if self.size == 0:
                return None
            return self.buffer[self.head]
    
    def delete(self):
        with self.lock:
            if self.size == 0:
                return False
            self.buffer[self.head] = None
            self.head = (self.head + 1) % self.capacity
            self.size -= 1
            return True

# 运行测试
if __name__ == "__main__":
    print("Testing OrderedRingBuffer...")
    ordered_bench = SchedulerBenchmark(OrderedRingBuffer)
    ordered_bench.run_benchmarks()
    
    print("\nTesting SimpleLockRingBuffer...")
    simple_bench = SchedulerBenchmark(SimpleLockRingBuffer)
    simple_bench.run_benchmarks()