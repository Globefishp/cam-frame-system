from threading import Lock, Condition, Event
from collections import deque
from typing import Optional, List, Any
import time

class ControlledRingBuffer:
    '''
    仅进行过调度测试，没有进行过功能测试。
    '''
    def __init__(self, size: int):
        self.buffer = [None] * size
        self.size = size
        self.head = 0  # 生产者写入位置
        self.tail = 0  # 编码消费者读取位置
        self.lock = Lock()  # 保护head/tail的锁
        self.scheduler_condition = Condition(self.lock)  # 条件变量

        # 等待队列和优先级管理
        self.waiting_queues = {i: deque() for i in range(3)}
        self.waiting_num = 0
        self.priority_order = [i for i in range(3)]  # 优先级顺序
        self.current_holder = None  # 当前持有锁的线程优先级
    
    def _enter_queue(self, caller):
        """进入等待队列并返回是否需要等待"""
        with self.scheduler_condition:
            if self.current_holder is None:
                # 没有线程持有锁，直接获取
                self.current_holder = caller
                return False
            else:
                # 加入等待队列
                event = Event()
                self.waiting_num += 1
                self.waiting_queues[caller].append(event)
                return True
    
    def _exit_and_schedule(self, caller_type):
        """释放锁并选择下一个线程"""
        with self.scheduler_condition:
            # 检查所有优先级队列，按优先级顺序循环选择下一个线程
            if self.waiting_num:
                for priority in [(1 + i + self.current_holder) % 3 for i in range(3)]:
                    if self.waiting_queues[priority]:
                        self.waiting_num -= 1
                        next_event = self.waiting_queues[priority].popleft()
                        self.current_holder = priority
                        next_event.set() # 通知正在等待的下一个线程
                        return
            
            # 没有下一个线程正在等待，退出
            self.current_holder = None
            self.scheduler_condition.notify_all()

    def put(self, item) -> None:
        """
        生产者写入数据（线程安全）
        要求延迟低于16.66ms，否则会丢帧。
        """
        need_wait = self._enter_queue(0)
        
        if need_wait:
            event = self.waiting_queues[0][-1]
            event.wait()

        try:
            with self.lock:
                self.buffer[self.head] = item
                self.head = (self.head + 1) % self.size
                if self.head == self.tail:  # 缓冲区满，移动tail丢弃旧数据
                    self.tail = (self.tail + 1) % self.size
                self.available.notify_all()  # 唤醒等待的消费者
        finally:
            self._exit_and_schedule(0)

    def get_latest(self) -> Optional[object]:
        """
        定时消费者获取最新帧（非阻塞，线程安全）
        要求优先获取锁，保证闭环控制系统延迟最小化。
        """
        need_wait = self._enter_queue(1)
        
        if need_wait:
            event = self.waiting_queues[1][-1]
            event.wait()
        
        try:
            with self.lock:
                if self.head == self.tail: 
                    return None  # 缓冲区为空
                return self.buffer[(self.head - 1) % self.size] 
                # 由于返回的是数据引用，不需要担心数据从缓冲区中被移除的问题。
        finally:
            self._exit_and_schedule(1)


    def get_all(self) -> List[object]:
        """
        编码消费者获取所有有效帧（阻塞直到有新数据，线程安全）
        实时度最低，但要保证不丢帧。
        如果没有新数据，则等待最多1s。请间隔调用以避免无法获取或写入。
        """
        need_wait = self._enter_queue(2)
        
        if need_wait:
            event = self.waiting_queues[2][-1]
            event.wait()

        try:
            with self.lock:
                while self.head == self.tail:  # 等待新数据
                    self.available.wait(timeout=1)
                
                # 计算有效帧范围
                if self.head > self.tail:
                    frames = self.buffer[self.tail:self.head]
                else:
                    frames = self.buffer[self.tail:] + self.buffer[:self.head]
                self.tail = self.head  # 移动tail标记已处理
                return frames.copy()  # 返回拷贝避免外部修改影响缓冲区
        finally:
            self._exit_and_schedule(2)

class RingBuffer:
    '''
    A ring buffer class designed for one producer, two type consumer scenarios.
    Known Issue: when existing multiple producers, `popleft_till_latest` may always return None (cannot get lock).
    压力测试场景下，put和get_latest的性能表现较好。但是popleft_till_latest常常不能获取到锁（即使等待了）
    '''
    def __init__(self, size: int):
        """Initialize a ring buffer with fixed capacity.
        
        Args:
            size: Maximum number of items the buffer can hold.
        """
        self.buffer = [None] * size  # Underlying storage
        self.size = size             # Maximum capacity
        self.head = 0                # Producer write position (next insertion index)
        self.tail = 0                # Consumer read position (oldest valid item)
        self.items = 0               # Current number of items in buffer
        self.lock = Lock()           # Thread synchronization lock
        self.available = Condition(self.lock)  # Condition variable for consumer notification

    def put(self, item: Any) -> None:
        """Producer method to add an item to the buffer (thread-safe).
        
        Notes:
            - Designed for low-latency operations (<16.66ms per call)
            - If buffer is full, overwrites the oldest item (drop behavior)
            - Wakes up any waiting consumers after insertion
            
        Args:
            item: Data to be added to the buffer
        """
        with self.lock:
            if self.items == self.size:  # 缓存满，清除最早的数据（tail）
                self.tail = (self.tail + 1) % self.size
                self.items -= 1
            self.buffer[self.head] = item
            self.head = (self.head + 1) % self.size
            self.items += 1
            self.available.notify_all()  # 唤醒等待中的消费者

    def get_latest(self) -> Optional[object]:
        """Consumer method to get the most recent item (non-blocking, thread-safe).
        
        Notes:
            - Designed for minimal latency in control systems
            - Returns reference to the item (not a copy)
            - Caller should NOT modify mutable items to avoid corrupting buffer
            
        Returns:
            The newest item if available, None if buffer is empty
        """
        with self.lock:
            if self.items == 0:
                return None
            return self.buffer[(self.head - 1) % self.size]

    def popleft_till_latest(self, timeout: int = 1) -> Optional[List[object]]:
        """Consumer method to retrieve all items except the newest one (blocking, thread-safe).
        
        Notes:
            - Blocks until at least 2 items are available
            - Returns a copy of items to prevent external modification
            - Designed for batch processing where data integrity is critical
            - If only 1 item is available, please `put` an item to terminate waiting.
        
        Args:
            timeout: Maximum time to wait for new data (in seconds)
            
        Returns:
            List of all items except the most recent one
        """
        with self.lock:
            time_start = time.time()
            while (self.items <= 1) and (time.time() - time_start < timeout): # 仅剩最后1帧，不取。
                self.available.wait(timeout=timeout)   # 释放锁，允许对数据进行修改，以等待新数据
            
            if self.items <= 1:  # 超时或只有1帧
                return None
            # 计算有效帧范围
            popleft_end_idx = (self.head - 1) % self.size
            if popleft_end_idx > self.tail:
                frames = self.buffer[self.tail : popleft_end_idx]
            else:
                frames = self.buffer[self.tail:] + self.buffer[:popleft_end_idx]
            self.tail = popleft_end_idx  # 移动tail，等效于pop数据
            self.items = 1  # 保留最后一帧
            return frames  # 上述slice操作都会返回浅拷贝的新列表，无需再浅拷贝。