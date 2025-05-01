import pytest
from threading import Thread
from time import sleep
from ringbuffer import RingBuffer  # 替换your_module为你的模块名

class TestRingBuffer:
    def test_initialization(self):
        """测试缓冲区初始化"""
        rb = RingBuffer(5)
        assert rb.size == 5
        assert rb.items == 0
        assert rb.buffer == [None] * 5

    def test_put_and_get_latest(self):
        """测试基本插入和获取最新项"""
        rb = RingBuffer(3)
        rb.put(1)
        assert rb.get_latest() == 1
        rb.put(2)
        assert rb.get_latest() == 2

    def test_buffer_wrapping(self):
        """测试缓冲区环绕行为"""
        rb = RingBuffer(3)
        rb.put(1)
        rb.put(2)
        rb.put(3)  # 缓冲区满
        rb.put(4)  # 应该覆盖1
        assert rb.get_latest() == 4
        assert rb.items == 3
        assert rb.popleft_till_latest() == [2, 3]

    def test_popleft_till_latest(self):
        """测试批量获取功能"""
        rb = RingBuffer(5)
        rb.put(1)
        rb.put(2)
        rb.put(3)
        result = rb.popleft_till_latest()
        assert result == [1, 2]
        assert rb.get_latest() == 3
        assert rb.items == 1

    def test_thread_safety(self):
        """测试线程安全性"""
        rb = RingBuffer(100)
        results = []
        
        def consumer():
            items = rb.popleft_till_latest()
            results.extend(items)
        
        # 启动消费者线程(会阻塞等待数据)
        t = Thread(target=consumer)
        t.start()
        
        # 生产者线程(主线程)
        rb.put(1)
        rb.put(2)
        
        t.join()  # 等待消费者完成
        assert results == [1]

    def test_empty_buffer(self):
        """测试空缓冲区行为"""
        rb = RingBuffer(3)
        assert rb.get_latest() is None
    
    def test_timeout(self):
        """测试超时行为"""
        rb = RingBuffer(3)
        rb.put(1)
        assert rb.popleft_till_latest(timeout=0.1) is None  # 超时
        rb.put(2)  # 新数据
        assert rb.popleft_till_latest(timeout=0.1) == [1]  # 成功获取