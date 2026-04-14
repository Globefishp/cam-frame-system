# test_frameserver.py

# This file test the thread-safety of FrameServer and its broker.

import pytest
import threading
import time
import numpy as np
from typing import List

# 假设你的代码结构如下
from ringbuffers.shared_ring_buffer_v2a import ProcessSafeSharedRingBuffer
from frameserver import FrameServer

# ==========================================
# 测试夹具 (Fixtures)
# ==========================================

@pytest.fixture
def ring_buffer():
    """
    提供真实的共享内存 RingBuffer。
    保证每个测试用例结束后，共享内存被正确释放销毁，防止内存泄漏。
    """
    buffer_capacity = 60
    frame_shape = (2048, 2048, 1) # 模拟大分辨率单通道数据
    dtype = np.uint16
    
    rb = ProcessSafeSharedRingBuffer(
        create=True,
        buffer_capacity=buffer_capacity,
        frame_shape=frame_shape,
        dtype=dtype
    )
    yield rb
    
    rb.close()
    rb.unlink()

@pytest.fixture
def empty_buffer():
    """用于快速测试的小容量 Buffer"""
    rb = ProcessSafeSharedRingBuffer(
        create=True, buffer_capacity=10, frame_shape=(10, 10, 3), dtype=np.uint8
    )
    yield rb
    rb.close()
    rb.unlink()

# ==========================================
# 阶段 1：状态机与生命周期测试
# ==========================================

def test_start_and_stop(empty_buffer):
    server = FrameServer(empty_buffer)
    assert server.broker_running is False
    
    assert server.start() is True
    assert server.broker_running is True
    
    server.stop()
    assert server.broker_running is False

def test_multiple_start(empty_buffer):
    server = FrameServer(empty_buffer)
    server.start()
    thread_obj = server._broker_thread
    
    # 重复调用 start，不应该创建新线程
    server.start()
    server.start()
    assert server._broker_thread is thread_obj
    
    server.stop()

def test_rapid_start_stop(empty_buffer):
    server = FrameServer(empty_buffer)
    for _ in range(50):
        server.start()
        server.stop()
        assert server.broker_running is False

# ==========================================
# 阶段 2：严格同步消费者测试
# ==========================================

def test_single_strict_consumer(empty_buffer):
    server = FrameServer(empty_buffer)
    server.register_consumer("Encoder")
    server.start()
    
    # 放入一帧
    frame = np.zeros((1, 10, 10, 3), dtype=np.uint8)
    empty_buffer.put(frame)
    
    # 消费者获取
    data, count = server.get_sync(0)
    assert data is not None
    assert count == 1
    
    # 此时如果不 release，放入第二帧，Broker 也不应该增加 count
    empty_buffer.put(frame)
    time.sleep(0.1) # 给后台一点反应时间
    assert server._current_batch_count == 1 
    
    # 释放后，Broker 获取第二帧
    server.release_sync("Encoder", count)
    data2, count2 = server.get_sync(count)
    assert count2 == 2
    server.release_sync("Encoder", count2)
    
    server.stop()

def test_multiple_strict_consumers_blocking(empty_buffer):
    server = FrameServer(empty_buffer, batch_size=1)
    server.register_consumer("A")
    server.register_consumer("B")
    server.start()
    
    empty_buffer.put(np.zeros((1, 10, 10, 3), dtype=np.uint8))
    empty_buffer.put(np.zeros((1, 10, 10, 3), dtype=np.uint8))
    
    data_A1, count_A1 = server.get_sync(0)
    data_B1, count_B1 = server.get_sync(0)
    
    # A 释放，B 不释放
    server.release_sync("A", count_A1)
    time.sleep(0.05)
    
    # 验证 Broker 确实被 B 阻塞，没有去拿第二帧
    assert server._current_batch_count == 1
    
    # B 释放，系统恢复流转
    server.release_sync("B", count_B1)
    time.sleep(0.05)
    assert server._current_batch_count == 2
    
    server.release_sync("A", 2)
    server.release_sync("B", 2)
    server.stop()

# ==========================================
# 阶段 3：异步消费者测试 (Async Bypass Protocol)
# ==========================================

def test_async_copy_independence(empty_buffer):
    """
    验证 get_async_copy() 返回的是深拷贝，修改副本不应影响原始缓冲，
    且两者不共享内存地址。
    """
    server = FrameServer(empty_buffer)
    server.start()
    
    # 写入特定特征的数据 (全为 1)
    test_data = np.ones((1, 10, 10, 3), dtype=np.uint8)
    empty_buffer.put(test_data)
    
    # 等待 Broker 抓取数据
    time.sleep(0.05)
    
    # 获取副本
    copy_batch, count = server.get_async_copy()
    assert copy_batch is not None
    assert count == 1
    
    # 验证内存不共享
    # 注意：_current_batch 是 List[NDArray]，需要逐个校验
    internal_batch = server._current_batch
    for i in range(len(copy_batch)):
        assert not np.shares_memory(copy_batch[i], internal_batch[i]), f"Batch[{i}] should not share memory"
    
    # 修改副本，验证不影响内部原始数据
    copy_batch[0][0, 0, 0, 0] = 99
    assert internal_batch[0][0, 0, 0, 0] == 1, "Modifying copy should not affect internal buffer"
    
    server.stop()

def test_async_view_integrity(empty_buffer):
    """
    验证 get_async_view() 返回的是原始视图，两者应共享内存。
    """
    server = FrameServer(empty_buffer)
    server.start()
    
    empty_buffer.put(np.ones((1, 10, 10, 3), dtype=np.uint8))
    time.sleep(0.05)
    
    # 获取视图
    view_batch, count = server.get_async_view()
    internal_batch = server._current_batch
    
    assert view_batch is not None
    # 验证内存共享：修改视图，内部数据应同步变化
    view_batch[0][0, 0, 0, 0] = 123
    assert internal_batch[0][0, 0, 0, 0] == 123, "View should share memory with internal buffer"
    
    # 进一步通过 numpy 接口验证底层指针
    assert np.shares_memory(view_batch[0], internal_batch[0])
    
    server.stop()

def test_async_on_stopped_broker(empty_buffer):
    """
    验证在未启动 (Not Started) 或已停止 (Stopped) 时调用 Async API 的安全性。
    应返回 (None, last_id) 而非抛出异常。
    """
    server = FrameServer(empty_buffer)
    
    # 1. 测试未启动状态
    data_copy, count_copy = server.get_async_copy()
    data_view, count_view = server.get_async_view()
    
    assert data_copy is None
    assert data_view is None
    assert count_copy == 0
    
    # 2. 运行并停止
    server.start()
    empty_buffer.put(np.ones((1, 10, 10, 3), dtype=np.uint8))
    time.sleep(0.05)
    assert server._current_batch_count == 1
    
    server.stop()
    
    # 3. 测试停止后的状态
    # 此时 broker_running 为 False，虽然 count 是 1，但 data 必须返回 None
    data_after, count_after = server.get_async_copy()
    assert data_after is None
    assert count_after == 1 # ID 应该保留

# ==========================================
# 阶段 4：批量读取与优雅排空测试
# ==========================================

def test_batch_size_drainage(ring_buffer):
    """
    测试目的：
    验证 batch_size != 1 时，当 stop 被调用，能够正确排空不足一个 batch 的剩余帧。
    要求所有 Strict Consumer 都能收到最后这个大小不同的 Batch。
    """
    BATCH_SIZE = 5
    TOTAL_FRAMES = 12 # 将产生两个 5 帧的 batch，和最后一个 2 帧的 batch
    
    server = FrameServer(ring_buffer, batch_size=BATCH_SIZE)
    server.register_consumer("Consumer1")
    server.register_consumer("Consumer2")
    
    # 预先塞入所有数据
    frame = np.ones((1, 2048, 2048, 1), dtype=np.uint16)
    for _ in range(TOTAL_FRAMES):
        ring_buffer.put(frame)
        
    server.start()
    
    # 在 0.1 秒后调用 stop，模拟真实异步环境
    # 此时前 10 帧(2个Batch)应该被处理了，系统卡在等下一个 5 帧。
    # 收到 stop 后，应该把最后的 2 帧组成最后的 batch 送出来。
    def trigger_stop():
        time.sleep(0.1)
        server.stop()
        
    threading.Thread(target=trigger_stop).start()
    
    def consumer_routine(name, result_list):
        last_id = 0
        while True:
            data, current_id = server.get_sync(last_id)
            if data is None:
                break
            
            # 计算这个 batch 中包含的真实帧数 (处理环形内存跨界引起的 List[NDArray])
            frames_in_batch = sum(arr.shape[0] for arr in data)
            result_list.append(frames_in_batch)
            
            server.release_sync(name, current_id)
            last_id = current_id

    # 启动两个消费者线程
    c1_records = []
    c2_records =[]
    t1 = threading.Thread(target=consumer_routine, args=("Consumer1", c1_records))
    t2 = threading.Thread(target=consumer_routine, args=("Consumer2", c2_records))
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
    # 验证断言
    # 两个消费者都应该正好收到 3 个 Batch，帧数分别是 5, 5, 2
    assert c1_records ==[5, 5, 2], f"Consumer1 records mismatch: {c1_records}"
    assert c2_records ==[5, 5, 2], f"Consumer2 records mismatch: {c2_records}"
    
    # Broker 应该完全停止
    assert server.broker_running is False
    assert ring_buffer.unread_count == 0

# ==========================================
# 阶段 4：优雅排空与收尾测试 (极其严苛)
# ==========================================

def test_graceful_drain_no_data_loss(empty_buffer):
    server = FrameServer(empty_buffer)
    server.register_consumer("SlowConsumer")
    
    # 预先在底层塞入 8 帧数据
    frame = np.zeros((1, 10, 10, 3), dtype=np.uint8)
    for _ in range(8):
        empty_buffer.put(frame)
        
    server.start()
    
    # 在另外的线程瞬间触发 Stop
    def trigger_stop():
        time.sleep(0.05) # 稍微等 Broker 跑起来
        server.stop()
        
    threading.Thread(target=trigger_stop).start()
    
    # 验证消费者能否慢慢吃完所有残留数据
    total_consumed_batches = 0
    last_id = 0
    while True:
        data, current_id = server.get_sync(last_id)
        if data is None:
            break
        
        # 模拟极其缓慢的处理速度 (故意拖慢 Broker 退出时间)
        time.sleep(0.1) 
        server.release_sync("SlowConsumer", current_id)
        last_id = current_id
        total_consumed_batches += 1

    # 如果 get_sync 逻辑正确，消费者应该完美拿到这 8 帧分成的若干 Batch (根据 batch_size=5，应该是 2 个批次)
    assert total_consumed_batches >= 2 
    # 收尾后应完全退出
    assert server.broker_running is False

# ==========================================
# 阶段 5：极端并发压力测试 (性能与线程安全)
# ==========================================

@pytest.mark.parametrize("strict_num, async_num", [(2, 3), (1, 5)])
def test_multithread_chaos(ring_buffer, strict_num, async_num):
    print(f"\n--- Starting Chaos Test (Strict: {strict_num}, Async: {async_num}) ---")
    server = FrameServer(ring_buffer)
    
    for i in range(strict_num):
        server.register_consumer(f"Strict_{i}")
        
    server.start()
    
    stop_event = threading.Event()
    stats = {}
    
    # === 1. 高速生产者 (Provider) ===
    def provider_loop():
        frame = np.ones((1, 2048, 2048, 1), dtype=np.uint16)
        put_count = 0
        while not stop_event.is_set():
            # 疯狂写入，RingBuffer 满了自然会阻塞
            if ring_buffer.put(frame, timeout=0.01):
                put_count += 1
        stats['Provider_Frames'] = put_count

    # === 2. 严格消费者 ===
    def strict_consumer_loop(name):
        last_id = 0
        delays =[]
        consumed_batches = 0
        while not stop_event.is_set() or server.broker_running:
            t0 = time.perf_counter()
            data, current_id = server.get_sync(last_id)
            if data is None:
                break
            
            delays.append((time.perf_counter() - t0) * 1000) # 记录 Stall (ms)
            
            # 模拟计算负载 (加入随机抖动)
            # time.sleep(np.random.uniform(0.001, 0.005))
            
            server.release_sync(name, current_id)
            last_id = current_id
            consumed_batches += 1
            
        stats[name] = {'delays': delays, 'batches': consumed_batches}

    # === 3. 异步消费者 ===
    def async_consumer_loop(name):
        last_id = 0
        delays =[]
        while not stop_event.is_set():
            t0 = time.perf_counter()
            data, current_id = server.get_async_view()
            if current_id > last_id and data is not None:
                delays.append((time.perf_counter() - t0) * 1000)
                last_id = current_id
                time.sleep(np.random.uniform(0.005, 0.02)) # 异步消费者处理较慢
            else:
                time.sleep(0.001)
                
        stats[name] = {'delays': delays}

    # === 启动所有线程 ===
    threads =[]
    prod_t = threading.Thread(target=provider_loop)
    prod_t.start()
    threads.append(prod_t)
    
    for i in range(strict_num):
        t_obj = threading.Thread(target=strict_consumer_loop, args=(f"Strict_{i}",))
        t_obj.start()
        threads.append(t_obj)
        
    for i in range(async_num):
        t_obj = threading.Thread(target=async_consumer_loop, args=(f"Async_{i}",))
        t_obj.start()
        threads.append(t_obj)

    # === 测试持续时间 ===
    test_duration = 3.0 # 跑 3 秒的极限压测
    time.sleep(test_duration)
    
    # === 触发优雅收尾 ===
    stop_event.set() # 停止生产者和异步消费者
    server.stop()    # 停止 Broker，等待严格消费者吃完最后的缓冲
    
    for t_obj in threads:
        t_obj.join(timeout=5.0)
        assert not t_obj.is_alive(), "A thread is deadlocked!"
        
    # === 打印统计结果 ===
    print(f"\n[Test Results] Provider injected {stats['Provider_Frames']} frames.")
    for key, data in stats.items():
        if key == 'Provider_Frames': continue
        delays = data['delays']
        if not delays: continue
        avg_d = np.mean(delays)
        max_d = np.max(delays)
        std_d = np.std(delays)
        batches = data.get('batches', 'N/A')
        print(f"  {key: <10} | Batches: {str(batches):<4} | Delay(ms): Avg={avg_d:.2f}, Max={max_d:.2f}, Std={std_d:.2f}")

    assert server.broker_running is False

# TODO: 完善batch_size!=1时, 收尾测试的情况.