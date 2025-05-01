import pytest
import time
import threading
import statistics
from ringbuffer import RingBuffer

# @pytest.mark.benchmark
@pytest.mark.parametrize("buffer_size,producers,consumers,timing_consumers", [
    (1000, 1, 1, 1),    # 基础场景
    (10000, 10, 10, 10), # 高负载场景
    (100, 2, 8, 10)      # 消费者密集型
])
def test_ringbuffer_throughput(
        buffer_size, producers, consumers, timing_consumers, benchmark_cycle=1<<8, benchmark_duration=5):
    """多线程压力测试（集成到pytest）"""
    buffer = RingBuffer(buffer_size)
    stop_signal = threading.Event()
    stats = {'producers': [], 'consumers': [], 'timing_consumers': []}
    lock = threading.Lock()

    def timed_op(self, func, *args):
        start = time.perf_counter_ns()
        func(*args)
        return time.perf_counter_ns() - start

    def producer(pid):
        start = time.time()
        for i in range(benchmark_cycle):
            buffer.put(f"data-{pid}")
            # time.sleep(0.016)
            if stop_signal.is_set(): 
                print(f"Producer {pid} stopped at {i}")
                break
        stats['producers'].append(i / (time.time() - start))

    def consumer(cid):
        start = time.time()
        for i in range(benchmark_cycle):
            # time.sleep(0.5)
            items = buffer.popleft_till_latest()
            if items:
                print(f"Consumer{cid} success at {i}, got {len(items)} items.")
            if stop_signal.is_set(): break
        stats['consumers'].append(i / (time.time() - start))
    
    def timing_consumer(cid):
        start = time.time()
        for i in range(benchmark_cycle):
            buffer.get_latest()
            # time.sleep(1)
            if stop_signal.is_set(): break
        stats['timing_consumers'].append(i / (time.time() - start))

    # 启动线程
    threads = []
    for i in range(producers):
        t = threading.Thread(target=producer, args=(i,))
        t.start()
        threads.append(t)
    for i in range(consumers):
        t = threading.Thread(target=consumer, args=(i,))
        t.start()
        threads.append(t)
    for i in range(timing_consumers):
        t = threading.Thread(target=timing_consumer, args=(i,))
        t.start()
        threads.append(t)

    # 运行测试
    time.sleep(benchmark_duration)
    stop_signal.set()
    for t in threads:
        t.join()
    # for i in range(2*consumers):
    #     buffer.put(1) # 确保消费者退出循环（能取到数据）
    #     time.sleep(0.001)

    # 断言和输出（pytest -s会自动捕获打印内容）
    producer_avg = statistics.mean(stats['producers'])
    consumer_avg = statistics.mean(stats['consumers'])
    timing_consumer_avg = statistics.mean(stats['timing_consumers'])
    print(f"\nBufferSize: {buffer_size} | Producers: {producers} | Consumers: {consumers}")
    print(f"Producer Avg: {producer_avg:.2f} ops/sec")
    print(f"Consumer Avg: {consumer_avg:.2f} ops/sec")
    print(f"Timing Consumer Avg: {timing_consumer_avg:.2f} ops/sec")

    # 添加基本性能断言
    assert producer_avg > 10, "吞吐量过低" 
    assert timing_consumer_avg > 10, "访问速度过低"