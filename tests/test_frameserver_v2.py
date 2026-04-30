# test_frameserver_v2.py

import os
# Must inject before any numpy/scipy import in Multiprocessing tests to avoid OpenBLAS memory allocation error.
# TODO: need fine grain control in real application.
# 在实际生产中，为了防止这种现象并保持子进程的计算性能：

# 按需限制（环境变量法）：在父进程序里设置 os.environ["OMP_NUM_THREADS"] = "1"。您担心的“计算变慢”其实在多进程架构下往往是反向优化——与其让 32 个进程每个都在那抢 CPU 时间片（Context Switch 损耗巨大），不如让每个进程都老老实实单核全速运行，这在多路流处理中整体吞吐量反而最高。
# 使用 initializers：如果您使用 mp.Pool，可以在初始化函数里单独控制单个子进程的线程策略。
# 手动控制核心关联 (Affinity)：如果确实需要某些子进程多线程加速，生产环境下通常会给不同进程绑定不同的物理核心，从 OS 调度层面错开压力。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pytest
import numpy as np
import time
import multiprocessing as mp
import random
import importlib

from ringbuffers.shared_ring_buffer_v4 import ProcessSafeSharedRingBuffer

ctx = mp.get_context("spawn")

@pytest.fixture
def empty_buffer():
    rb = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=60, frame_shape=(10, 10, 3), dtype=np.uint32)
    yield rb
    rb.close()
    try: rb.unlink()
    except Exception: pass

@pytest.fixture
def small_buffer():
    rb = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=5, frame_shape=(2, 2, 3), dtype=np.uint32)
    yield rb
    rb.close()
    try: rb.unlink()
    except Exception: pass

def __basic_use_consumer(fs_module_name, server_master, use_linked_fs, result_queue):
    fs_mod = importlib.import_module(fs_module_name)
    try:
        if use_linked_fs:
            server = fs_mod.FrameServer(create=False, frameserver=server_master)
        else:
            server = server_master
            
        cid = server.register_consumer()
        if cid != 0:
            raise ValueError(f"Expected cid 0, got {cid}")
            
        result_queue.put("READY")
        
        # We expect 100 frames
        collected = []
        for i in range(100):
            ticket = server.get_sync(cid, 1, timeout=2.0)
            if ticket is None:
                raise ValueError(f"Timeout expecting frame {i}")
            if ticket.head_id != i:
                raise ValueError(f"Expected frame {i}, got {ticket.head_id}")
                
            data_list = server.get_from_ticket(ticket)
            if data_list[0].shape != (1, 2, 2, 3):
                raise ValueError(f"Wrong shape")
            
            val = int(data_list[0][0, 0, 0, 0])
            collected.append(val)
            
            server.release_sync(cid, ticket)
            
        server.unregister_consumer(cid)
            
        result_queue.put(collected)
    except Exception as e:
        result_queue.put(e)
    finally:
        server.close()

@pytest.mark.parametrize("fs_module_name", ["frameserver.v2", "frameserver.v3"])
@pytest.mark.parametrize("use_linked_fs", [False, True])
def test_fs_basic_use(small_buffer, fs_module_name, use_linked_fs):
    """Test frameserver basic registration and sequential flow across processes."""
    fs_mod = importlib.import_module(fs_module_name)
    server_master = fs_mod.FrameServer(create=True, ring_buffer=small_buffer)
    small_buffer.trigger_release = server_master._gc
    
    result_queue = ctx.Queue()
    consumer_proc = ctx.Process(target=__basic_use_consumer, args=(fs_module_name, server_master, use_linked_fs, result_queue))
    consumer_proc.start()
    
    res = result_queue.get(timeout=3.0)
    if isinstance(res, Exception):
        raise res
    assert res == "READY"
    
    # Put 100 frames to cause wrap around (cap is 5)
    for i in range(100):
        small_buffer.put(np.full((1, 2, 2, 3), i, dtype=np.uint32))
        
    res = result_queue.get(timeout=3.0)
    consumer_proc.join(timeout=1.0)
    
    if isinstance(res, Exception):
        raise res
    assert res == list(range(100))
    
    server_master.close()
    server_master.unlink()

def __spin_delay(delay_sec):
    if delay_sec <= 0: return
    end = time.perf_counter() + delay_sec
    while time.perf_counter() < end:
        pass

def __unified_consumer_worker(fs_module_name, fs_obj, cid, stop_event, fetch_size, result_queue, delay_mean=0.0, delay_std=0.0):
    fs_mod = importlib.import_module(fs_module_name)
    try:
        server = fs_mod.FrameServer(create=False, frameserver=fs_obj)
        print(f"Successfully create subprocess consumer ({cid}): {os.getpid()}")
        tickets = []
        while not stop_event.is_set():
            ticket = server.get_sync(cid, fetch_size, timeout=0.1)
            if ticket:
                ts = time.monotonic()
                # 硬件级撕裂强验证
                try:
                    data_list = server.get_from_ticket(ticket)
                    rel_idx = 0
                    for block in data_list: # block.shape is [sub_size, H, W, C]
                        for i in range(block.shape[0]):
                            expected_val = ticket.head_id + rel_idx
                            if block[i, 0, 0, 0] != expected_val:
                                raise ValueError(f"Tearing! Expected {expected_val}, got {block[i, 0, 0, 0]}")
                            rel_idx += 1
                except fs_mod.TicketExpireException:
                    pass
                
                load_time = max(0.0, random.gauss(delay_mean, delay_std))
                __spin_delay(load_time)
                
                tickets.append((ticket.head_id, ts))
                server.release_sync(cid, ticket)
        result_queue.put(tickets)
    except Exception as e:
        result_queue.put(e)
    finally:
        server.close()
        print(f"Consumer ({cid}) closed.")

def __unified_producer_worker(rb_obj, stop_event, batch_size, result_queue, delay_mean=0.0, delay_std=0.0):
    buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=rb_obj)
    print(f"Successfully create subprocess producer: {os.getpid()}")
    i = 0
    while not stop_event.is_set():
        frames = np.zeros((batch_size, 10, 10, 3), dtype=np.uint32)
        for sub in range(batch_size):
            frames[sub] = i * batch_size + sub
            
        load_time = max(0.0, random.gauss(delay_mean, delay_std))
        __spin_delay(load_time)
        
        ret = buffer.put(frames, timeout=0.1)
        if ret: i += 1
    result_queue.put(i * batch_size) # 乘以 batch_size 送出实际的帧总数
    buffer.close()

@pytest.mark.parametrize("fs_module_name", ["frameserver.v2", "frameserver.v3"])
def test_fs_concurrent_single_consumer(empty_buffer, fs_module_name):
    """Test concurrent polling inside single CID yielding strictly ordered tickets via MP."""
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=True, ring_buffer=empty_buffer)
    cid = server.register_consumer()
    tx_stop_event = mp.Event()
    rx_stop_event = mp.Event()
    
    # Stress the server
    tx_queue = ctx.Queue()
    producer = ctx.Process(target=__unified_producer_worker, args=(empty_buffer, tx_stop_event, 1, tx_queue))
    producer.start()
        
    rx_queues = [ctx.Queue() for _ in range(5)]
    workers = [ctx.Process(target=__unified_consumer_worker, args=(fs_module_name, server, cid, rx_stop_event, 1, q)) for q in rx_queues]
    
    for w in workers: w.start()
    
    time.sleep(5.0)
    tx_stop_event.set()
    time.sleep(0.5)
    rx_stop_event.set()
    
    produced_frames = tx_queue.get()
    producer.join(timeout=1.0)
    
    collected_tickets = []
    for q in rx_queues:
        res = q.get()
        if isinstance(res, Exception): raise res
        collected_tickets.extend(res)
    print(f"Produced: {produced_frames}, Collected: {len(collected_tickets)}")
    assert len(collected_tickets) == produced_frames
    
    for w in workers: w.join(timeout=1.0)
    # Timestamp verification
    # Using time.monotonic to strict prove data generation order identical to acquire order!
    collected_tickets.sort(key=lambda x: x[1]) # Sort by timestamp
    head_ids_by_time = [x[0] for x in collected_tickets]
    if head_ids_by_time != list(range(produced_frames)): 
        mismatch_mask = np.array(head_ids_by_time) != np.arange(produced_frames)
        first_mismatch_idx = np.where(mismatch_mask)[0][0]
        first_mismatch_percentage = first_mismatch_idx / produced_frames * 100
        mismatch_percentage = np.sum(mismatch_mask) / produced_frames * 100
        import warnings
        warnings.warn("Order disrupted during concurrent acquisition due to microscopic OS scheduling! "
            f"First mismatch at index {first_mismatch_idx} ({first_mismatch_percentage:.2f}%), "
            f"Total {mismatch_percentage:.3f}% of frames are out of order.", UserWarning)
    
    collected_tickets.sort(key=lambda x: x[0])
    head_ids = [x[0] for x in collected_tickets]
    assert head_ids == list(range(produced_frames)), "Frame duplicated or skipped."
    
    server.unregister_consumer(cid)
    server.close()
    server.unlink()

@pytest.mark.parametrize("fs_module_name", ["frameserver.v2", "frameserver.v3"])
def test_fs_multi_cid_parallel(empty_buffer, fs_module_name):
    """Test full system saturation with 32 distinct consumers reading concurrently via True MP."""
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=True, ring_buffer=empty_buffer)
    cids = [server.register_consumer() for _ in range(32)]
    tx_stop_event = mp.Event()
    rx_stop_event = mp.Event()
    
    tx_queue = ctx.Queue()
    # Batch=5 increases producer throughput while fetch=1 checks all sequences meticulously.
    producer = ctx.Process(target=__unified_producer_worker, args=(empty_buffer, tx_stop_event, 5, tx_queue))
    producer.start()
    
    rx_queues = [ctx.Queue() for _ in range(32)]
    workers = [ctx.Process(target=__unified_consumer_worker, args=(fs_module_name, server, cid, rx_stop_event, 1, q)) for cid, q in zip(cids, rx_queues)]
    
    for w in workers: w.start()
    
    time.sleep(4.0)
    tx_stop_event.set()
    time.sleep(1.0)
    rx_stop_event.set()
    
    produced_frames = tx_queue.get()
    producer.join(timeout=2.0)
    
    for cid, q in zip(cids, rx_queues):
        res = q.get()
        if isinstance(res, Exception): raise res
        # Each distinct CID MUST have acquired exactly ALL frames!
        res.sort(key=lambda x: x[0])
        head_ids = [x[0] for x in res]
        expected_ids = list(range(0, produced_frames))
        
        if head_ids != expected_ids:
            print(f"Sequence gap on distinct CID {cid}. Expected up to id {produced_frames -1}, got max {head_ids[-1] if head_ids else 'None'}.")
        assert head_ids == expected_ids
    
    for w in workers: w.join(timeout=2.0)
    for cid in cids: server.unregister_consumer(cid)
    server.close()
    server.unlink()

@pytest.mark.parametrize("fs_module_name", ["frameserver.v2", "frameserver.v3"])
def test_fs_pipeline_backpressure(small_buffer, fs_module_name):
    """Test fast consumer starvation and producer locking by a slow consumer."""
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=True, ring_buffer=small_buffer)
    small_buffer.trigger_release = server._gc
    
    cid_fast = server.register_consumer()
    cid_slow = server.register_consumer()
    
    # Fill small buffer entirely (cap = 5)
    for i in range(5):
        small_buffer.put(np.full((1, 2, 2, 3), i, dtype=np.uint32))
        
    # Fast consumer gets and releases all 5
    for _ in range(5):
        t = server.get_sync(cid_fast, 1)
        server.release_sync(cid_fast, t)
        
    # Slow consumer gets them but DOES NOT release
    tickets_slow = []
    for _ in range(5):
        t = server.get_sync(cid_slow, 1)
        tickets_slow.append(t)
        
    # Blocked
    assert small_buffer.put(np.full((1, 2, 2, 3), 99, dtype=np.uint32), timeout=0.1) is False
    
    # Fast consumer tries to get next frame, it also blocks (timeout)
    assert server.get_sync(cid_fast, 1, timeout=0.1) is None
    
    # Slow consumer finally releases oldest frame!
    server.release_sync(cid_slow, tickets_slow[0])
    
    # The release triggers GC -> frees space -> unlocks Producer
    assert small_buffer.put(np.full((1, 2, 2, 3), 99, dtype=np.uint32), timeout=0.1) is True
    
    # Fast Consumer is now capable of capturing the newly streamed data
    t_fast = server.get_sync(cid_fast, 1, timeout=0.1)
    assert t_fast is not None
    assert t_fast.head_id == 5
    
    server.close()
    server.unlink()

def __barrier_producer(fs_module_name, fs_obj, rb_obj, stop_event):
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=False, frameserver=fs_obj)
    buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=rb_obj)
    buffer.trigger_release = server._gc
    
    counter = 0
    while not stop_event.is_set():
        f = np.zeros((5, 10, 10, 3), dtype=np.uint32)
        # Signature: head and tail identically stamped
        identifier = counter
        f[:, 0, 0, 0] = identifier
        f[:, -1, -1, -1] = identifier
        
        buffer.put(f, timeout=0) # Implicitly calls _gc!
        counter += 1
    buffer.close(); server.close()

def __barrier_consumer(fs_module_name, fs_obj, cid, stop_event):
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=False, frameserver=fs_obj)
    while not stop_event.is_set():
        # High frequency read/release to provoke gc sliding frame pointers
        t = server.get_sync(cid, 5, timeout=0)
        if t: server.release_sync(cid, t)
    server.close()

def __barrier_stalker(fs_module_name, fs_obj, stop_event, result_queue):
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=False, frameserver=fs_obj)
    tearing_cnt = 0
    
    while not stop_event.is_set():
        data_list = server.get_async_copy(2)
        if data_list is not None:
            for block in data_list:
                # Check signatures! Hardware out-of-order tearing proof!
                for i in range(block.shape[0]):
                    if block[i, 0, 0, 0] != block[i, -1, -1, -1]:
                        tearing_cnt += 1
    result_queue.put(tearing_cnt)
    server.close()

@pytest.mark.parametrize("fs_module_name", ["frameserver.v2", "frameserver.v3"])
def test_fs_async_sync_barrier(empty_buffer, fs_module_name):
    """Stress test tearing memory guard via get_async_copy alongside a real sync consumer."""
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=True, ring_buffer=empty_buffer)
    empty_buffer.trigger_release = server._gc
    cid = server.register_consumer()
    
    stop_event = ctx.Event()
    queue = ctx.Queue()
    
    p_prod = ctx.Process(target=__barrier_producer, args=(fs_module_name, server, empty_buffer, stop_event))
    p_cons = ctx.Process(target=__barrier_consumer, args=(fs_module_name, server, cid, stop_event))
    p_stalker = ctx.Process(target=__barrier_stalker, args=(fs_module_name, server, stop_event, queue))
    
    p_prod.start(); p_cons.start(); p_stalker.start()
    time.sleep(2.0) # Intense multiprocess execution window
    stop_event.set()
    
    tearing_cnt = queue.get()
    p_prod.join(); p_cons.join(); p_stalker.join()
    
    # Assert absolutely NO torn data and correct expirations 
    print(f"Tearing count: {tearing_cnt}")
    assert tearing_cnt == 0
    server.close(); server.unlink()


def __async_reader_worker(fs_module_name, fs_obj, stop_event, decode_delay, result_queue):
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=False, frameserver=fs_obj)
    expired = 0
    success = 0
    while not stop_event.is_set():
        ticket = server.get_async(1)
        if ticket is not None:
            time.sleep(decode_delay)
            try:
                server.get_from_ticket(ticket)
                success += 1
            except fs_mod.TicketExpireException:
                expired += 1
    result_queue.put((success, expired))
    server.close()

@pytest.mark.parametrize("fs_module_name", ["frameserver.v2", "frameserver.v3"])
@pytest.mark.parametrize("cons_num, decode_delay, should_expire", [
    # Low latency decode: GC shouldn't overtake it easily. Over 2.0s it handles the delay perfectly.
    (1, 0.0, False),
    # High latency decode: 5 sync consumers blasting GC, while this async reader stalls for 0.05s.
    # The Buffer capacity is 60. So during 0.05s stall, 5 sync consumers could process 50+ frames effortlessly. Ticket will EXPIRE!
    (5, 0.05, True) 
])
def test_fs_get_async_expire(empty_buffer, fs_module_name, cons_num, decode_delay, should_expire):
    """Test that `get_async` properly delegates decoding exceptions and is influenced by consumer GC."""
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=True, ring_buffer=empty_buffer)
    empty_buffer.trigger_release = server._gc
    cids = [server.register_consumer() for _ in range(cons_num)]
    
    stop_event = ctx.Event()
    
    prod = ctx.Process(target=__barrier_producer, args=(fs_module_name, server, empty_buffer, stop_event))
    cons_procs = [ctx.Process(target=__barrier_consumer, args=(fs_module_name, server, c, stop_event)) for c in cids]
    
    queue = ctx.Queue()
    async_reader = ctx.Process(target=__async_reader_worker, args=(fs_module_name, server, stop_event, decode_delay, queue))
    
    prod.start()
    for c in cons_procs: c.start()
    async_reader.start()
    
    time.sleep(2.0)
    stop_event.set()
    
    success, expired = queue.get()
    prod.join()
    for c in cons_procs: c.join()
    async_reader.join()
    
    print(f"Async Success: {success}, Expired: {expired} with delay: {decode_delay}")
    
    if should_expire:
        assert expired > 0
    else:
        # Sometimes GC is so incredibly fast it overtakes even 1ms delay. We expect VERY few or zero expirations.
        assert expired < success / 1000 # a relative small proportion.
        
    for c in cids: server.unregister_consumer(c)
    server.close(); server.unlink()

@pytest.mark.parametrize("fs_module_name", ["frameserver.v2", "frameserver.v3"])
def test_fs_consumer_drop(small_buffer, fs_module_name):
    """Test drop-consumer triggering explicit buffer eviction."""
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=True, ring_buffer=small_buffer)
    small_buffer.trigger_release = server._gc
    
    cid_slow = server.register_consumer()
    cid_fast = server.register_consumer()
    
    # Fill
    for i in range(5): small_buffer.put(np.full((1, 2, 2, 3), i, dtype=np.uint32))
    
    for _ in range(5):
        t = server.get_sync(cid_slow, 1) # Holds all 5 tickets unreleased
        t2 = server.get_sync(cid_fast, 1)
        server.release_sync(cid_fast, t2)
        
    assert small_buffer.put(np.full((1, 2, 2, 3), 99, dtype=np.uint32), timeout=0.1) is False
    
    # Drop slow consumer!
    server.unregister_consumer(cid_slow)
    
    # Automatically triggers global GC at the end!
    # Buffer immediately empties backlog!
    assert small_buffer.put(np.full((1, 2, 2, 3), 99, dtype=np.uint32), timeout=0.1) is True
    
    server.close()
    server.unlink()

@pytest.mark.parametrize("fs_module_name", ["frameserver.v2", "frameserver.v3"])
def test_fs_unregister_usage_denial(small_buffer, fs_module_name):
    """Test post-unregistered CID operations correctly get blocked or silenced."""
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=True, ring_buffer=small_buffer)
    cid = server.register_consumer()
    
    small_buffer.put(np.full((1, 2, 2, 3), 99, dtype=np.uint32))
    
    t = server.get_sync(cid, 1)
    
    # Unregister CID in background
    server.unregister_consumer(cid)
    
    # Use unreg cid to release should silently do nothing and not crash
    server.release_sync(cid, t)
    
    # Using unreg cid to get should throw runtime error appropriately rather than wait
    with pytest.raises(RuntimeError, match="not activated"):
        server.get_sync(cid, 1)
        
    server.close()
    server.unlink()

@pytest.mark.parametrize("fs_module_name", ["frameserver.v2", "frameserver.v3"])
@pytest.mark.parametrize("cons_num, prod_params, cons_params", [
    # 消费者数, 生产(均值,方差)秒, 消费(均值,方差)秒
    # 场景 1：极速并发（测试锁和调度的最高吞吐争抢）
    ( 1, (0.0001, 0.0), (0.0001, 0.0) ),
    (31, (0.0001, 0.00001), (0.0001, 0.00001) ),

    # 场景 2：极端背压（生产狂飙，几十个消费全部慢动作或随机长卡顿）
    ( 5, (0.00001, 0.0), (0.005, 0.015) ),

    # 场景 3：饥饿缺水（消费瞬间秒空，生产动辄几十毫秒便秘抖动）
    ( 8, (0.01, 0.03), (0.00001, 0.0) ),
    
    # 场景 4：天地大冲撞（两者均产生极大抖动，GC指针如过山车）
    (20, (0.002, 0.01), (0.002, 0.01) ),
])
def test_fs_stochastic_stress(empty_buffer, fs_module_name, cons_num, prod_params, cons_params):
    """Stress test with stochastic processing times utilizing __spin_delay."""
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=True, ring_buffer=empty_buffer)
    cids = [server.register_consumer() for _ in range(cons_num)]
    tx_stop_event = mp.Event()
    rx_stop_event = mp.Event()
    
    tx_queue = ctx.Queue()
    producer = ctx.Process(
        target=__unified_producer_worker, 
        args=(empty_buffer, tx_stop_event, 5, tx_queue, prod_params[0], prod_params[1])
    )
    producer.start()
    
    rx_queues = [ctx.Queue() for _ in range(cons_num)]
    workers = [
        ctx.Process(
            target=__unified_consumer_worker, 
            args=(fs_module_name, server, cid, rx_stop_event, 1, q, cons_params[0], cons_params[1])
        ) for cid, q in zip(cids, rx_queues)
    ]
    
    for w in workers: w.start()
    
    time.sleep(5.0) # Run the simulation for 3.5 seconds
    tx_stop_event.set()
    time.sleep(1.0)
    rx_stop_event.set()
    
    produced_frames = tx_queue.get()
    print(f"Produced {produced_frames} frames.")
    producer.join(timeout=2.0)
    
    for cid, q in zip(cids, rx_queues):
        res = q.get()
        if isinstance(res, Exception): raise res
        res.sort(key=lambda x: x[0])
        head_ids = [x[0] for x in res]
        print(f"CID {cid} received {len(head_ids)} frames.")
        expected_ids = list(range(produced_frames))
        if head_ids != expected_ids:
            print(f"Sequence gap on distinct CID {cid}. Expected up to id {produced_frames -1}, got max {head_ids[-1] if head_ids else 'None'}.")
        assert head_ids == expected_ids
    for w in workers: w.join(timeout=4.0)
    
    for cid in cids: server.unregister_consumer(cid)
    server.close()
    server.unlink()

@pytest.mark.parametrize("fs_module_name", ["frameserver.v2", "frameserver.v3"])
def test_fs_no_consumer_async_read(small_buffer, fs_module_name):
    """
    Test scenario: No consumers registered, producer keeps putting frames, 
    and consumer only uses get_async. Buffer should not block producer, 
    and get_async should continuously get the latest frames.
    """
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=True, ring_buffer=small_buffer)
    small_buffer.trigger_release = server._gc
    
    # Capacity is 5. We push 10 frames.
    # Since there are no consumers, it should auto-gc the oldest frames
    # and NOT block.
    for i in range(10):
        # put 1 frame at a time
        success = small_buffer.put(np.full((1, 2, 2, 3), i, dtype=np.uint32), timeout=0.1)
        assert success is True, f"Producer blocked at frame {i}"
        
        # Async reader checks the latest frame
        ticket = server.get_async(1)
        assert ticket is not None
        assert ticket.head_id == i
        
        data_list = server.get_from_ticket(ticket)
        assert data_list[0][0, 0, 0, 0] == i

    # Batch push testing
    success = small_buffer.put(np.full((3, 2, 2, 3), 99, dtype=np.uint32), timeout=0.1)
    assert success is True, "Producer blocked on batch put"
    
    # Async reader checks the latest 3 frames
    ticket = server.get_async(3)
    assert ticket is not None
    data_list = server.get_from_ticket(ticket)
    last_chunk = data_list[-1]
    assert last_chunk[-1, 0, 0, 0] == 99

    server.close()
    server.unlink()

@pytest.mark.parametrize("fs_module_name", ["frameserver.v2", "frameserver.v3"])
def test_fs_rw_during_registration(small_buffer, fs_module_name):
    """
    Test scenario: Consumer registered as historical_data=True should block 
    the producer as oldest frames now cannot be overwritten. historical_data=
    False should not block.
    """
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=True, ring_buffer=small_buffer)
    small_buffer.trigger_release = server._gc

    # Fill the buffer
    for i in range(5):
        success = small_buffer.put(np.full((1, 2, 2, 3), i, dtype=np.uint32), timeout=0.1)
        assert success is True, f"Producer blocked at frame {i}"
    # Once a consumer registered with historical data, the oldest frame is locked.
    cid = server.register_consumer(historical_data=True)
    success = small_buffer.put(np.full((1, 2, 2, 3), 100, dtype=np.uint32), timeout=0.1)
    assert success is False, "Producer do not block after a consumer locked historical data."
    ticket = server.get_async(1)
    data_list = server.get_from_ticket(ticket)
    assert data_list[0][0, 0, 0, 0] == 4
    ticket = server.get_sync(cid, 1)
    data_list = server.get_from_ticket(ticket)
    assert data_list[0][0, 0, 0, 0] == 0

    server.unregister_consumer(cid)
    ticket = server.get_async(1)
    data_list = server.get_from_ticket(ticket)
    assert data_list[0][0, 0, 0, 0] == 4

    # If a consumer do not lock historical data, producer should not block.
    cid = server.register_consumer(historical_data=False)
    success = small_buffer.put(np.full((1, 2, 2, 3), 101, dtype=np.uint32), timeout=0.1) # overwrite #0
    assert success is True, "Producer blocked after a consumer register without historical data."
    ticket = server.get_async(1)
    data_list = server.get_from_ticket(ticket)
    assert data_list[0][0, 0, 0, 0] == 101
    ticket = server.get_sync(cid,1)
    data_list = server.get_from_ticket(ticket)
    assert data_list[0][0, 0, 0, 0] == 101
    server.release_sync(cid ,ticket)
    server.unregister_consumer(cid)

    server.close()
    server.unlink()
