import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pytest
import warnings
import numpy as np
import time
import multiprocessing as mp
import random

from ringbuffers.shared_ring_buffer_v4 import ProcessSafeSharedRingBuffer
from frameserver.v3 import FrameServer
from frameserver.v3 import TicketExpireException, MAX_LINKED_BUFFERS

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

@pytest.fixture
def multi_buffers():
    rbs = [ProcessSafeSharedRingBuffer(create=True, buffer_capacity=60, frame_shape=(10, 10, 3), dtype=np.uint32) for _ in range(4)]
    yield rbs
    for rb in rbs:
        rb.close()
        try: rb.unlink()
        except Exception: pass

def test_fs3_init(empty_buffer: ProcessSafeSharedRingBuffer):
    """Test initialize with a buffer with some data"""
    rb = empty_buffer
    for i in range(3):
        f=np.full((1,10,10,3), i, dtype=np.uint32)
        rb.put(f)
    rb.get(1) # read_ptr = 1, write_ptr = 3

    fs=FrameServer(create=True, ring_buffer=rb)

    cid = fs.register_consumer() # default: history_data=False
    ticket = fs.get_sync(cid, 1, timeout=0.1)
    assert ticket == None # no put after reg, no data available.
    fs.unregister_consumer(cid)

    cid = fs.register_consumer(historical_data=True)
    ticket = fs.get_sync(cid, 1)
    # fs's 1st consumer's 1st frame is id0
    assert ticket.head_id == 0 

    data = fs.get_from_ticket(ticket)
    assert data[0][0, 0, 0, 0] == 1 # 2nd frame produced.
    fs.release_sync(cid, ticket)
    fs.unregister_consumer(cid)

    cid = fs.register_consumer(historical_data=True)
    ticket = fs.get_sync(cid, 1)
    assert ticket.head_id == 1
    
    # 2nd frame
    data = fs.get_from_ticket(ticket)
    assert data[0][0, 0, 0, 0] == 2 

    fs.release_sync(cid, ticket)
    ticket = fs.get_sync(cid, 1, timeout=0.1)
    # no more frames in buffer
    assert ticket == None 
    fs.unregister_consumer(cid)
    
    fs.close(); fs.unlink()
    rb.close()

def test_fs3_init_sync(empty_buffer):
    """Test attaching an external buffer with some existing unread data properly aligns offsets."""
    rb_a = empty_buffer
    fs = FrameServer(create=True, ring_buffer=rb_a)
    cid = fs.register_consumer()

    for i in range(10):
        f = np.full((1, 10, 10, 3), i, dtype=np.uint32)
        rb_a.put(f)

    # Consumer reads 5 frames
    for _ in range(3):
        t = fs.get_sync(cid, 1)
        fs.release_sync(cid, t)

    rb_b = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=60, frame_shape=(10, 10, 3), dtype=np.uint32)
    for i in range(100, 105):
        f = np.full((1, 10, 10, 3), i, dtype=np.uint32)
        rb_b.put(f)
    rb_b.get(1)

    # Read ptr of rb_b is 1, it has 4 frames remaining.
    # Read ptr of rb_a is 3.
    # Current consumer frontier is 3.
    
    fs_b = FrameServer(create=False, ring_buffer=rb_b, frameserver=fs)
    
    for i in range(4):
        t_next = fs.get_sync(cid, 1) # i=0, Ticket head id should be 3
        assert t_next.head_id == 3 + i

        data_a = fs.get_from_ticket(t_next)
        assert data_a[0][0, 0, 0, 0] == 3 + i

        data_b = fs_b.get_from_ticket(t_next)
        assert data_b[0][0, 0, 0, 0] == 101 + i # i=0, Should get the first frame from rb_b!
        # TODO: i=5 buffer b out of bound. 
    
    fs.close(); fs.unlink()
    fs_b.close()
    rb_b.close(); rb_b.unlink()


def test_fs3_bind_limits(empty_buffer):
    """Test duplicate binding protection and slot limits."""
    fs = FrameServer(create=True, ring_buffer=empty_buffer)
    
    rb_new = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=60, frame_shape=(10, 10, 3), dtype=np.uint32)
    
    fs_child = FrameServer(create=False, ring_buffer=rb_new, frameserver=fs)
    
    with pytest.raises(RuntimeError, match="is already linked"):
        fs_dup = FrameServer(create=False, ring_buffer=rb_new, frameserver=fs)
    fs_child.close()
    
    rbs = []
    fs_children = []
    
    # We already have 1 (the original). Can attach MAX_LINKED_BUFFERS - 1 more.
    for i in range(MAX_LINKED_BUFFERS - 1):
        r = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=60, frame_shape=(10, 10, 3), dtype=np.uint32)
        rbs.append(r)
        fs_children.append(FrameServer(create=False, ring_buffer=r, frameserver=fs))
        
    r_overflow = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=60, frame_shape=(10, 10, 3), dtype=np.uint32)
    with pytest.raises(RuntimeError, match="Exceed the max number of linked buffers supported"):
        fs_overflow = FrameServer(create=False, ring_buffer=r_overflow, frameserver=fs)
        
    for child in fs_children: child.close()
    for r in rbs: r.close(); r.unlink()
    r_overflow.close(); r_overflow.unlink()
    rb_new.close(); rb_new.unlink()
    fs.close(); fs.unlink()

def __spin_delay(delay_sec):
    if delay_sec <= 0: return
    end = time.perf_counter() + delay_sec
    while time.perf_counter() < end:
        pass

def __unified_producer_worker(rb_obj, stream_id, stop_event, batch_size, result_queue, delay_mean=0.0, delay_std=0.0, timeout=0.05):
    buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=rb_obj)
    i = 0
    timeouts = 0
    while not stop_event.is_set():
        frames = np.zeros((batch_size, 10, 10, 3), dtype=np.uint32)
        for sub in range(batch_size):
            # Signature incorporates stream_id for verification
            frames[sub] = (stream_id * 1000000) + (i * batch_size + sub)
            
        load_time = max(0.0, random.gauss(delay_mean, delay_std))
        __spin_delay(load_time)
        
        ret = buffer.put(frames, timeout=timeout)
        if ret: 
            i += 1
        else:
            timeouts += 1
    result_queue.put({"produced": i * batch_size, "timeouts": timeouts})
    buffer.close()


def __multi_stream_consumer_worker(servers, cid, stop_event, fetch_size, result_queue, delay_mean=0.0, delay_std=0.0, lazy_release=False, slow_stream_idx=None):
    try:
        
        tickets = []
        while not stop_event.is_set():
            # Get ticket from master stream
            ticket = servers[0].get_sync(cid, fetch_size, timeout=0.1)
            if ticket:
                ts = time.monotonic()
                try:
                    # Validate data across ALL streams
                    for stream_id, s in enumerate(servers):
                        data_list = s.get_from_ticket(ticket, timeout=0.2)
                        if not data_list: 
                            if stop_event.is_set(): break
                            else: 
                                warnings.warn(f"Stream {stream_id} timed out, {ticket}", UserWarning)
                                continue
                        rel_idx = 0
                        for block in data_list:
                            for i in range(block.shape[0]):
                                expected_val = (stream_id * 1000000) + (ticket.head_id + rel_idx)
                                if block[i, 0, 0, 0] != expected_val:
                                    raise ValueError(f"Tearing on stream {stream_id}! Expected {expected_val}, got {block[i, 0, 0, 0]}")
                                rel_idx += 1
                except TicketExpireException:
                    pass
                
                load_time = max(0.0, random.gauss(delay_mean, delay_std))
                __spin_delay(load_time)
                
                tickets.append((ticket.head_id, ts))
                
                if lazy_release:
                    # Only release on some streams, or simulate massive delay for slow_stream
                    for idx, s in enumerate(servers):
                        if idx == slow_stream_idx:
                            pass # Don't release on the slow stream
                        else:
                            s.release_sync(cid, ticket)
                else:
                    for s in servers:
                        s.release_sync(cid, ticket)
                        
        result_queue.put(tickets)
    except Exception as e:
        result_queue.put(e)
    finally:
        for s in servers:
            s.close()

@pytest.mark.parametrize("n_streams, k_consumers", [
    (2, 4),
    (3, 2),
    (4, 8)
])
def test_fs3_multi_stream_concurrent_stress(multi_buffers, n_streams, k_consumers):
    """Stress test with n streams and k consumers validating absolute alignment."""
    rbs = multi_buffers[:n_streams]
    server_master = FrameServer(create=True, ring_buffer=rbs[0])
    rbs[0].trigger_release = server_master._gc
    
    # Attach all other streams
    servers = [server_master]
    for i in range(1, n_streams):
        s = FrameServer(create=False, ring_buffer=rbs[i], frameserver=server_master)
        rbs[i].trigger_release = s._gc
        servers.append(s)
        
    cids = [server_master.register_consumer() for _ in range(k_consumers)]
    
    tx_stop_event = mp.Event()
    rx_stop_event = mp.Event()
    
    tx_queues = [ctx.Queue() for _ in range(n_streams)]
    producers = []
    for i in range(n_streams):
        p = ctx.Process(target=__unified_producer_worker, args=(rbs[i], i, tx_stop_event, 2, tx_queues[i], 0.001, 0.0))
        producers.append(p)
        p.start()
        
    rx_queues = [ctx.Queue() for _ in range(k_consumers)]
    workers = []
    for cid, q in zip(cids, rx_queues):
        w = ctx.Process(target=__multi_stream_consumer_worker, args=(servers, cid, rx_stop_event, 1, q, 0.002, 0.0))
        workers.append(w)
        w.start()
        
    time.sleep(5.0)
    tx_stop_event.set()
    time.sleep(1.0)
    rx_stop_event.set()
    
    produced_counts = []
    for q in tx_queues:
        res = q.get()
        produced_counts.append(res["produced"])
        
    for p in producers: p.join()
    
    min_produced = min(produced_counts)
    
    for cid, q in zip(cids, rx_queues):
        res = q.get()
        if isinstance(res, Exception): raise res
        res.sort(key=lambda x: x[0])
        head_ids = [x[0] for x in res]
        expected_ids = list(range(min_produced))
        
        # In a concurrent environment, it's possible that the producers produced slightly different amounts 
        # before stopping. We assert that all tickets up to min_produced are correctly received.
        # But wait! consumers could have received tickets that are missing in some streams because producers stopped.
        # So we only assert intersection.
        if head_ids[:min_produced] != expected_ids:
            print(f"Sequence gap on distinct CID {cid}. Expected up to id {min_produced -1}.")
        assert head_ids[:min_produced] == expected_ids
        
    for w in workers: w.join()
    for c in cids: server_master.unregister_consumer(c)
    for s in servers: s.close()
    server_master.unlink()


def test_fs3_lazy_gc_backpressure(multi_buffers):
    """Test lazy GC backpressure. One stream is not released, causing its producer to block while others proceed."""
    rbs = multi_buffers[:2]
    server_master = FrameServer(create=True, ring_buffer=rbs[0])
    rbs[0].trigger_release = server_master._gc
    
    server_b = FrameServer(create=False, ring_buffer=rbs[1], frameserver=server_master)
    rbs[1].trigger_release = server_b._gc
    
    cid = server_master.register_consumer()
    
    tx_stop_event = mp.Event()
    rx_stop_event = mp.Event()
    
    tx_queues = [ctx.Queue() for _ in range(2)]
    
    p_fast = ctx.Process(target=__unified_producer_worker, args=(rbs[0], 0, tx_stop_event, 1, tx_queues[0], 0.0, 0.0, 0.05))
    p_slow = ctx.Process(target=__unified_producer_worker, args=(rbs[1], 1, tx_stop_event, 1, tx_queues[1], 0.0, 0.0, 0.05))
    
    rx_queue = ctx.Queue()
    # slow_stream_idx = 1
    w = ctx.Process(target=__multi_stream_consumer_worker, args=([server_master, server_b], cid, rx_stop_event, 1, rx_queue, 0.001, 0.0, True, 1))
    
    p_fast.start()
    p_slow.start()
    w.start()
    
    time.sleep(3.0)
    tx_stop_event.set()
    time.sleep(0.5)
    rx_stop_event.set()
    
    res_fast = tx_queues[0].get()
    res_slow = tx_queues[1].get()
    
    p_fast.join(); p_slow.join()
    print(f"Fast Stream Produced: {res_fast['produced']}, Timeouts: {res_fast['timeouts']}")
    print(f"Slow Stream Produced: {res_slow['produced']}, Timeouts: {res_slow['timeouts']}")
    
    res_cons = rx_queue.get()
    if isinstance(res_cons, Exception): raise res_cons
    
    w.join()
    # Fast stream should have produced a lot, slow stream blocked by capacity
    # buffer capacity is 60, so slow stream shouldn't produce much more than 60
    assert res_fast["produced"] >= 120 # first 60 frames must have released, later frames could be block by stream 1.
    assert res_slow["produced"] == 60
    assert res_slow["timeouts"] > 10
    
    server_master.unregister_consumer(cid)
    server_b.close()
    server_master.close()
    server_master.unlink()

def __malicious_attacher(fs_obj, rb_obj, stop_event, result_queue):
    """Constantly attaches and immediately drops to simulate crash/interruption."""
    success_count, rejected_count = 0, 0
    while not stop_event.is_set():
        try:
            s = FrameServer(create=False, ring_buffer=rb_obj, frameserver=fs_obj)
            success_count += 1
            time.sleep(0.01)
            if random.random() < 0.5:
                # Simulate a graceful close
                s.close()
            else:
                # Simulate an ungraceful crash - rely on python garbage collector / del to handle it
                # Or we just drop the reference to invoke __del__
                s = None 
        except Exception as e:
            rejected_count += 1
            pass
        time.sleep(0.05)
    result_queue.put((success_count, rejected_count))


def test_fs3_concurrent_lifecycle_interruption(empty_buffer, small_buffer):
    """Test attaching/detaching streams concurrently does not lock up the main stream A."""
    server_master = FrameServer(create=True, ring_buffer=empty_buffer)
    empty_buffer.trigger_release = server_master._gc
    cid = server_master.register_consumer()
    
    tx_stop_event = mp.Event()
    rx_stop_event = mp.Event()
    tx_queue = ctx.Queue()
    rx_queue = ctx.Queue()
    
    # Normal operations on main buffer
    p_prod = ctx.Process(target=__unified_producer_worker, args=(empty_buffer, 0, tx_stop_event, 1, tx_queue))
    p_cons = ctx.Process(target=__multi_stream_consumer_worker, args=([server_master], cid, rx_stop_event, 1, rx_queue, 0.001))
    
    # Malicious attachers on a second buffer
    ret_queue = [ctx.Queue() for _ in range(4)]
    malicious_procs = [ctx.Process(target=__malicious_attacher, args=(server_master, small_buffer, tx_stop_event, ret_queue[i])) for i in range(4)]
    
    p_prod.start()
    p_cons.start()
    for m in malicious_procs: m.start()
    
    time.sleep(4.0)
    tx_stop_event.set()
    time.sleep(1.0)
    rx_stop_event.set()
    
    attachment_stats = [ret_queue[i].get() for i in range(4)]
    for m in malicious_procs: m.join(timeout=2.0)

    produced = tx_queue.get()
    p_prod.join()
    
    res = rx_queue.get()
    if isinstance(res, Exception): raise res
    
    p_cons.join()

    # Assert malicious attachers worked fine
    for i in range(4): print(f"Attacker {i}: success: {attachment_stats[i][0]}, rejected: {attachment_stats[i][1]}")
    assert all(stat[0] > 0 for stat in attachment_stats)
    
    # Assert normal operations were not interrupted or deadlocked
    assert produced["produced"] > 100
    assert len(res) == produced["produced"]
    
    # Verify the reference counting recovered
    # There should only be 1 buffer linked (the master buffer)
    enabled_buffers = np.where(server_master._rb_linked_fs_count > 0)[0]
    assert len(enabled_buffers) == 1
    assert enabled_buffers[0] == 0
    
    server_master.unregister_consumer(cid)
    server_master.close()
    server_master.unlink()
