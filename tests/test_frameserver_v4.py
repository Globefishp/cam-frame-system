# Sketched by Google Gemini 3.1 pro, reviewed & corrected by Haiyun Huang 2026

import os
# 按需限制（环境变量法）：在父进程序里设置以避免科学计算库启动多线程影响多进程测试
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pytest
import importlib
import time
import multiprocessing as mp
import numpy as np

ctx = mp.get_context("spawn")

@pytest.fixture
def ring_buffer_class(request):
    try:
        fs_mod_name = request.getfixturevalue("fs_module_name")
    except Exception:
        fs_mod_name = "frameserver.v4"
    fs_mod = importlib.import_module(fs_mod_name)
    return fs_mod.ProcessSafeSharedRingBuffer

@pytest.fixture
def empty_buffer(ring_buffer_class):
    rb = ring_buffer_class(create=True, buffer_capacity=60, frame_shape=(10, 10, 3), dtype=np.uint32)
    yield rb
    rb.close()
    try: rb.unlink()
    except Exception: pass

# ==============================================================================
# Bug 1: register_consumer and _gc race condition
# ==============================================================================

def __producer_worker(fs_module_name, fs_obj, rb_obj, stop_event):
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=False, frameserver=fs_obj)
    RBClass = fs_mod.ProcessSafeSharedRingBuffer
    buffer = RBClass(create=False, source_buffer=rb_obj)
    buffer.trigger_release = server._gc
    
    i = 0
    while not stop_event.is_set():
        f = np.zeros((1, 10, 10, 3), dtype=np.uint32)
        buffer.put(f, timeout=0.1)
        i += 1
        # Optional: yield time to encourage concurrency interleaved scheduling
        # time.sleep(0.001)
    buffer.close()
    server.close()

def __consumer_worker_a(fs_module_name, fs_obj, cid, stop_event):
    """ 常规消费者，疯狂读取并释放，触发GC以快速推进 _rb_oldest_frame_ids """
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=False, frameserver=fs_obj)
    while not stop_event.is_set():
        t = server.get_sync(cid, 1, timeout=0.1)
        if t:
            server.release_sync(t)
    server.close()

def __consumer_worker_b(fs_module_name, fs_obj, stop_event, result_queue):
    """ 频繁注册并读取首个ticket的消费者，这是出 Bug 的当事人 """
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=False, frameserver=fs_obj)
    
    errors = []
    successes = 0
    while not stop_event.is_set():
        try:
            cid = server.register_consumer(historical_data=True)
            t = server.get_sync(cid, 1, timeout=0.1)
            if t:
                try:
                    server.get_from_ticket(t)
                    successes += 1
                except fs_mod.TicketExpireException as e:
                    errors.append("TicketExpireException: " + str(e))
                finally:
                    server.release_sync(t)
            server.unregister_consumer(cid)
            if len(errors) > 0:
                break
        except Exception as e:
            errors.append("Unexpected Error: " + str(e))
            break
            
    result_queue.put({"successes": successes, "errors": errors})
    server.close()

@pytest.mark.parametrize("fs_module_name", ["frameserver.v3", "frameserver.v4"])
def test_fs_register_gc_race(empty_buffer, fs_module_name):
    """
    Test the race condition between register_consumer and _gc.
    Expected: v3 may fail with TicketExpireException, v4 should pass reliably.
    """
    fs_mod = importlib.import_module(fs_module_name)
    server_master = fs_mod.FrameServer(create=True, ring_buffer=empty_buffer)
    empty_buffer.trigger_release = server_master._gc
    
    cid_a = server_master.register_consumer(historical_data=True)
    
    stop_event = ctx.Event()
    result_queue = ctx.Queue()

    
    p_prod = ctx.Process(target=__producer_worker, args=(fs_module_name, server_master, empty_buffer, stop_event))
    p_prod.start()

    p_cons_a = []
    for _ in range(8): # Regular consumer
        p = ctx.Process(target=__consumer_worker_a, args=(fs_module_name, server_master, cid_a, stop_event))
        p.start()
        p_cons_a.append(p)
    
    # Frequently reg consumer
    p_cons_b = []
    for _ in range(3):
        p = ctx.Process(target=__consumer_worker_b, args=(fs_module_name, server_master, stop_event, result_queue))
        p.start()
        p_cons_b.append(p)
    
    time.sleep(10.0) # Run stress test for 3 seconds
    stop_event.set()
    
    res = result_queue.get()
    
    p_prod.join(timeout=2.0)
    for p in p_cons_a:
        p.join(timeout=2.0)
    for p in p_cons_b:
        p.join(timeout=2.0)
    
    server_master.unregister_consumer(cid_a)
    server_master.close()
    server_master.unlink()
    
    if fs_module_name == "frameserver.v3":
        if len(res["errors"]) > 0:
            pytest.xfail(f"v3 correctly reproduced the race condition: {res['errors'][0]}")
        else:
            pytest.fail("v3 failed to reproduce the race condition within the timeframe, this is stochastic but expected.")
    else:
        assert len(res["errors"]) == 0, f"v4 failed with errors: {res['errors']}"
        print(f"v4 successful iterations without error: {res['successes']}")


# ==============================================================================
# Bug 2: get_sync lost wakeup condition
# ==============================================================================

def __lost_wakeup_producer(fs_module_name, fs_obj, rb_obj, ready_event, stop_event, result_queue):
    fs_mod = importlib.import_module(fs_module_name)
    RBClass = fs_mod.ProcessSafeSharedRingBuffer
    buffer = RBClass(create=False, source_buffer=rb_obj)
    
    import random
    produced_times = []

    ready_event.wait(); time.sleep(0.01)
    
    i = 0
    while not stop_event.is_set():
        # Slow producer, delay randomly between 0.008 and 0.015s
        delay = random.uniform(0.008, 0.015)
        wake_time = time.perf_counter() + delay
        
        f = np.zeros((1, 10, 10, 3), dtype=np.uint32)
        f[0, 0, 0, 0] = i # stamp with index
        if buffer.put(f, timeout=0.1):
            produced_times.append((i, time.perf_counter()))
            i += 1
        time.sleep(wake_time - time.perf_counter())
            
    result_queue.put(produced_times)
    buffer.close()

def __lost_wakeup_consumer(fs_module_name, fs_obj, cid, ready_event, stop_event, result_queue):
    fs_mod = importlib.import_module(fs_module_name)
    server = fs_mod.FrameServer(create=False, frameserver=fs_obj)
    
    ready_event.set()

    consumed_times = []
    while not stop_event.is_set():
        # Fast consumer with slow timeout (0.01s)
        # If timeout occurs, consumer waits for full 0.01s instead of being woken up
        t = server.get_sync(cid, 1, timeout=0.01)
        now = time.perf_counter()
        if t:
            data = server.get_from_ticket(t)
            idx = data[0][0, 0, 0, 0]
            consumed_times.append((idx, now))
            server.release_sync(t)
            
    result_queue.put(consumed_times)
    server.close()

@pytest.mark.parametrize("fs_module_name", ["frameserver.v3", "frameserver.v4"])
def test_fs_get_sync_lost_wakeup(empty_buffer, fs_module_name):
    """
    Test the lost wakeup bug in get_sync where condition.wait might miss notify_all.
    """
    fs_mod = importlib.import_module(fs_module_name)
    server_master = fs_mod.FrameServer(create=True, ring_buffer=empty_buffer)
    empty_buffer.trigger_release = server_master._gc
    
    cid = server_master.register_consumer(historical_data=True)
    
    ready_event= ctx.Event()
    stop_event = ctx.Event()
    prod_queue = ctx.Queue()
    cons_queue = ctx.Queue()
    
    p_prod = ctx.Process(target=__lost_wakeup_producer, args=(fs_module_name, server_master, empty_buffer, ready_event, stop_event, prod_queue))
    p_cons = ctx.Process(target=__lost_wakeup_consumer, args=(fs_module_name, server_master, cid, ready_event, stop_event, cons_queue))
    
    p_prod.start()
    p_cons.start()
    
    time.sleep(10.0) # run for 10 seconds
    stop_event.set()
    
    prod_times = prod_queue.get()
    cons_times = cons_queue.get()
    
    p_prod.join(timeout=2.0)
    p_cons.join(timeout=2.0)
    
    server_master.unregister_consumer(cid)
    server_master.close()
    server_master.unlink()
    
    # Calculate delay metrics
    prod_dict = dict(prod_times)
    cons_dict = dict(cons_times)
    
    delays = []
    for idx, c_time in cons_dict.items():
        if idx in prod_dict:
            p_time = prod_dict[idx]
            delay = c_time - p_time
            delays.append(delay)
            
    delays = np.array(delays)
    timeout_threshold = 0.009 # 0.01s timeout, if delay > 0.009s it means it waited for timeout despite data being produced
    
    if len(delays) > 0:
        timed_out_count = np.sum(delays > timeout_threshold)
        print(f"Total produced: {len(prod_times)}, Total processed: {len(delays)}")
        print(f"Max delay: {np.max(delays):.4f}s, mean delay: {np.mean(delays):.4f}s")
        print(f"Delays > {timeout_threshold}s: {timed_out_count}")
        print(f"  Delay pos: {np.where(delays > timeout_threshold)[0]}")
        print(f"  Delay times: {delays[delays > timeout_threshold]}")
    else:
        pytest.fail("No frames processed during test!")
        
    if fs_module_name == "frameserver.v3":
        if timed_out_count > 0:
            pytest.xfail(f"v3 correctly reproduced the lost wakeup condition: {timed_out_count} frames were significantly delayed.")
        else:
            pytest.fail("v3 failed to reproduce lost wakeup within timeframe.")
    else:
        assert timed_out_count == 0, f"v4 has {timed_out_count} frames delayed more than {timeout_threshold}s, lost wakeup still exists!"
