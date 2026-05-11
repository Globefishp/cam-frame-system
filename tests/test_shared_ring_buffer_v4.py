import pytest
import numpy as np
import time
import multiprocessing as mp
from ringbuffers.shared_ring_buffer_v4 import ProcessSafeSharedRingBuffer

ctx = mp.get_context("spawn")

def shm_keeper_process(buffer_obj, ready_event, stop_event):
    """Auxiliary process to keep SHM alive for tests on Windows."""
    ready_event.set()
    stop_event.wait()
    buffer_obj.close()

@pytest.fixture
def shm_buffer():
    """Standard empty buffer fixture."""
    buf = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=5, frame_shape=(2, 2, 3), dtype=np.uint8)
    yield buf
    buf.close()
    try: buf.unlink()
    except Exception: pass

@pytest.fixture
def shm_buffer_with_keeper():
    """Buffer fixture with a background keeper to test reattach safely."""
    buf = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=5, frame_shape=(2, 2, 3), dtype=np.uint8)
    ready_event = ctx.Event()
    stop_event = ctx.Event()
    
    keeper = ctx.Process(target=shm_keeper_process, args=(buf, ready_event, stop_event))
    keeper.start()
    ready_event.wait()
    
    yield buf
    
    stop_event.set()
    keeper.join()
    buf.close()
    try: buf.unlink()
    except Exception: pass

# ====================================================================================
# 拆分后的基础运转与刁钻边界测试 (取代原 test_v4_basic_use)
# ====================================================================================

def test_v4_basic_put_get_release(shm_buffer):
    """Test standard put, get, release linear logic without wrapping."""
    # Put 3 frames
    frames = np.full((3, 2, 2, 3), 11, dtype=np.uint8)
    assert shm_buffer.put(frames) is True
    assert shm_buffer.occupied_count_ == 3
    assert shm_buffer.unread_count() == 3
    
    # Get 2 frames
    data_list, ticket = shm_buffer.get(2)
    assert len(data_list) == 1
    assert data_list[0].shape[0] == 2
    assert np.all(data_list[0] == 11)
    
    # Check pointers after get (Data should remain occupied until released)
    assert shm_buffer.unread_count() == 1
    assert shm_buffer.occupied_count_ == 3
    assert shm_buffer.read_ptr == 2
    
    # Release the 2 frames read
    assert shm_buffer.release(ticket) == 2
    assert shm_buffer.occupied_count_ == 1

def test_v4_wrap_around_segmentation(shm_buffer):
    """Test memory wrap-around calculation for both put and get (Two-segment IO)."""
    # Push pointers to index 3 (Capacity is 5)
    shm_buffer.put(np.full((3, 2, 2, 3), 11, dtype=np.uint8))
    shm_buffer.get(3)
    shm_buffer.release(3)
    
    assert shm_buffer.write_ptr == 3
    assert shm_buffer.read_ptr == 3
    assert shm_buffer.occupied_count_ == 0
    
    # Put 4 frames: 2 frames at tail [3, 4], 2 frames at head[0, 1]
    assert shm_buffer.put(np.full((4, 2, 2, 3), 22, dtype=np.uint8)) is True
    assert shm_buffer.write_ptr == 2 # Wrapped to 2
    
    # Get 4 frames: should return a list of exactly 2 segment arrays
    data_list, ticket = shm_buffer.get(4)
    assert len(data_list) == 2
    assert data_list[0].shape[0] == 2 # from index 3 to 5 (tail)
    assert data_list[1].shape[0] == 2 # from index 0 to 2 (head)
    assert np.all(data_list[0] == 22)
    assert np.all(data_list[1] == 22)

def test_v4_ghost_read_prevention(shm_buffer):
    """
    [CRITICAL FIX TEST] 
    Verify that reading from a full buffer correctly resets the `is_full` flag,
    preventing `unread_count` from wrapping around to capacity falsely (Ghost Reads).
    """
    # Fill up the buffer exactly
    assert shm_buffer.put(np.full((5, 2, 2, 3), 33, dtype=np.uint8)) is True
    assert shm_buffer.is_full is True
    assert shm_buffer.unread_count() == 5
    
    # Read all data out
    data_list, ticket = shm_buffer.get(5)
    assert np.sum([len(frames) for frames in data_list]) == 5
    
    # After reading, unread count MUST be 0. 
    # If this fails and returns 5, the `is_full` clear logic in `get()` is broken.
    assert shm_buffer.unread_count() == 0
    
    # Getting 1 more frame should timeout instead of reading old data
    assert shm_buffer.get(1, timeout=0.1) is None

def test_v4_overwrite_prevention(shm_buffer):
    """
    [CRITICAL FIX TEST]
    Verify that `put` checks `occupied_count` instead of just `is_full` 
    to prevent overwriting unreleased data.
    """
    # Put 4 items (Buffer capacity is 5, 1 slot free)
    assert shm_buffer.put(np.full((4, 2, 2, 3), 44, dtype=np.uint8)) is True
    
    # Get 4 items (Moves read_ptr, but data is NOT released)
    shm_buffer.get(4)
    
    # Buffer is NOT 100% full (1 slot free), but if we try to put 2 items, it MUST block/timeout!
    # If this succeeds, it means `put` overwrote unreleased valid data.
    assert shm_buffer.put(np.full((2, 2, 2, 3), 55, dtype=np.uint8), timeout=0.1) is False
    
    # Release 3 items (Now 1 unreleased + 3 newly freed + 1 original free = 4 slots free)
    shm_buffer.release(3)
    
    # Now putting 2 items should successfully pass
    assert shm_buffer.put(np.full((2, 2, 2, 3), 66, dtype=np.uint8), timeout=0.1) is True

def test_v4_put_smaller_shape(shm_buffer):
    """
    Test that put() correctly handles frames smaller than the slot size,
    including scenarios where data wraps around the ring buffer.
    """
    # Buffer capacity=5, frame_shape=(2, 2, 3) -> 12 pixels/frame
    # Smaller shape (1, 2, 3) -> 6 pixels/frame
    
    # --- 1. Linear Put (No wrap-around) ---
    smaller_frames_1 = np.full((3, 1, 2, 3), 73, dtype=np.uint8)
    assert shm_buffer.put(smaller_frames_1) is True
    assert shm_buffer.occupied_count_ == 3
    
    data_list, _ = shm_buffer.get(3)
    full_slots = data_list[0]
    for i in range(3):
        assert np.array_equal(full_slots[i].flatten()[:6], smaller_frames_1[i].flatten())
    
    shm_buffer.release(3)
    assert shm_buffer.occupied_count_ == 0
    # Pointer is now at 3 (3 frames put, 3 released)
    assert shm_buffer.write_ptr == 3
    
    # --- 2. Wrap-around Put ---
    # Put 4 frames: will use slots 3, 4 (segment 1) and 0, 1 (segment 2)
    smaller_frames_2 = np.arange(4 * 6, dtype=np.uint8).reshape((4, 1, 2, 3)) + 10
    assert shm_buffer.put(smaller_frames_2) is True
    assert shm_buffer.occupied_count_ == 4
    assert shm_buffer.write_ptr == 2 # (3 + 4) % 5 = 2
    
    # Get all 4 frames (should return 2 segments due to wrap-around)
    data_list, _ = shm_buffer.get(4)
    assert len(data_list) == 2
    
    # Segment 1 (slots 3, 4)
    assert data_list[0].shape[0] == 2
    for i in range(2):
        assert np.array_equal(data_list[0][i].flatten()[:6], smaller_frames_2[i].flatten())
        
    # Segment 2 (slots 0, 1)
    assert data_list[1].shape[0] == 2
    for i in range(2):
        assert np.array_equal(data_list[1][i].flatten()[:6], smaller_frames_2[i+2].flatten())

    # --- 3. Error Case (Exceeds capacity) ---
    larger_frames = np.full((1, 3, 2, 3), 80, dtype=np.uint8)
    with pytest.raises(ValueError, match="exceeds slot capacity"):
        shm_buffer.put(larger_frames)


# ====================================================================================
# 多进程参数化测试：覆盖对象直接传递与 create=False 挂载模式
# ====================================================================================

def __child_test_worker_param(buf: ProcessSafeSharedRingBuffer, attach_mode: str, result_queue: mp.Queue):
    """Auxiliary worker for inter-process parameterized test."""
    worker_buf = None
    try:
        if attach_mode == "direct_pickle":
            # 模式 A: 依赖 __setstate__ 反序列化自动挂载
            worker_buf = buf 
        elif attach_mode == "create_false":
            # 模式 B: 显式调用 create=False 并通过 source_buffer 挂载
            worker_buf = ProcessSafeSharedRingBuffer(create=False, source_buffer=buf)
        else:
            raise ValueError("Unknown attach mode")

        # 1. 验证能否正确读出主进程写的数据
        assert worker_buf.occupied_count_ == 2
        data_list, ticket = worker_buf.get(2)
        assert np.all(data_list[0] == 88)
        
        # 2. 释放并写入新数据供主进程验证
        worker_buf.release(ticket)
        worker_buf.put(np.full((3, 2, 2, 3), 99, dtype=np.uint8))
        
        result_queue.put(True)
    except Exception as e:
        result_queue.put(e)
    finally:
        if worker_buf is not None:
            worker_buf.close()

@pytest.mark.parametrize("attach_mode",["direct_pickle", "create_false"])
def test_v4_inter_process_modes(shm_buffer_with_keeper, attach_mode):
    """
    Test two major ways of sharing the RingBuffer across processes.
    - direct_pickle: Sending `buf` directly, utilizing custom `__reduce__`/`__setstate__`.
    - create_false: Creating a new instance using `source_buffer=buf` parameter.
    """
    buf = shm_buffer_with_keeper
    
    # 主进程写入一些初始数据
    buf.put(np.full((2, 2, 2, 3), 88, dtype=np.uint8))
    
    queue = ctx.Queue()
    child = ctx.Process(target=__child_test_worker_param, args=(buf, attach_mode, queue))
    child.start()
    
    # 主进程主动关闭自己的引用（仅在直接传递模式下，因为我们需要验证反序列化）
    # 但由于有 keeper 进程，共享内存不会被操作系统回收
    buf.close()
    
    # 等待子进程完成任务
    child.join(timeout=3.0)
    assert not child.is_alive(), f"Child process stalled in mode: {attach_mode}"
    
    # 检查子进程结果
    res = queue.get(timeout=1.0)
    if isinstance(res, Exception):
        raise res
    assert res is True
    
    # 主进程重新挂载回来，验证子进程写回的数据
    # (此方法利用了你的 reattach 设计)
    buf.reattach()
    assert buf.occupied_count_ == 3
    data_list, ticket = buf.get(3)
    assert np.all(data_list[0] == 99)


def test_rb_overflow_and_timeout(shm_buffer):
    """Test buffer overflow wait and GC function injection."""
    # Fill exactly 5 frames (capacity)
    frames = np.full((5, 2, 2, 3), 22, dtype=np.uint8)
    assert shm_buffer.put(frames) is True
    
    # Another put should timeout
    single_frame = np.full((1, 2, 2, 3), 33, dtype=np.uint8)
    assert shm_buffer.put(single_frame, timeout=0.1) is False
    
    # Let's test gc_func triggering
    gc_triggered = []
    def fake_gc(num):
        gc_triggered.append(True)
        # GC frees 1 element
        shm_buffer.release(num)
        
    shm_buffer.trigger_release = fake_gc
    
    # Now it should trigger GC and then put should succeed
    assert shm_buffer.put(single_frame, timeout=0.1) is True
    assert len(gc_triggered) == 1


def test_rb_empty_and_timeout(shm_buffer):
    """Test empty buffer scenarios."""
    res = shm_buffer.get(1, timeout=0.1)
    assert res is None
    
    # Test block waiting for data
    def put_delayed():
        time.sleep(0.3)
        shm_buffer.put(np.full((1, 2, 2, 3), 44, dtype=np.uint8))
        
    import threading
    threading.Thread(target=put_delayed).start()
    
    # Block and wait
    start_time = time.monotonic()
    data_list, ticket = shm_buffer.get(1, timeout=1.0)
    end_time = time.monotonic()
    
    assert data_list is not None
    assert np.all(data_list[0] == 44)
    # verify timing
    assert 0.2 < (end_time - start_time) < 1.0


def test_rb_release_cap_truncation(shm_buffer):
    """Test over-release limits."""
    shm_buffer.put(np.full((2, 2, 2, 3), 55, dtype=np.uint8))
    
    # Release 10 frames when only 2 exist
    released = shm_buffer.release(10)
    assert released == 2
    assert shm_buffer.occupied_count_ == 0

    # Ensure buffer can still safely write and wrap correctly after emptying out
    shm_buffer.put(np.full((5, 2, 2, 3), 66, dtype=np.uint8))
    assert shm_buffer.occupied_count_ == 5
    data_list, ticket = shm_buffer.get(5)
    assert len(data_list) == 1 or len(data_list) == 2 # Could be wrapped
    assert np.all(data_list[0] == 66)


def test_rb_invalid_memory_attach():
    """Test OOM and OS errors during attach/create."""
    # OOM Test 
    with pytest.raises(RuntimeError):
        # 100 * 1024 * 1024 * 1024 frames of 1 byte = 100 TB.
        # Definitely throws ENOMEM or Overflow on valid logic.
        buf = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=100_000_000, frame_shape=(1000, 1000, 1000))
        
    # Attach to closed
    with pytest.raises(ValueError): # Source buffer missing names
        b1 = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=1, frame_shape=(1, 1, 1))
        b1.close()
        b1.unlink()
        # Invalid attach
        b2 = ProcessSafeSharedRingBuffer(create=False, source_buffer=b1)

def test_rb_closed_lifecycle(shm_buffer):
    """Test behavior specifically against closed objects without dedicated guards."""
    shm_buffer.close()
    
    # Access on closed obj. Will vary dynamically (Python vs C level Exceptions)
    with pytest.raises(Exception):
        shm_buffer.put(np.full((1, 2, 2, 3), 9, dtype=np.uint8))
        
    with pytest.raises(Exception):
        shm_buffer.get(1)
        
    with pytest.raises(Exception):
        shm_buffer.read_from(0, 1)

def test_rb_read_from_unconventional(shm_buffer):
    """Test manual read_from overriding normal safe limits but bounding indices."""
    # Write 3 frames
    shm_buffer.put(np.full((3, 2, 2, 3), 66, dtype=np.uint8))
    
    # We can read them back via read_from zero copy
    res = shm_buffer.read_from(0, 3)
    assert len(res) == 1
    assert np.all(res[0] == 66)
    
    # Buffer currently holds 3 items. Next index is 3.
    # What if we ask it to read from index 3 with length 3?
    # It wraps around boundaries. capacity=5. Index 3 to 1.
    res2 = shm_buffer.read_from(3, 3)
    assert len(res2) == 2 # Splitted! Length 2 at tail, Length 1 at head
    assert res2[0].shape[0] == 2
    assert res2[1].shape[0] == 1
    assert np.all(res2[1] == 66)
    # Note: the data itself is 'garbage' or uninitialized 0, but the method operates without crashing!


def __child_test_worker(buf: ProcessSafeSharedRingBuffer, result_queue: mp.Queue):
    """Auxiliary worker for inter-process check."""
    try:
        # Buffer is implicitly reconstructed and attached automatically via mp internals.
        assert buf.occupied_count_ == 1
        data_list, ticket = buf.get(1)
        assert np.all(data_list[0] == 77)
        buf.release(1)
        result_queue.put(True)
    except Exception as e:
        result_queue.put(e)
    finally:
        buf.close()

def test_v4_inter_process_reattach(shm_buffer_with_keeper):
    """Test that MP effectively sends buffer context to a child process."""
    buf = shm_buffer_with_keeper
    buf.put(np.full((1, 2, 2, 3), 77, dtype=np.uint8))
    
    queue = ctx.Queue()
    child = ctx.Process(target=__child_test_worker, args=(buf, queue))
    child.start()
    
    # We close main process copy intentionally!
    buf.close() 
    
    # Wait for child
    child.join(timeout=2.0)
    assert not child.is_alive(), "Child process stalled!"
    
    # Get outcome
    res = queue.get(timeout=1.0)
    if isinstance(res, Exception):
        raise res
    assert res is True

# TODO: 缺少create=False的验证
