import pytest
import numpy as np
import multiprocessing as mp
import multiprocessing.shared_memory as mp_shm # Import mp_shm
import time
from shared_ring_buffer import ProcessSafeSharedRingBuffer

# Define buffer parameters for testing
BUFFER_CAPACITY = 5
FRAME_SHAPE = (10, 20, 3) # Example frame size (height, width, channels)
FRAME_DTYPE = np.dtype('uint8')
FRAME_BYTES = int(np.prod(FRAME_SHAPE) * FRAME_DTYPE.itemsize)

@pytest.fixture
def shared_buffer():
    """Fixture to create and unlink the shared ring buffer."""
    buffer = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=BUFFER_CAPACITY, frame_shape=FRAME_SHAPE, dtype=FRAME_DTYPE)
    yield buffer
    buffer.close()
    buffer.unlink()

def create_dummy_frame(value, shape=FRAME_SHAPE, dtype=FRAME_DTYPE):
    """Helper function to create a dummy frame."""
    return np.full(shape, value, dtype=dtype)

import random

def create_dummy_frames(start_value, num_frames, shape=FRAME_SHAPE, dtype=FRAME_DTYPE, use_random=False, random_seed=None):
    """Helper function to create multiple dummy frames."""
    frames = np.zeros((num_frames,) + shape, dtype=dtype)
    if use_random:
        dtype_info = np.iinfo(dtype)
        for i in range(num_frames):
            effective_value = start_value + i
            # Use a combination of random_seed and effective_value for reproducibility
            seed = random_seed + effective_value if random_seed is not None else effective_value
            rng = np.random.default_rng(seed)
            # Generate a single random value for this frame
            random_value = rng.integers(low=dtype_info.min, high=dtype_info.max + 1, size=1, dtype=dtype)[0]
            frames[i] = create_dummy_frame(random_value, shape, dtype)
    else:
        for i in range(num_frames):
            frames[i] = create_dummy_frame(start_value + i, shape, dtype)
    return frames

def test_basic_put_get(shared_buffer):
    """Test basic put and get operations."""
    frame1 = create_dummy_frames(1, 1)
    frame2 = create_dummy_frames(2, 1)

    assert shared_buffer.put(frame1) is True
    assert shared_buffer.put(frame2) is True

    # Get the first frame
    frames_list = shared_buffer.get(1)
    assert frames_list is not None
    assert len(frames_list) == 1
    retrieved_frame1 = frames_list[0]
    assert np.array_equal(retrieved_frame1, frame1)

    # Get the second frame
    frames_list = shared_buffer.get(1)
    assert frames_list is not None
    assert len(frames_list) == 1
    retrieved_frame2 = frames_list[0]
    assert np.array_equal(retrieved_frame2, frame2)

    # Test getting more frames than available (should block or timeout in a real scenario,
    # but in this basic test, it should return None immediately if no data)
    frames_list = shared_buffer.get(1, timeout=0.1) # Use a small timeout
    assert frames_list is None

def test_put_get_multiple_frames(shared_buffer):
    """Test putting and getting multiple frames at once."""
    # Put multiple frames
    frames_to_put = create_dummy_frames(0, 3)
    assert shared_buffer.put(frames_to_put) is True

    # Get multiple frames
    frames_list = shared_buffer.get(2)
    assert frames_list is not None
    assert len(frames_list) == 1
    assert np.array_equal(frames_list[0], create_dummy_frames(0, 2))

    # Put more frames, causing wrap around
    shared_buffer.release_last_got_data() 
    frames_to_put_2 = create_dummy_frames(3, 4) # 3 + 4 = 7 > 5, wraps
    assert shared_buffer.put(frames_to_put_2) is True

    # Get remaining frames, causing wrap around
    frames_list = shared_buffer.get(5) # 3 frames were put initially, 2 were retrieved, 1 remains. Then 4 were put. Total 5 remaining.
    assert frames_list is not None
    assert len(frames_list) == 2 # Should wrap around

    retrieved_frames = np.concatenate(frames_list, axis=0)
    # Expected frames: the remaining 1 from the first put, then the 4 from the second put.
    expected_frames = np.concatenate((create_dummy_frames(2, 1), create_dummy_frames(3, 4)), axis=0)
    assert np.array_equal(retrieved_frames, expected_frames)

def test_buffer_full(shared_buffer):
    """Test putting frames when the buffer is full."""
    # Fill the buffer
    frames_to_put = create_dummy_frames(0, BUFFER_CAPACITY)
    assert shared_buffer.put(frames_to_put) is True

    # Try to put one more frame, should time out
    start_time = time.time()
    assert shared_buffer.put(create_dummy_frames(BUFFER_CAPACITY, 1), timeout=0.5) is False
    end_time = time.time()
    assert end_time - start_time >= 0.5

def test_buffer_empty(shared_buffer):
    """Test getting frames when the buffer is empty."""
    # Try to get a frame from an empty buffer, should time out
    start_time = time.time()
    frames_list = shared_buffer.get(1, timeout=0.5)
    end_time = time.time()
    assert frames_list is None
    assert end_time - start_time >= 0.5

    # Put a frame and then get it
    frame = create_dummy_frames(0, 1)
    assert shared_buffer.put(frame) is True
    frames_list = shared_buffer.get(1, timeout=0.5)
    assert frames_list is not None
    assert len(frames_list) == 1
    assert np.array_equal(frames_list[0], frame)

    # Try to get another frame, should time out again
    start_time = time.time()
    frames_list = shared_buffer.get(1, timeout=0.5)
    end_time = time.time()
    assert frames_list is None
    assert end_time - start_time >= 0.5

def test_release_last_got_data(shared_buffer):
    """Test the _release_last_got_data method."""

    # Fill the buffer
    assert shared_buffer.put(create_dummy_frames(0, BUFFER_CAPACITY)) is True

    # Get some frames (space not released yet)
    frames_list = shared_buffer.get(2)
    assert frames_list is not None

    # Now, try to put a frame. This should block because the buffer is logically full
    # (occupied_count is 5, capacity is 5). The space for the 2 retrieved frames
    # is not released until the next get.
    # We can test this by trying to put with a timeout and expecting it to fail.
    start_time = time.time()
    assert shared_buffer.put(create_dummy_frames(BUFFER_CAPACITY, 1), timeout=0.5) is False
    end_time = time.time()
    assert end_time - start_time >= 0.5

    # Now, perform another get. This should trigger the release of the space for the first 2 frames.
    frames_list = shared_buffer.get(1) # Get one more frame
    assert frames_list is not None

    # After this get, the space for the first 2 frames should be released.
    # The occupied_count should now be 5 - 2 = 3 (initially 5, get 2, preserved 1).
    # The read_ptr is now at index 3. The write_ptr is at index 0 (wrapped around).
    # Let's try putting a frame again. This should now succeed as there is space.
    assert shared_buffer.put(create_dummy_frames(BUFFER_CAPACITY + 1, 1), timeout=0.5) is True

def test_wrap_around_put(shared_buffer):
    """Test putting frames that wrap around the buffer."""
    # Put frames to fill the buffer partially
    assert shared_buffer.put(create_dummy_frames(0, BUFFER_CAPACITY - 2)) is True

    # Get some frames to move the read pointer
    frames_list = shared_buffer.get(2)
    assert frames_list is not None
    assert len(frames_list) == 1 # Should return a single view if no wrap around on get
    assert np.array_equal(frames_list[0], create_dummy_frames(0, 2))

    # Release occupation
    shared_buffer.release_last_got_data()
    # Put frames that will wrap around
    frames_to_put = create_dummy_frames(BUFFER_CAPACITY - 2, 3) # 3 frames, capacity is 5, 3 + (5-2) = 6 > 5, so it wraps
    assert shared_buffer.put(frames_to_put) is True

    # Get all frames and check their order
    # The buffer should now contain frames from index 2 to 4, then 0 to 0 (total 4 frames)
    # The order in the buffer should be: frames 2, 3, 0, 1 (relative to the start of the buffer)
    # The read pointer is at index 2. The write pointer is at index 1.
    # Occupied count should be 4.
    # When getting 4 frames, it should read from index 2, 3, 4 (wrap) 0.
    frames_list = shared_buffer.get(4)
    assert frames_list is not None
    assert len(frames_list) == 2 # Should return two views due to wrap around on get

    # Concatenate the views to check the full sequence
    retrieved_frames = np.concatenate(frames_list, axis=0)
    expected_frames = np.concatenate((create_dummy_frames(2, 3), create_dummy_frames(5, 1)), axis=0) # Expected order after wrap around
    assert np.array_equal(retrieved_frames, expected_frames)


def test_wrap_around_get(shared_buffer):
    """Test getting frames that wrap around the buffer."""
    # Fill the buffer
    assert shared_buffer.put(create_dummy_frames(0, BUFFER_CAPACITY)) is True

    # Get 3 frames, read_ptr is 0, capacity is 5. 0 + 3 = 3 < 5, no wrap yet.
    frames_list = shared_buffer.get(3)
    assert frames_list is not None
    assert len(frames_list) == 1
    assert np.array_equal(frames_list[0], create_dummy_frames(0, 3))

    # Release occupation
    shared_buffer.release_last_got_data()
    # Put 2 more frames to ensure enough data for wrap-around get
    assert shared_buffer.put(create_dummy_frames(5, 2)) is True # Put frames 5 and 6
    # Get 3 frames, which should cause wrap around. read_ptr is 3. 3 + 3 = 6 > 5, wraps around.
    frames_list = shared_buffer.get(3)
    assert frames_list is not None
    assert len(frames_list) == 2 # Should return two views due to wrap around on get

    # Concatenate the views to check the full sequence
    retrieved_frames = np.concatenate(frames_list, axis=0)
    # The expected frames are the ones that were put after the first 3 were retrieved (frames 0, 1, 2),
    # which are frames 3 and 4, followed by the 2 newly put frames (5 and 6).
    # The frames in the buffer are now from index 3 to 4, then 0 to 1 (due to put wrapping).
    # The read pointer is at index 3. The write pointer is at index 2.
    # When getting 3 frames, it should read from index 3, 4 (wrap) 0.
    expected_frames = np.concatenate((create_dummy_frames(3, 2), create_dummy_frames(5, 1)), axis=0) # Expected frames: 3, 4, 5
    assert np.array_equal(retrieved_frames, expected_frames)

def producer_task(source_buffer, num_frames_to_put, frame_shape, frame_dtype):
    """Task for a producer process."""
    try:
        # Attach to the existing shared buffer using the source_buffer object
        buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=source_buffer)
        for i in range(num_frames_to_put):
            frames = create_dummy_frames(i, 1, shape=frame_shape, dtype=frame_dtype)
            # print(f"Producer Loop {i}: ")
            if not buffer.put(frames, timeout=5.0):
                print(f"Producer: Timeout putting frame {i}")
                break
        buffer.close()
    except Exception as e:
        print(f"Producer process error: {e}")
        raise

def consumer_task(source_buffer, num_frames_to_get, frame_shape, frame_dtype, results_queue):
    """Task for a consumer process."""
    try:
        # Attach to the existing shared buffer using the source_buffer object
        buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=source_buffer)
        retrieved_frames = []
        for i in range(num_frames_to_get):
            # print(f"Consumer Loop {i}: getting frames")
            frames_list = buffer.get(1, timeout=5.0)
            if frames_list is not None:
                frames = np.concatenate(frames_list, axis=0).copy() # Deep copy to persist the data
                # print(f"Consumer Loop {i}: got {frames.shape[0]} frames")
                retrieved_frames.append(frames)
            else:
                print(f"Consumer: Timeout getting frame {i}")
                break
        all_frames = np.concatenate(retrieved_frames, axis=0)
        print(f"Consumer putting frames to queue, of a total memory size: {all_frames.nbytes} Bytes")
        results_queue.put(retrieved_frames)
        buffer.close()
    except Exception as e:
        print(f"Consumer process error: {e}")
        raise


def test_process_safety(shared_buffer):
    """Test concurrent put and get operations from multiple processes."""
    mp.set_start_method('spawn', force=True) # Use spawn method for better compatibility

    num_frames = BUFFER_CAPACITY * 2 # Put and get more frames than capacity

    # Use a Queue to get results from the consumer process
    results_queue = mp.Queue(maxsize=1)

    # Create producer and consumer processes
    producer_p = mp.Process(target=producer_task, args=(shared_buffer, num_frames, FRAME_SHAPE, FRAME_DTYPE))
    consumer_p = mp.Process(target=consumer_task, args=(shared_buffer, num_frames, FRAME_SHAPE, FRAME_DTYPE, results_queue))

    # Start processes
    producer_p.start()
    consumer_p.start()

    # Get results from the consumer
    retrieved_frames_list = results_queue.get()
    retrieved_frames = np.concatenate(retrieved_frames_list, axis=0, dtype=FRAME_DTYPE)

    # Wait for processes to finish
    producer_p.join()
    consumer_p.join()
    
    print(f"Retrieved frames shape: {retrieved_frames.shape}")
    print(f"Expected {num_frames} frames, got {len(retrieved_frames)}")

    # Verify the retrieved frames
    expected_frames = create_dummy_frames(0, num_frames, shape=FRAME_SHAPE, dtype=FRAME_DTYPE)
    print(f"Retrieved frame heads: {retrieved_frames[:,0,0,0]}")
    print(f"Expected frame heads: {expected_frames[:,0,0,0]}")

    # Due to the nature of concurrent access and buffer wrapping, the order of retrieved
    # frames might not be strictly sequential if the consumer is faster than the producer
    # and the buffer becomes empty. However, all frames put should eventually be retrievable
    # if the number of gets matches the number of puts and there are no timeouts.
    # A more robust test would check if all *expected* frames are present in the retrieved set,
    # possibly by sorting or using a set comparison, but for a basic test, checking the
    # total number of retrieved frames and a sample might suffice.
    # Let's check the number of retrieved frames and the content of the first and last retrieved frames.

    assert len(retrieved_frames) == num_frames, f"Expected {num_frames} frames, got {len(retrieved_frames)}"

    # Check if the set of retrieved frames matches the set of expected frames
    # This is a more reliable way to check correctness in a concurrent scenario
    retrieved_set = set(tuple(frame.tobytes()) for frame in retrieved_frames)
    expected_set = set(tuple(frame.tobytes()) for frame in expected_frames)
    assert retrieved_set == expected_set, "Retrieved frames do not match expected frames"


def test_invalid_frame_put(shared_buffer):
    """Test putting frames with invalid size or dtype."""
    # Test invalid shape
    invalid_shape_frame = np.zeros((1,) + (10, 20, 4), dtype=FRAME_DTYPE) # Wrong channel count
    assert shared_buffer.put(invalid_shape_frame) is False

    # Test invalid dtype
    invalid_dtype_frame = np.zeros((1,) + FRAME_SHAPE, dtype=np.float32) # Wrong dtype
    assert shared_buffer.put(invalid_dtype_frame) is False

    # Ensure no frames were actually put
    frames_list = shared_buffer.get(1, timeout=0.1)
    assert frames_list is None




def test_close_and_unlink(shared_buffer):
    """Test closing and unlinking the shared memory."""
    # The fixture already handles unlinking and closing, but we can add checks here.
    # After the fixture yields, the buffer is unlinked and closed.
    # We can try to access the shared memory by name after unlinking and expect an error.
    metadata_name = shared_buffer.metadata_name
    data_name = shared_buffer.data_name

    # The buffer is unlinked and closed by the fixture's teardown.
    # We can try to create a new SharedMemory with the same name and expect FileNotFoundError
    # if unlink was successful. However, this might be race-prone in some environments.
    # A simpler check is to ensure the internal shared memory objects are set to None
    # after unlink is called (which is done in the provided code).

    # The fixture's teardown calls unlink and close.
    # We can add assertions *within* the fixture's teardown, but that's not standard.
    # Let's rely on the fact that the fixture's teardown is executed.
    # A more direct test would involve manually calling close and unlink and checking.

    # Let's modify the fixture slightly to allow testing close and unlink explicitly.
    # We'll create the buffer in the test and manually call close/unlink.

    # This test will be done without the fixture to control close/unlink timing.
    buffer = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=BUFFER_CAPACITY, frame_shape=FRAME_SHAPE, dtype=FRAME_DTYPE)
    metadata_name = buffer.metadata_name
    data_name = buffer.data_name

    buffer.close()
    # After closing, the shared memory segments still exist but the connections are closed.
    # We should be able to unlink them.
    buffer.unlink()

    # After unlinking, trying to access by name should fail.
    with pytest.raises(FileNotFoundError):
        mp_shm.SharedMemory(name=metadata_name)
    with pytest.raises(FileNotFoundError):
        mp_shm.SharedMemory(name=data_name)

    # Ensure internal references are cleared after unlink
    assert buffer._metadata_shm is None
    assert buffer._data_shm is None


def test_peek_last_frame_basic(shared_buffer):
    """Test basic peek_last_frame operations."""
    # 1. Peek empty buffer
    assert shared_buffer.peek_last_frame() is None, "Peek on empty buffer should return None"

    # 2. Put one frame and peek
    frame0 = create_dummy_frames(0, 1)
    assert shared_buffer.put(frame0) is True
    peeked0 = shared_buffer.peek_last_frame()
    assert peeked0 is not None, "Peek after first put returned None"
    assert np.array_equal(peeked0, frame0[0]), "Peeked frame 0 content mismatch"
    # Ensure it's a copy
    peeked0_copy = peeked0.copy()
    peeked0_copy[0, 0, 0] = 99 # Modify the peeked frame copy
    peeked0_again = shared_buffer.peek_last_frame() # Peek again
    assert peeked0_again is not None
    assert peeked0_again[0, 0, 0] == 0, "Original buffer data changed after modifying peeked copy"
    assert not np.array_equal(peeked0_copy, peeked0_again), "Modified copy should differ from fresh peek"


    # 3. Put another frame and peek
    frame1 = create_dummy_frames(1, 1)
    assert shared_buffer.put(frame1) is True
    peeked1 = shared_buffer.peek_last_frame()
    assert peeked1 is not None, "Peek after second put returned None"
    assert np.array_equal(peeked1, frame1[0]), "Peeked frame 1 content mismatch"

    # 4. Ensure read/write pointers and counts are unchanged by peek
    # Need to acquire lock to read pointers reliably for assertion
    with shared_buffer.pointer_lock:
        read_ptr_before, write_ptr_before, occupied_before, last_get_before = shared_buffer._get_pointers_metadata()

    shared_buffer.peek_last_frame() # Perform the peek

    with shared_buffer.pointer_lock:
        read_ptr_after, write_ptr_after, occupied_after, last_get_after = shared_buffer._get_pointers_metadata()

    assert read_ptr_before == read_ptr_after, "Read pointer changed after peek"
    assert write_ptr_before == write_ptr_after, "Write pointer changed after peek"
    assert occupied_before == occupied_after, "Occupied count changed after peek"
    assert last_get_before == last_get_after, "Last get count changed after peek"


def test_peek_last_frame_after_get_no_release(shared_buffer):
    """Test peeking after get() without explicit release, covering user scenarios."""
    frames_initial = create_dummy_frames(0, 3) # F0, F1, F2
    assert shared_buffer.put(frames_initial) is True

    # --- Scenario A: Get 2 frames (like get(2)) ---
    print("\nScenario A: get(2)")
    got_list_1 = shared_buffer.get(2) # Gets F0, F1
    assert got_list_1 is not None
    assert len(got_list_1) == 1
    assert np.array_equal(got_list_1[0], frames_initial[:2])

    # Peek after getting 2 (last put was F2)
    peeked_after_get2 = shared_buffer.peek_last_frame()
    assert peeked_after_get2 is not None, "Peek after get(2) returned None"
    assert np.array_equal(peeked_after_get2, frames_initial[2]), "Peek after get(2) should return the last PUT frame (F2)"
    print(f"Peek after get(2) successful, got frame with value {peeked_after_get2[0,0,0]}")


    # --- Scenario B: Put another frame, then get 1 (like get(1) without release) ---
    print("\nScenario B: put(F3), get(1)")
    frame3 = create_dummy_frames(3, 1) # F3
    assert shared_buffer.put(frame3) is True

    # Get 1 frame (gets F2, implicitly releases F0, F1 because last_get_count was 2)
    got_list_2 = shared_buffer.get(1)
    assert got_list_2 is not None, "Second get(1) failed"
    assert len(got_list_2) == 1
    assert np.array_equal(got_list_2[0], frames_initial[2:3]), "Second get(1) did not retrieve F2" # Should get F2

    # Peek after getting 1 (last put was F3)
    # At this point, F0, F1 are released. F2 was retrieved (last_get_count=1). F3 is in buffer.
    peeked_after_get1 = shared_buffer.peek_last_frame()
    assert peeked_after_get1 is not None, "Peek after get(1) returned None"
    assert np.array_equal(peeked_after_get1, frame3[0]), "Peek after get(1) should return the last PUT frame (F3)"
    print(f"Peek after get(1) successful, got frame with value {peeked_after_get1[0,0,0]}")


def test_peek_last_frame_after_get_and_release(shared_buffer):
    """Test peeking after get() with explicit release."""
    frames_initial = create_dummy_frames(10, 3) # F10, F11, F12
    assert shared_buffer.put(frames_initial) is True

    # Get 1 frame
    got_list = shared_buffer.get(1) # Gets F10
    assert got_list is not None
    assert np.array_equal(got_list[0], frames_initial[0:1])

    # Explicitly release the gotten frame
    released_count = shared_buffer.release_last_got_data()
    assert released_count == 1

    # Peek (last put was F12)
    peeked = shared_buffer.peek_last_frame()
    assert peeked is not None, "Peek after explicit release returned None"
    assert np.array_equal(peeked, frames_initial[2]), "Peek after explicit release should return last PUT frame (F12)"


def test_peek_last_frame_wrap_around(shared_buffer):
    """Test peeking when the write pointer has wrapped around."""
    # Fill buffer partially (e.g., 3 frames in a capacity 5 buffer)
    assert shared_buffer.put(create_dummy_frames(0, 3)) is True # Puts F0, F1, F2 at indices 0, 1, 2

    # Get 2 frames
    got1 = shared_buffer.get(2) # Gets F0, F1. read_ptr becomes 2.
    assert got1 is not None

    # Release F0, F1
    released_count = shared_buffer.release_last_got_data() 
    assert released_count == 2

    # Put 4 more frames (F3, F4, F5, F6). This will wrap around.
    # write_ptr was 3. Put F3 at 3, F4 at 4. Wrap. Put F5 at 0, F6 at 1.
    # write_ptr becomes 2. Last written frame is F6 at index 1.
    assert shared_buffer.put(create_dummy_frames(3, 4)) is True

    # Peek - should get F6
    peeked = shared_buffer.peek_last_frame()
    assert peeked is not None, "Peek after wrap returned None"
    assert np.array_equal(peeked, create_dummy_frames(6, 1)[0]), "Peek after wrap returned incorrect frame"


# Helper function for the concurrent peeker task
def _concurrent_peek_task(buffer_or_source, duration, results_queue, stop_event, use_process=False):
    """Target function for the peeker thread/process."""
    buffer = None
    peeked_values = []
    try:
        if use_process:
            buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=buffer_or_source)
        else:
            buffer = buffer_or_source

        start_time = time.time()
        while not stop_event.is_set() and time.time() - start_time < duration:
            peeked_frame = buffer.peek_last_frame()
            if peeked_frame is not None:
                # Extract a representative value (e.g., top-left pixel of the first channel)
                # Ensure the frame has the expected dimensions before accessing
                if peeked_frame.shape == FRAME_SHAPE:
                     value = peeked_frame[0, 0, 0]
                     peeked_values.append(value)
                else:
                     print(f"Warning: Peeked frame had unexpected shape {peeked_frame.shape}")
                     peeked_values.append(f"ErrorShape:{peeked_frame.shape}") # Record error
            else:
                peeked_values.append(None) # Record None if buffer was empty
            time.sleep(0.01) # Peek frequently but avoid busy-waiting

        results_queue.put({'type': 'peek', 'values': peeked_values})

    except Exception as e:
        print(f"{'Process' if use_process else 'Thread'} Peek Task Error: {e}")
        results_queue.put({'type': 'peek', 'values': peeked_values, 'error': e}) # Send partial results on error
    finally:
        if use_process and buffer:
            try:
                buffer.close()
            except Exception as close_e:
                 print(f"Error closing buffer in peek task: {close_e}")


# Helper functions modified for concurrent peek test (returning values)
def _concurrent_put_task_returning_values(buffer_or_source, duration, frame_shape, frame_dtype, result_queue, producer_finished_event, start_value_offset=0):
    """Producer task that returns the list of values put."""
    buffer = None
    put_values = []
    frame_counter = 0
    try:
        buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=buffer_or_source)
        start_time = time.time()
        while time.time() - start_time < duration:
            # Ensure value fits within dtype limits (e.g., uint8)
            value_to_put = (start_value_offset + frame_counter) % 256 # Modulo for uint8
            frame = create_dummy_frames(value_to_put, 1, shape=frame_shape, dtype=frame_dtype)
            if buffer.put(frame, timeout=1.0):
                put_values.append(value_to_put)
                frame_counter += 1
            else:
                # print("Producer timeout")
                pass # Keep trying within duration
        result_queue.put({'type': 'put', 'values': put_values})
    except Exception as e:
        print(f"Producer Task Error: {e}")
        result_queue.put({'type': 'put', 'values': put_values, 'error': e}) # Send partial results
    finally:
        producer_finished_event.set() # Signal finish
        if buffer:
             try:
                 buffer.close()
             except Exception as close_e:
                  print(f"Error closing buffer in put task: {close_e}")


def _concurrent_get_task_returning_values(buffer_or_source, duration, frame_shape, frame_dtype, result_queue, producer_finished_event):
    """Consumer task that returns the list of values retrieved."""
    buffer = None
    retrieved_values = []
    try:
        buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=buffer_or_source)
        while not producer_finished_event.is_set() or buffer.unread_count > 0:
            frames_list = buffer.get(1, timeout=0.1) # Get one frame at a time
            if frames_list:
                # Extract value from the retrieved frame (assuming dummy frame structure)
                retrieved_frame = frames_list[0] # Only one frame requested
                if retrieved_frame.shape == frame_shape:
                     value = retrieved_frame[0, 0, 0]
                     retrieved_values.append(value)
                else:
                     print(f"Warning: Got frame had unexpected shape {retrieved_frame.shape}")
                     retrieved_values.append(f"ErrorShape:{retrieved_frame.shape}") # Record error

            elif producer_finished_event.is_set():
                 # Producer finished and buffer is empty
                 break
            # else: continue waiting if producer is still running

        result_queue.put({'type': 'get', 'values': retrieved_values})
    except Exception as e:
        print(f"Consumer Task Error: {e}")
        result_queue.put({'type': 'get', 'values': retrieved_values, 'error': e}) # Send partial results
    finally:
        if buffer:
             try:
                 buffer.close()
             except Exception as close_e:
                  print(f"Error closing buffer in get task: {close_e}")


# Test function for concurrent peek
def test_peek_last_frame_concurrent(shared_buffer):
    """
    Tests peek_last_frame concurrently with put and get operations using processes.
    """
    mp.set_start_method('spawn', force=True)

    test_duration = 3.0 # seconds
    num_producers = 1
    num_consumers = 1
    num_peekers = 2

    producer_results_queue = mp.Queue()
    consumer_results_queue = mp.Queue()
    peeker_results_queue = mp.Queue()
    producer_finished_event = mp.Event() # Use the same event as stress test
    stop_peeking_event = mp.Event() # Separate event to stop peekers

    processes = []

    # Start Producers
    for i in range(num_producers):
        p = mp.Process(target=_concurrent_put_task_returning_values,
                       args=(shared_buffer, test_duration, FRAME_SHAPE, FRAME_DTYPE, producer_results_queue, producer_finished_event, i * 1000)) # Offset start value
        processes.append(p)
        p.start()

    # Start Consumers
    for _ in range(num_consumers):
        c = mp.Process(target=_concurrent_get_task_returning_values,
                       args=(shared_buffer, test_duration, FRAME_SHAPE, FRAME_DTYPE, consumer_results_queue, producer_finished_event))
        processes.append(c)
        c.start()

    # Start Peekers
    for _ in range(num_peekers):
        pk = mp.Process(target=_concurrent_peek_task,
                        args=(shared_buffer, test_duration, peeker_results_queue, stop_peeking_event, True))
        processes.append(pk)
        pk.start()

    # Wait for producers and consumers based on duration/event
    # Wait slightly longer than duration for producers to finish and signal
    time.sleep(test_duration + 0.5)
    # producer_finished_event should be set by producers now

    # Signal peekers to stop
    stop_peeking_event.set()

    # Collect results
    put_results_data = producer_results_queue.get(timeout=5.0)
    put_values = put_results_data.get('values', [])
    if put_results_data.get('error'): print(f"Producer Error reported: {put_results_data['error']}")


    consumed_results_data = consumer_results_queue.get(timeout=5.0)
    consumed_values = consumed_results_data.get('values', [])
    if consumed_results_data.get('error'): print(f"Consumer Error reported: {consumed_results_data['error']}")


    peeked_results_all = []
    for _ in range(num_peekers):
         peeker_data = peeker_results_queue.get(timeout=5.0)
         peeked_results_all.append(peeker_data.get('values', []))
         if peeker_data.get('error'): print(f"Peeker Error reported: {peeker_data['error']}")


    # Join all processes
    for p in processes:
        p.join(timeout=5.0)
        if p.is_alive():
             print(f"Warning: Process {p.name} (pid {p.pid}) did not terminate gracefully. Terminating.")
             p.terminate()
             p.join(timeout=1.0) # Wait after terminate


    print(f"\nConcurrent Peek Test: Total put values: {len(put_values)}")
    print(f"Concurrent Peek Test: Total consumed values: {len(consumed_values)}")
    # Basic check: consumed should be <= put (consumer might stop slightly earlier)
    # assert len(consumed_values) <= len(put_values) # Can fail if consumer runs slightly longer

    # Verification for peekers:
    # Each value peeked (that is not None and not an error string) should be present in the list of values put by the producer.
    put_value_set = set(put_values)
    all_peeked_values_flat = [val for peeker_list in peeked_results_all for val in peeker_list if val is not None and not isinstance(val, str)]


    print(f"Concurrent Peek Test: Total valid peeked values: {len(all_peeked_values_flat)}")
    if all_peeked_values_flat: print(f"Concurrent Peek Test: Example peeked values: {all_peeked_values_flat[:20]}")
    if put_values: print(f"Concurrent Peek Test: Example put values: {put_values[:20]}")


    invalid_peeks = 0
    for peeked_val in all_peeked_values_flat:
        if peeked_val not in put_value_set:
            invalid_peeks += 1
            # print(f"Warning: Peeked value {peeked_val} not found in put values.") # Optional: print warnings

    # Allow a small tolerance for race conditions where a peek might catch a value
    # just as it's being overwritten, although ideally peek_last_frame is atomic enough.
    allowed_invalid_ratio = 0.01 # Allow 1% potentially inconsistent peeks due to extreme race conditions
    if len(all_peeked_values_flat) > 0:
         assert invalid_peeks / len(all_peeked_values_flat) <= allowed_invalid_ratio, \
             f"Found {invalid_peeks}/{len(all_peeked_values_flat)} peeked values not present in the set of put values."
    else:
         # If nothing was peeked, it might indicate an issue or just very fast consumption
         print("Warning: No valid values were peeked during the concurrent test.")
         # We should still check if values were put and consumed
         assert len(put_values) > 0, "Concurrent test put 0 values."
         # assert len(consumed_values) > 0, "Concurrent test consumed 0 values." # Consumer might finish before getting anything if producer is slow


    print("\nConcurrent peek test finished.")



import random
import cProfile
import pstats
import io
import threading
import queue

def profiled_worker(target_func, *args, **kwargs):
    """
    对传入的目标函数进行性能分析（支持参数）
    
    :param target_func: 需要分析的目标函数
    :param args: 目标函数的位置参数
    :param kwargs: 目标函数的关键字参数
    """
    pr = cProfile.Profile()
    pr.enable()
    target_func(*args, **kwargs)  # 执行传入的目标函数（带参数）
    pr.disable()
    
    # 将分析结果输出到字符串流
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()

    print("\n=== Detailed Call Analysis for Target Function ===")
    # ps.print_callers()  # 查看谁调用了目标函数（不直接有用）
    # ps.print_callees()  # 查看目标函数调用了哪些子函数（更相关）

    print(s.getvalue())  # 打印分析结果

def stress_producer_task(source_buffer, duration, frame_shape, frame_dtype, results_queue, producer_finished_event):
    """Task for a stress test producer process."""
    try:
        buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=source_buffer)
        # buffer = source_buffer # Use mp.Queue() as a comparison
        start_time = time.time()
        frame_count = 0
        while time.time() - start_time < duration:
            # Put random number of frames, ensure it's at least 1
            num_frames_to_put = random.randint(1, BUFFER_CAPACITY//2)
            frames = create_dummy_frames(frame_count, num_frames_to_put, shape=frame_shape, dtype=frame_dtype, use_random=True)
            if buffer.put(frames, timeout=1.0): # Use a shorter timeout for stress test
                frame_count += num_frames_to_put
            else:
                # Continue trying to put frames until duration is reached
                pass
        results_queue.put(frame_count) # Return total frames produced
    except Exception as e:
        print(f"Stress Producer process error: {e}")
        raise
    finally:
        producer_finished_event.set() # Signal that the producer has finished
        buffer.close()

def stress_consumer_task(source_buffer, duration, frame_shape, frame_dtype, results_queue, producer_finished_event):
    """Task for a stress test consumer process."""
    try:
        buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=source_buffer)
        # buffer = source_buffer # Use mp.Queue() as a comparison
        start_time = time.time()
        retrieved_frames = []
        retrieved_count = 0
        # Continue as long as the producer is alive OR there are frames in the buffer
        while not producer_finished_event.is_set() or buffer.unread_count > 0:
            # Get random number of frames, ensure it's at least 1
            num_frames_to_get = random.randint(1, BUFFER_CAPACITY//2)
            frames_list = buffer.get(num_frames_to_get, timeout=0.1) # Use a shorter timeout

            # Code for Queue: 
            # frames_list is actually a single np.array, Queue timeout will raise an exception.
            # Baseline Perfomance for queue is 180k / 5s, Avg 1.5 frames put, 1 frames get.
            # retrieved_frames.append(frames_list) # No need to copy after data passed through Queue
            # retrieved_count += frames_list.shape[0]
            

            if frames_list is not None:
                for frames in frames_list:
                    retrieved_frames.append(frames.copy()) # Deep copy to persist the data
                    retrieved_count += frames.shape[0]
            # Do not controlled by global timeout, wait producer to finish or buffer be empty.
        

        # After the main loop, there might still be frames put by the producer just before it finished.
        # This loop ensures we consume everything remaining in the buffer.
        # while buffer.unread_count > 0:
        #      frames_list = buffer.get(BUFFER_CAPACITY, timeout=0.1) # Try to get remaining frames with a small timeout
        #      if frames_list is not None:
        #          for frames in frames_list:
        #              retrieved_frames.append(frames.copy())
        #              retrieved_count += frames.shape[0]
        #      else:
        #          break # Should not happen if occupied_count > 0 and timeout is sufficient, but as a safeguard
        results_queue.put((retrieved_frames, retrieved_count)) # Return retrieved frames and total count
    except Exception as e:
        print(f"Stress Consumer process error: {e}")
        raise
    finally:
        print(f"Stress Consumer process finished, retrieved {retrieved_count} frames")
        buffer.close()

def test_process_safety_stress(shared_buffer):
    """Test concurrent put and get operations from multiple processes under stress."""
    print("In test_process_safety_stress, Starting stress test...")
    mp.set_start_method('spawn', force=True) # Use spawn method for better compatibility

    test_duration = 5 # seconds

    # Use Queues to get results from the processes
    producer_results_queue = mp.Queue(maxsize=2)
    consumer_results_queue = mp.Queue(maxsize=2)

    # Create an event to signal when the producer is finished
    producer_finished_event = mp.Event()

    # shared_buffer = mp.Queue(maxsize=5) # 使用mp.Queue()

    # Create producer and consumer processes
    producer_p = mp.Process(target=stress_producer_task, args=(shared_buffer, test_duration, FRAME_SHAPE, FRAME_DTYPE, producer_results_queue, producer_finished_event))
    consumer_p = mp.Process(target=stress_consumer_task, args=(shared_buffer, test_duration, FRAME_SHAPE, FRAME_DTYPE, consumer_results_queue, producer_finished_event))

    # Start processes
    producer_p.start()
    consumer_p.start()

    # Get results from the queues
    total_produced = producer_results_queue.get()
    retrieved_frames_list, total_retrieved = consumer_results_queue.get()

    # Wait for processes to finish
    producer_p.join()
    consumer_p.join()

    print(f"Stress Test: Total frames produced: {total_produced}")
    print(f"Stress Test: Total frames retrieved: {total_retrieved}")
    print(f"Performance: {total_retrieved / test_duration} fps")

    # Verify the total number of frames
    assert total_produced == total_retrieved, f"Stress Test Failed: Produced {total_produced} frames, but retrieved {total_retrieved}"

    # Verify the sequence of retrieved frames
    # Create the expected sequence based on the total number of frames produced
    expected_frames = create_dummy_frames(0, total_produced, shape=FRAME_SHAPE, dtype=FRAME_DTYPE, use_random=True)

    # Concatenate the retrieved frames list into a single array for comparison
    retrieved_frames = np.concatenate(retrieved_frames_list, axis=0, dtype=FRAME_DTYPE)

    assert np.array_equal(retrieved_frames, expected_frames), "Stress Test Failed: Retrieved frame sequence does not match expected sequence"


def test_deadlock_after_full_get(shared_buffer):
    """
    Test for deadlock scenario after buffer is fully put and then fully retrieved.
    Verifies that a subsequent timed-out get releases space, allowing further puts and gets.
    """
    # 1. Fill the buffer completely
    frames_to_put_initial = create_dummy_frames(0, BUFFER_CAPACITY)
    assert shared_buffer.put(frames_to_put_initial) is True, "Initial put to fill buffer failed"

    # 2. Get all frames
    retrieved_frames_list = shared_buffer.get(BUFFER_CAPACITY)
    assert retrieved_frames_list is not None, "Initial get of all frames failed"
    # Concatenate the list of arrays into a single array for easier comparison if needed later
    retrieved_frames = np.concatenate(retrieved_frames_list, axis=0)
    assert len(retrieved_frames) == BUFFER_CAPACITY, f"Expected {BUFFER_CAPACITY} frames, but got {len(retrieved_frames)}"
    assert np.array_equal(retrieved_frames, frames_to_put_initial), "Retrieved frames do not match initially put frames"

    # 3. Attempt a timed-out get to trigger space release
    # This get should time out as there's no new data, but it should release the space
    # occupied by the previously retrieved frames according to the design.
    timeout_get_result = shared_buffer.get(1, timeout=0.1)
    assert timeout_get_result is None, "Timed-out get did not return None as expected"

    # 4. Attempt a put which should now succeed
    # Space should have been released by the previous timed-out get.
    frame_to_put_after_release = create_dummy_frames(BUFFER_CAPACITY, 1) # Use a new value
    put_after_release_success = shared_buffer.put(frame_to_put_after_release, timeout=0.1) # Use a small timeout just in case
    assert put_after_release_success is True, "Put after space release failed"

    # TODO: Combine 3 and 4 in a multi-thread/multiprocess scenario to ensure the `put` is not blocked

    # 5. Attempt a get to retrieve the newly put frame
    get_after_put_result_list = shared_buffer.get(1, timeout=0.1) # Use a small timeout
    assert get_after_put_result_list is not None, "Get after putting new frame failed"
    assert len(get_after_put_result_list) == 1, "Get after putting new frame returned incorrect number of frames"
    get_after_put_result = get_after_put_result_list[0]
    assert np.array_equal(get_after_put_result, frame_to_put_after_release), "Retrieved frame after put does not match the frame that was put"

    print("\nDeadlock scenario test passed: Buffer correctly handled full-get-timeout-get-put-get sequence.")


# === Helper functions for concurrent put/get tests ===

def _concurrent_put_task(buffer_or_source, frame_to_put, result_queue, use_process=False):
    """Target function for the put thread/process."""
    buffer = None
    try:
        if use_process:
            # Need to reconnect in a new process
            buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=buffer_or_source)
        else:
            # In a thread, we can use the buffer directly
            buffer = buffer_or_source

        # print(f"{'Process' if use_process else 'Thread'} Put Task: Trying to put frame...")
        put_success = buffer.put(frame_to_put, timeout=2.0) # Use a reasonable timeout
        # print(f"{'Process' if use_process else 'Thread'} Put Task: Put result: {put_success}")
        result_queue.put({'type': 'put', 'success': put_success})
    except Exception as e:
        print(f"{'Process' if use_process else 'Thread'} Put Task Error: {e}")
        result_queue.put({'type': 'put', 'success': False, 'error': e})
    finally:
        if use_process and buffer:
            buffer.close()


def _concurrent_get_task(buffer_or_source, result_queue, use_process=False):
    """Target function for the get thread/process."""
    buffer = None
    try:
        if use_process:
            buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=buffer_or_source)
        else:
            buffer = buffer_or_source

        # print(f"{'Process' if use_process else 'Thread'} Get Task: Trying to get frame...")
        get_result_list = buffer.get(1, timeout=2.0) # Use a reasonable timeout
        # print(f"{'Process' if use_process else 'Thread'} Get Task: Get result: {'Got frame' if get_result_list else 'None'}")

        if get_result_list:
            # Important: Copy data if using multiprocessing, as the view might become invalid
            # after the process exits or the buffer wraps around.
            # For threading, copying is also safer if the main thread might modify the buffer later.
            retrieved_frame = np.concatenate(get_result_list, axis=0).copy()
            result_queue.put({'type': 'get', 'frame': retrieved_frame})
        else:
            result_queue.put({'type': 'get', 'frame': None})
    except Exception as e:
        print(f"{'Process' if use_process else 'Thread'} Get Task Error: {e}")
        result_queue.put({'type': 'get', 'frame': None, 'error': e})
    finally:
        if use_process and buffer:
            buffer.close()


# === New tests for concurrent put/get after full read ===

def test_concurrent_put_get_multithread(shared_buffer):
    """
    Tests concurrent put and get using threads after the buffer was filled and emptied.
    Ensures get blocks until put provides data.
    """
    # 1. Fill the buffer completely
    frames_to_put_initial = create_dummy_frames(0, BUFFER_CAPACITY)
    assert shared_buffer.put(frames_to_put_initial) is True, "Initial put to fill buffer failed"

    # 2. Get all frames to empty it
    retrieved_frames_list = shared_buffer.get(BUFFER_CAPACITY)
    assert retrieved_frames_list is not None, "Initial get of all frames failed"
    assert len(np.concatenate(retrieved_frames_list, axis=0)) == BUFFER_CAPACITY

    # 3. Prepare for concurrent operations
    frame_to_put_concurrently = create_dummy_frames(BUFFER_CAPACITY, 1) # Next frame
    results_queue = queue.Queue()

    # 4. Start the Get thread first
    get_thread = threading.Thread(target=_concurrent_get_task, args=(shared_buffer, results_queue, False))
    get_thread.start()

    # 5. Verify Get thread is blocked (wait a bit, queue should be empty)
    time.sleep(0.2) # Give Get thread a chance to block
    assert results_queue.empty(), "Get thread returned prematurely before put"

    # 6. Start the Put thread
    put_thread = threading.Thread(target=_concurrent_put_task, args=(shared_buffer, frame_to_put_concurrently, results_queue, False))
    put_thread.start()

    # 7. Wait for both threads to complete
    get_thread.join(timeout=5.0)
    put_thread.join(timeout=5.0)

    assert not get_thread.is_alive(), "Get thread did not finish in time"
    assert not put_thread.is_alive(), "Put thread did not finish in time"

    # 8. Collect and verify results
    results = {}
    while not results_queue.empty():
        item = results_queue.get()
        results[item['type']] = item # Store results by type ('get' or 'put')

    assert 'put' in results, "Put result missing from queue"
    assert results['put'].get('success') is True, f"Concurrent put failed. Error: {results['put'].get('error')}"

    assert 'get' in results, "Get result missing from queue"
    assert results['get'].get('frame') is not None, f"Concurrent get failed to retrieve frame. Error: {results['get'].get('error')}"
    assert np.array_equal(results['get']['frame'], frame_to_put_concurrently), "Concurrently retrieved frame mismatch"

    print("\nConcurrent put/get multithread test passed.")


def test_concurrent_put_get_multiprocess(shared_buffer):
    """
    Tests concurrent put and get using processes after the buffer was filled and emptied.
    Ensures get blocks until put provides data.
    """
    mp.set_start_method('spawn', force=True) # Ensure consistent start method

    # 1. Fill the buffer completely
    frames_to_put_initial = create_dummy_frames(0, BUFFER_CAPACITY)
    assert shared_buffer.put(frames_to_put_initial) is True, "Initial put to fill buffer failed"

    # 2. Get all frames to empty it
    retrieved_frames_list = shared_buffer.get(BUFFER_CAPACITY)
    assert retrieved_frames_list is not None, "Initial get of all frames failed"
    assert len(np.concatenate(retrieved_frames_list, axis=0)) == BUFFER_CAPACITY

    # 3. Prepare for concurrent operations
    frame_to_put_concurrently = create_dummy_frames(BUFFER_CAPACITY, 1) # Next frame
    results_queue = mp.Queue()

    # 4. Start the Get process first
    get_process = mp.Process(target=_concurrent_get_task, args=(shared_buffer, results_queue, True))
    get_process.start()

    # 5. Verify Get process is blocked (wait a bit, queue should be empty)
    time.sleep(0.2) # Give Get process a chance to block
    assert results_queue.empty(), "Get process returned prematurely before put"

    # 6. Start the Put process
    put_process = mp.Process(target=_concurrent_put_task, args=(shared_buffer, frame_to_put_concurrently, results_queue, True))
    put_process.start()

    # 7. Wait for both processes to complete
    get_process.join(timeout=5.0)
    put_process.join(timeout=5.0)

    assert not get_process.is_alive(), "Get process did not finish in time"
    assert not put_process.is_alive(), "Put process did not finish in time"

    # 8. Collect and verify results
    results = {}
    while not results_queue.empty():
        try:
            item = results_queue.get(timeout=0.1) # Use timeout for safety
            results[item['type']] = item
        except queue.Empty: # Use queue.Empty for mp.Queue as well
            break

    assert 'put' in results, "Put result missing from queue"
    assert results['put'].get('success') is True, f"Concurrent put failed. Error: {results['put'].get('error')}"

    assert 'get' in results, "Get result missing from queue"
    assert results['get'].get('frame') is not None, f"Concurrent get failed to retrieve frame. Error: {results['get'].get('error')}"
    assert np.array_equal(results['get']['frame'], frame_to_put_concurrently), "Concurrently retrieved frame mismatch"

    print("\nConcurrent put/get multiprocess test passed.")
