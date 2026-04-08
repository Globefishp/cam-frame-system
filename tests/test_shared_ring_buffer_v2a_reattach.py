# Google Gemini 3.1 Pro & 3 flash

import pytest
import numpy as np
import multiprocessing as mp
import time
from ringbuffers.shared_ring_buffer_v2a import ProcessSafeSharedRingBuffer

# Use 'spawn' for Windows compatibility and cleaner isolation
ctx = mp.get_context("spawn")

# Helper function for the writer process
def writer_process(buffer_obj, done_event, value_to_write):
    """A process that simply puts one frame into the buffer and exits."""
    try:
        frame = np.full((1, 2, 2, 3), value_to_write, dtype=np.uint8)
        buffer_obj.put(frame, timeout=2)
    finally:
        # Ensure the event is set and the buffer handle is closed
        done_event.set()
        buffer_obj.close()

def shm_keeper_process(buffer_obj, ready_event, stop_event):
    """
    A helper process to hold SHM references. 
    On Windows, SHM is destroyed if ref count hits zero.
    """
    # Simply having buffer_obj in this process increases ref count
    ready_event.set()
    stop_event.wait()
    buffer_obj.close()

@pytest.fixture
def shared_buffer_with_keeper():
    """
    Fixture that provides a buffer and a background process 
    to keep shared memory alive during close/reattach cycles.
    """
    buf = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=5, frame_shape=(2, 2, 3))
    ready_event = ctx.Event()
    stop_event = ctx.Event()
    
    keeper = ctx.Process(target=shm_keeper_process, args=(buf, ready_event, stop_event))
    keeper.start()
    ready_event.wait() # Ensure keeper is attached
    
    yield buf
    
    stop_event.set()
    keeper.join()
    buf.close()
    try:
        buf.unlink()
    except:
        pass

def test_reattach_lifecycle(shared_buffer_with_keeper):
    """Scenario 1: Test the basic close -> reattach cycle."""
    buf = shared_buffer_with_keeper
    
    # Initial state check
    assert buf._metadata_shm.buf is not None
    assert buf._metadata_ctypes is not None

    buf.close()
    
    # Re-attach
    buf.reattach()
    
    # Verify restoration
    assert buf._metadata_shm.buf is not None
    assert buf._metadata_ctypes is not None
    assert buf.buffer_capacity == 5

def test_reattach_data_persistence(shared_buffer_with_keeper):
    """Scenario 2: Ensure data and pointers remain consistent after reattach."""
    buf = shared_buffer_with_keeper
    
    # 1. Put data
    test_val = 123
    frame = np.full((1, 2, 2, 3), test_val, dtype=np.uint8)
    buf.put(frame)
    assert buf.unread_count == 1
    
    # 2. Close and simulate process "re-entry"
    buf.close()
    
    # 3. Reattach and verify
    buf.reattach()
    assert buf.unread_count == 1
    
    # 4. Get data and verify content
    got_list = buf.get(1)
    assert got_list is not None
    assert np.all(got_list[0] == test_val)

def test_reattach_negative_after_unlink():
    """Scenario 4: Reattach must fail if the memory is globally unlinked."""
    buf = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=2, frame_shape=(1, 1, 1))
    
    # No keeper process here. 
    # Close and Unlink globally.
    buf.close()
    buf.unlink()
    
    # Reattach should fail because the SHM segments are gone from the OS
    with pytest.raises((FileNotFoundError, RuntimeError)):
        buf.reattach()

def test_reattach_after_external_write(shared_buffer_with_keeper):
    """
    Scenario 1: Test reattaching and reading data written by another process.
    This verifies that reattach correctly syncs with the externally modified state.
    """
    buf = shared_buffer_with_keeper
    done_event = ctx.Event()
    value_to_write = 88
    
    # Start a writer process to add data to the buffer
    writer = ctx.Process(target=writer_process, args=(buf, done_event, value_to_write))
    writer.start()
    
    # Immediately close the main process's handle.
    # The 'keeper' process ensures the SHM segment itself survives.
    buf.close()

    # Wait for the writer to finish its job
    done_event.wait(timeout=5)
    writer.join(timeout=5)
    
    # Now, reattach in the main process
    buf.reattach()
    assert buf._metadata_ctypes is not None, "Internal ctypes mapping should be restored after reattach"
    
    # Verify that we can read the data written by the external process
    assert buf.unread_count == 1
    retrieved_list = buf.get(1, timeout=2)
    assert retrieved_list is not None
    retrieved_frame = retrieved_list[0]
    
    assert np.all(retrieved_frame == value_to_write), "Data read after reattach does not match what was written externally"

def test_multiple_reattach_cycles_stability(shared_buffer_with_keeper):
    """
    Scenario 2: Stress test reattach by calling it in a rapid loop.
    The buffer must remain functional and not leak resources or corrupt its state.
    """
    buf = shared_buffer_with_keeper
    num_cycles = 20

    # Perform a high number of close -> reattach cycles
    for i in range(num_cycles):
        # Check initial state (should be attached)
        assert buf._metadata_ctypes is not None, f"Cycle {i}: Buffer should be attached at the start"
        
        buf.close()
        
        buf.reattach()
        assert buf._metadata_ctypes is not None, f"Cycle {i}: ctypes mapping should be restored after reattach"
        # A simple check to ensure the connection is valid
        assert buf.buffer_capacity == 5
        
    # After all the cycles, the buffer must still be fully operational.
    # This is the most important verification.
    final_test_val = 99
    final_frame = np.full((1, 2, 2, 3), final_test_val, dtype=np.uint8)
    
    assert buf.put(final_frame, timeout=2), "Putting data failed after multiple reattach cycles"
    
    retrieved_list = buf.get(1, timeout=2)
    assert retrieved_list is not None, "Getting data failed after multiple reattach cycles"
    
    assert np.all(retrieved_list[0] == final_test_val), "Data integrity compromised after multiple reattach cycles"

def test_reattach_redundant_calls(shared_buffer_with_keeper):
    """
    Scenario: Accidental multiple calls to reattach() without calling close().
    Tests idempotency and resource handling when overwriting active SHM handles.
    """
    buf = shared_buffer_with_keeper
    
    # 1. Initial write
    frame1 = np.full((1, 2, 2, 3), 10, dtype=np.uint8)
    buf.put(frame1)
    
    # 2. Call reattach multiple times consecutively while the buffer is ACTIVE
    # This simulates a logic error in user code.
    try:
        buf.reattach()
        buf.reattach()
        buf.reattach()
    except BufferError as e:
        pytest.fail(f"Repeated reattach() triggered BufferError: {e}. "
                    "This usually happens when old memoryviews/ctypes pointers "
                    "prevent the previous SHM handle from closing.")
    except Exception as e:
        pytest.fail(f"Repeated reattach() failed with unexpected error: {e}")

    # 3. Verify object is still healthy
    assert buf._metadata_ctypes is not None
    assert buf.unread_count == 1
    
    # 4. Perform a write/read flow to ensure pointers are still valid
    frame2 = np.full((1, 2, 2, 3), 20, dtype=np.uint8)
    buf.put(frame2)
    
    got_list = buf.get(2) # Get both frames
    assert len(got_list) >= 1
    combined = np.concatenate(got_list, axis=0)
    assert np.all(combined[0] == 10)
    assert np.all(combined[1] == 20)
    
    # 5. Mixed close and multiple reattach
    buf.close()
    buf.reattach()
    buf.reattach() # Consecutive reattach after a close
    
    assert buf.unread_count == 0 # Data was consumed in step 4, then released by get/next op
    assert buf.buffer_capacity == 5

def test_reattach_stability_under_load(shared_buffer_with_keeper):
    """
    Scenario: Interleaving reattach calls with data operations.
    Ensures that re-mapping memory doesn't corrupt the data being processed.
    """
    buf = shared_buffer_with_keeper
    
    for i in range(10):
        val = i + 1
        # Write
        buf.put(np.full((1, 2, 2, 3), val, dtype=np.uint8))
        
        # Accidental reattach mid-operation
        if i % 2 == 0:
            buf.reattach()
            
        # Read and verify
        got = buf.get(1)
        assert np.all(got[0] == val)
        
        # Another reattach
        buf.reattach()

    assert buf.unread_count == 0