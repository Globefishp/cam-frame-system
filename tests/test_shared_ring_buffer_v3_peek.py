import unittest
import numpy as np
import time
import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
from multiprocessing.sharedctypes import Synchronized

from shared_ring_buffer_v3 import ProcessSafeSharedRingBuffer

# Use a larger buffer for this test
LARGE_BUFFER_CAPACITY = 20

# Producer process function
def producer(buffer: ProcessSafeSharedRingBuffer, put_ids: list, total_put: Synchronized, stop_event: mp_sync.Event):
    buffer_link = ProcessSafeSharedRingBuffer(create=False, source_buffer=buffer)
    frame_id_counter = 0
    while not stop_event.is_set():
        frame_id = frame_id_counter % 256 # Modulo for uint8
        frame = (np.ones(FRAME_SIZE, dtype=FRAME_DTYPE) * frame_id).astype(FRAME_DTYPE)
        if buffer_link.put(np.array([frame]), timeout=0.1):
            put_ids.append(frame_id)
            total_put.value += 1
            frame_id_counter += 1
        time.sleep(0.01) # Control put rate

    buffer_link.close()

# Consumer process function
def consumer(buffer: ProcessSafeSharedRingBuffer, gotten_ids: list, total_gotten: Synchronized, stop_event: mp_sync.Event):
    buffer_link = ProcessSafeSharedRingBuffer(create=False, source_buffer=buffer)
    while not stop_event.is_set():
        frames_list = buffer_link.get(1, timeout=0.1)
        if frames_list:
            frame = frames_list[0]
            frame_id = frame[0, 0, 0] # Assuming frame ID is in the first element
            gotten_ids.append(frame_id)
            total_gotten.value += 1
            buffer_link.release_last_got_data() # Explicitly release
        time.sleep(0.015) # Control get rate

    buffer_link.close()

# Peeker process function
def peeker(buffer: ProcessSafeSharedRingBuffer, put_ids: list, gotten_ids: list, errors: list, successful_peeks: Synchronized, attempted_peeks: Synchronized, total_put: Synchronized, total_gotten: Synchronized, stop_event: mp_sync.Event):
    buffer_link = ProcessSafeSharedRingBuffer(create=False, source_buffer=buffer)
    while not stop_event.is_set():
        attempted_peeks.value += 1
        try:
            # Estimate available frames (this is an approximation in concurrent scenario)
            # A more accurate way would require reading buffer's internal state under lock,
            # but we'll rely on put/get counts for a rough estimate for test parameter generation.
            # The actual validation will check against put_ids.
            estimated_available = total_put.value - total_gotten.value
            if estimated_available <= 0:
                time.sleep(0.005) # Wait a bit if buffer seems empty
                continue

            # Generate random offset and num_frames within a plausible range
            max_offset = min(LARGE_BUFFER_CAPACITY - 1, max(0, estimated_available - 1))
            offset = np.random.randint(0, max_offset + 1) if max_offset >= 0 else 0
            max_num_frames = max(1, offset + 1)
            num_frames = np.random.randint(1, min(max_num_frames + 1, LARGE_BUFFER_CAPACITY + 1)) # Ensure num_frames is at least 1 and not exceeding buffer capacity

            peeked_frames = buffer_link.peek_frames(offset=offset, num_frames=num_frames, timeout=0.05)

            if peeked_frames is not None:
                successful_peeks.value += 1
                if peeked_frames.shape[0] != num_frames:
                    errors.append(f"Peeked incorrect number of frames: Expected {num_frames}, got {peeked_frames.shape[0]}")
                    continue

                peeked_ids = [frame[0, 0, 0] for frame in peeked_frames]

                # Verify frame validity and continuity
                if peeked_ids:
                    # Check if all peeked IDs were actually put
                    if not all(id in put_ids for id in peeked_ids):
                         errors.append(f"Peeked invalid frame ID(s): {peeked_ids}")

                    # Check for continuity (considering modulo 256)
                    if len(peeked_ids) > 1:
                        for i in range(len(peeked_ids) - 1):
                            expected_next_id = np.uint8((int(peeked_ids[i]) + 1) % 256)
                            if peeked_ids[i+1] != expected_next_id:
                                errors.append(f"Peeked non-consecutive frames: {peeked_ids[i]} followed by {peeked_ids[i+1]}")

        except Exception as e:
            errors.append(f"Exception in peeker: {e}")

        time.sleep(0.005) # Control peek rate

    buffer_link.close()


# Assuming ProcessSafeSharedRingBuffer is in the parent directory or in a module that can be imported
# Adjust the import path if necessary
try:
    from shared_ring_buffer_v3 import ProcessSafeSharedRingBuffer
except ImportError:
    # Fallback for different project structures
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from shared_ring_buffer_v3 import ProcessSafeSharedRingBuffer

BUFFER_CAPACITY = 5
FRAME_SIZE = (10, 10, 3) # Example frame size (height, width, channels)
FRAME_DTYPE = np.uint8

class TestSharedRingBufferV3Peek(unittest.TestCase):

    def setUp(self):
        """Set up a new buffer before each test."""
        self.buffer = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=BUFFER_CAPACITY, frame_shape=FRAME_SIZE, dtype=FRAME_DTYPE)

    def tearDown(self):
        """Clean up the buffer after each test."""
        if self.buffer:
            self.buffer.close()
            # Only unlink in the process that created it (the main test runner process)
            if hasattr(self.buffer, '_metadata_shm') and self.buffer._metadata_shm:
                 try:
                     self.buffer.unlink()
                 except FileNotFoundError:
                     pass # Already unlinked
                 except Exception as e:
                     print(f"Error during unlink in tearDown: {e}")


    def test_peek_frames_empty_buffer(self):
        """Test peeking from an empty buffer."""
        peeked_frames = self.buffer.peek_frames(offset=0, num_frames=1)
        self.assertIsNone(peeked_frames, "Peeking from an empty buffer should return None")

    def test_peek_frames_valid_offset_num_frames(self):
        """Test peeking with valid offset and num_frames."""
        # Put some frames into the buffer
        frames_to_put = []
        for i in range(BUFFER_CAPACITY):
            frame = (np.ones(FRAME_SIZE, dtype=FRAME_DTYPE) * (i + 1)).astype(FRAME_DTYPE)
            frames_to_put.append(frame)
            self.buffer.put(np.array([frame])) # Put one by one

        # Peek the last frame (offset=0, num_frames=1)
        peeked_last = self.buffer.peek_frames(offset=0, num_frames=1)
        self.assertIsNotNone(peeked_last)
        self.assertEqual(peeked_last.shape, (1,) + FRAME_SIZE)
        self.assertTrue(np.array_equal(peeked_last[0], frames_to_put[-1]))

        # Peek the first frame (offset=BUFFER_CAPACITY-1, num_frames=1)
        peeked_first = self.buffer.peek_frames(offset=BUFFER_CAPACITY-1, num_frames=1)
        self.assertIsNotNone(peeked_first)
        self.assertEqual(peeked_first.shape, (1,) + FRAME_SIZE)
        self.assertTrue(np.array_equal(peeked_first[0], frames_to_put[0]))

        # Peek a range of frames in the middle (offset=2, num_frames=2)
        # This should peek frames at index BUFFER_CAPACITY - 1 - 2 = BUFFER_CAPACITY - 3 and BUFFER_CAPACITY - 2
        # For BUFFER_CAPACITY = 5, this is index 2 and 3 (0-indexed)
        peeked_range = self.buffer.peek_frames(offset=2, num_frames=2)
        self.assertIsNotNone(peeked_range)
        self.assertEqual(peeked_range.shape, (2,) + FRAME_SIZE)
        self.assertTrue(np.array_equal(peeked_range[0], frames_to_put[BUFFER_CAPACITY - 3]))
        self.assertTrue(np.array_equal(peeked_range[1], frames_to_put[BUFFER_CAPACITY - 2]))

    def test_peek_frames_wrap_around(self):
        """Test peeking when the peek range wraps around the buffer."""
        # Put frames to fill the buffer
        initial_frames = []
        for i in range(BUFFER_CAPACITY):
            frame = (np.ones(FRAME_SIZE, dtype=FRAME_DTYPE) * (i + 1)).astype(FRAME_DTYPE)
            initial_frames.append(frame)
            self.buffer.put(np.array([frame]))

        # Get 2 frames to make space
        gotten_frames = self.buffer.get(2)[0] # unpack
        self.assertIsNotNone(gotten_frames)
        self.assertEqual(len(gotten_frames), 2)
        self.assertTrue(np.array_equal(gotten_frames[0], initial_frames[0]))
        self.assertTrue(np.array_equal(gotten_frames[1], initial_frames[1]))

        # Release the gotten frames
        released_count = self.buffer.release_last_got_data()
        self.assertEqual(released_count, 2)

        # Put 2 new frames, causing wrap around
        new_frames = []
        for i in range(2):
            frame = (np.ones(FRAME_SIZE, dtype=FRAME_DTYPE) * (BUFFER_CAPACITY + 1 + i)).astype(FRAME_DTYPE)
            new_frames.append(frame)
            self.buffer.put(np.array([frame]))

        # At this point, the buffer should contain frames:
        # [initial_frames[2], initial_frames[3], initial_frames[4], new_frames[0], new_frames[1]]
        # Values: [3, 4, 5, 6, 7] if BUFFER_CAPACITY is 5

        # Peek the last 3 frames (offset=2, num_frames=3)
        # This should peek frames with values 5, 6, 7 (initial_frames[4], new_frames[0], new_frames[1])
        peeked_wrap = self.buffer.peek_frames(offset=2, num_frames=3)
        self.assertIsNotNone(peeked_wrap)
        self.assertEqual(peeked_wrap.shape, (3,) + FRAME_SIZE)

        expected_peeked_frames = [initial_frames[4], new_frames[0], new_frames[1]]
        self.assertTrue(np.array_equal(peeked_wrap[0], expected_peeked_frames[0]))
        self.assertTrue(np.array_equal(peeked_wrap[1], expected_peeked_frames[1]))
        self.assertTrue(np.array_equal(peeked_wrap[2], expected_peeked_frames[2]))


    def test_peek_frames_invalid_input(self):
        """Test peeking with invalid offset or num_frames."""
        # Put some frames
        for i in range(3):
            frame = (np.ones(FRAME_SIZE, dtype=FRAME_DTYPE) * (i + 1)).astype(FRAME_DTYPE)
            self.buffer.put(np.array([frame]))

        available_frames = self.buffer.unread_count # Should be 3

        # Invalid offset (negative)
        peeked_invalid_offset_neg = self.buffer.peek_frames(offset=-1, num_frames=1)
        self.assertIsNone(peeked_invalid_offset_neg, "Peeking with negative offset should return None")

        # Invalid offset (greater than or equal to available frames)
        peeked_invalid_offset_large = self.buffer.peek_frames(offset=available_frames, num_frames=1)
        self.assertIsNone(peeked_invalid_offset_large, "Peeking with offset >= available frames should return None")

        # Invalid num_frames (zero)
        peeked_invalid_num_zero = self.buffer.peek_frames(offset=0, num_frames=0)
        self.assertIsNone(peeked_invalid_num_zero, "Peeking with num_frames=0 should return None")

        # Invalid num_frames (negative)
        peeked_invalid_num_neg = self.buffer.peek_frames(offset=0, num_frames=-1)
        self.assertIsNone(peeked_invalid_num_neg, "Peeking with negative num_frames should return None")

        # Invalid num_frames (greater than offset + 1)
        peeked_invalid_num_large = self.buffer.peek_frames(offset=1, num_frames=3) # offset=1 means 2nd last frame, can peek 1 or 2 frames
        self.assertIsNone(peeked_invalid_num_large, "Peeking with num_frames > offset + 1 should return None")

    # Note: Testing concurrent peek operations reliably in a single process unit test is challenging.
    # The current implementation's check for protected_count > 0 is a basic mechanism.
    # A more robust test would involve multiple processes. For now, we test the basic check.
    def test_peek_frames_another_peek_in_progress(self):
        """Test peeking when another peek operation is in progress (simulated)."""
        # Put a frame
        frame = (np.ones(FRAME_SIZE, dtype=FRAME_DTYPE) * 100).astype(FRAME_DTYPE)
        self.buffer.put(np.array([frame]))

        # Simulate another peek in progress by manually setting protected_count
        # This requires accessing internal metadata, which is not ideal for unit tests,
        # but demonstrates testing the intended behavior of the check.
        # In a real scenario, this would be tested with actual concurrent processes.
        with self.buffer._pointer_lock:
            self.buffer._metadata_ctypes.protected_count = 1

        # Attempt to peek
        peeked_frames = self.buffer.peek_frames(offset=0, num_frames=1, timeout=0.1) # Use a short timeout
        self.assertIsNone(peeked_frames, "Peeking when protected_count > 0 should return None")

        # Reset protected_count
        with self.buffer._pointer_lock:
            self.buffer._metadata_ctypes.protected_count = 0

    def test_peek_frames_after_get_before_release(self):
        """Test peeking after get but before release_last_got_data."""
        # Put some frames
        frames_to_put = []
        for i in range(3):
            frame = (np.ones(FRAME_SIZE, dtype=FRAME_DTYPE) * (i + 1)).astype(FRAME_DTYPE)
            frames_to_put.append(frame)
            self.buffer.put(np.array([frame]))

        # Get one frame
        gotten_frames = self.buffer.get(1)[0] # Remember to unpack frames
        self.assertIsNotNone(gotten_frames)
        self.assertEqual(len(gotten_frames), 1)
        self.assertTrue(np.array_equal(gotten_frames[0], frames_to_put[0]))

        # Peek the last frame (should be the 3rd frame put)
        peeked_last = self.buffer.peek_frames(offset=0, num_frames=1)
        self.assertIsNotNone(peeked_last)
        self.assertEqual(peeked_last.shape, (1,) + FRAME_SIZE)
        self.assertTrue(np.array_equal(peeked_last[0], frames_to_put[2]))

        # Peek the frame that was just gotten (should still be accessible via peek)
        # The frames in the buffer are conceptually [frame1, frame2, frame3]
        # After get(1), frame1 is "gotten" but not released.
        # The available frames for peek are still [frame1, frame2, frame3]
        # offset=2 should peek the first frame (frame1)
        peeked_gotten = self.buffer.peek_frames(offset=2, num_frames=1)
        self.assertIsNotNone(peeked_gotten)
        self.assertEqual(peeked_gotten.shape, (1,) + FRAME_SIZE)
        self.assertTrue(np.array_equal(peeked_gotten[0], frames_to_put[0]))


    def test_peek_frames_after_release(self):
        """Test peeking after release_last_got_data."""
        # Put some frames
        frames_to_put = []
        for i in range(3):
            frame = (np.ones(FRAME_SIZE, dtype=FRAME_DTYPE) * (i + 1)).astype(FRAME_DTYPE)
            frames_to_put.append(frame)
            self.buffer.put(np.array([frame]))

        # Get one frame and release it
        gotten_frames = self.buffer.get(1)
        self.assertIsNotNone(gotten_frames)
        released_count = self.buffer.release_last_got_data()
        self.assertEqual(released_count, 1)

        # Peek the last frame (should be the 3rd frame put)
        peeked_last = self.buffer.peek_frames(offset=0, num_frames=1)
        self.assertIsNotNone(peeked_last)
        self.assertEqual(peeked_last.shape, (1,) + FRAME_SIZE)
        self.assertTrue(np.array_equal(peeked_last[0], frames_to_put[2]))

        # Attempt to peek the frame that was released (should not be possible)
        # After releasing 1 frame, the available frames for peek are [frame2, frame3]
        # offset=1 should peek the first available frame (frame2)
        peeked_released = self.buffer.peek_frames(offset=1, num_frames=1)
        self.assertIsNotNone(peeked_released)
        self.assertEqual(peeked_released.shape, (1,) + FRAME_SIZE)
        self.assertTrue(np.array_equal(peeked_released[0], frames_to_put[1]))

        # Attempt to peek with an offset that would include the released frame (should fail)
        peeked_invalid_offset = self.buffer.peek_frames(offset=2, num_frames=1) # offset=2 would be the original frame 1
        self.assertIsNone(peeked_invalid_offset, "Peeking with offset including released frame should return None")


    def test_concurrent_peek_put_get(self):
        """Test peek correctness in a concurrent put/get/peek scenario."""

        print("\nRunning concurrent peek test...")
        large_buffer = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=LARGE_BUFFER_CAPACITY, frame_shape=FRAME_SIZE, dtype=FRAME_DTYPE)

        with mp.Manager() as manager:
            # Shared data structures
            put_frame_ids = manager.list()
            gotten_frame_ids = manager.list()
            peeker_errors = manager.list()
            successful_peeks = manager.Value('i', 0)
            attempted_peeks = manager.Value('i', 0)
            total_frames_put = manager.Value('i', 0)
            total_frames_gotten = manager.Value('i', 0)

            # Event to signal processes to stop
            stop_event = manager.Event()

            # Create processes
            producer_p = mp.Process(target=producer, args=(large_buffer, put_frame_ids, total_frames_put, stop_event))
            consumer_c = mp.Process(target=consumer, args=(large_buffer, gotten_frame_ids, total_frames_gotten, stop_event))
            peeker_pk = mp.Process(target=peeker, args=(large_buffer, put_frame_ids, gotten_frame_ids, peeker_errors, successful_peeks, attempted_peeks, total_frames_put, total_frames_gotten, stop_event))

            # Start processes
            producer_p.start()
            consumer_c.start()
            peeker_pk.start()

            # Run for a duration
            test_duration = 10 # seconds
            time.sleep(test_duration)

            # Signal processes to stop
            stop_event.set()

            # Wait for processes to finish
            producer_p.join()
            consumer_c.join()
            peeker_pk.join()

            # Assertions
            self.assertEqual(len(peeker_errors), 0, f"Peeker reported errors: {peeker_errors}")
            self.assertGreater(successful_peeks.value, 0, "Peeker did not successfully peek any frames")
            print(f"\nConcurrent Test Results:")
            print(f"Total frames put: {total_frames_put.value}")
            print(f"Total frames gotten: {total_frames_gotten.value}")
            print(f"Attempted peeks: {attempted_peeks.value}")
            print(f"Successful peeks: {successful_peeks.value}")
            print(f"Peek success rate: {successful_peeks.value / attempted_peeks.value if attempted_peeks.value > 0 else 0:.2f}")


        # Clean up the larger buffer
        large_buffer.close()
        try:
            large_buffer.unlink()
        except FileNotFoundError:
            pass # Already unlinked
        except Exception as e:
            print(f"Error during unlink of large buffer: {e}")


if __name__ == '__main__':
    unittest.main()
