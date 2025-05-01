import unittest
import numpy as np
import hashlib
import multiprocessing as mp
import multiprocessing.synchronize as mp_sync # Import multiprocessing.synchronize for Event
import time
import random # Import random
import warnings # Import warnings
import pytest # Import pytest
from typing import List, Optional, Tuple

# Assuming videoencoder.py and shared_ring_buffer.py are in the parent directory or accessible in the Python path
from videoencoder import BaseVideoEncoder
from shared_ring_buffer import ProcessSafeSharedRingBuffer
from videoencoder import X264Encoder # Import X264Encoder from the new file

# Define a simple dummy encoder for testing
class DummyVideoEncoder(BaseVideoEncoder):
    """
    A dummy video encoder for testing BaseVideoEncoder's IPC and process management.
    Instead of encoding, it calculates the hash of the frame and sends it back.
    """
    def __init__(self,
                 output_path: str,
                 frame_size: Tuple[int, int, int],
                 buffer_capacity: int = 10,
                 result_queue: Optional[mp.Queue] = None, # Queue to send results back
                 pause_event: Optional[mp_sync.Event] = None): # Optional event to pause worker
        super().__init__(output_path, frame_size, buffer_capacity)
        self._result_queue = result_queue
        self._pause_event = pause_event # Store the pause event
        self._initialized = False

    def _initialize_encoder(self):
        """Dummy initialization."""
        print(f"DummyVideoEncoder worker ({mp.current_process().pid}): Initializing dummy encoder.")
        # Simulate some initialization time
        time.sleep(0.1)
        self._initialized = True
        print(f"DummyVideoEncoder worker ({mp.current_process().pid}): Dummy encoder initialized.")

    def _encode_frames(self, frames_list: List[np.ndarray]):
        """Calculate frame hashes and send them back."""
        if not self._initialized:
            print(f"DummyVideoEncoder worker ({mp.current_process().pid}): Encoder not initialized, skipping frames.")
            return

        # Check for pause event before processing the frames
        if self._pause_event and not self._pause_event.is_set():
            print(f"DummyVideoEncoder worker ({mp.current_process().pid}): Paused, waiting for event.")
            self._pause_event.wait() # Wait until the event is set
            print(f"DummyVideoEncoder worker ({mp.current_process().pid}): Resumed.")

        for frames in frames_list:
            for frame in frames:
                # Calculate hash of the frame
                frame_hash = hashlib.sha256(frame.tobytes()).hexdigest()
                print(f"DummyVideoEncoder worker ({mp.current_process().pid}): Encoded frame with hash {frame_hash[:8]}...")
                print(f"DummyVideoEncoder worker ({mp.current_process().pid}): Frame shape: {frame.shape}, "
                      f"frame head: {frame[0,0,0]}, frame tail: {frame[-1,-1,-1]}")

                # Send hash back to the main process
                if self._result_queue:
                    try:
                        self._result_queue.put(frame_hash)
                        print(f"DummyVideoEncoder worker ({mp.current_process().pid}): Sent hash back to main process.")
                    except Exception as e:
                        print(f"DummyVideoEncoder worker ({mp.current_process().pid}): Error sending result back: {e}")

    def _uninitialize_encoder(self):
        """Dummy uninitialization."""
        print(f"DummyVideoEncoder worker ({mp.current_process().pid}): Uninitializing dummy encoder.")
        # Simulate some uninitialization time
        time.sleep(0.1)
        print(f"DummyVideoEncoder worker ({mp.current_process().pid}): Dummy encoder uninitialized.")

    # The _worker method remains unchanged as per user feedback.


# Unit test class for BaseVideoEncoder
class TestBaseVideoEncoder(unittest.TestCase):

    def setUp(self):
        """Set up for each test."""
        self.output_path = "dummy_output.mp4"
        self.frame_size = (48, 64, 3) # Example frame size (height, width, channels)
        self.buffer_capacity = 5
        self.result_queue = mp.Queue() # Queue to receive results from worker

    def tearDown(self):
        """Clean up after each test."""
        # Ensure the queue is closed and joined
        if self.result_queue:
            self.result_queue.close()
            self.result_queue.join_thread()

    def _create_dummy_frame(self, value, shape, dtype):
        """Helper function to create a dummy frame."""
        return np.full(shape, value, dtype=dtype)

    def _generate_dummy_frames(self, start_value, num_frames, use_random=False, random_seed=None):
        """Helper function to create multiple dummy frames."""
        frames = np.zeros((num_frames,) + self.frame_size, dtype=np.uint8) # Use class attributes for shape and dtype
        dtype_info = np.iinfo(np.uint8) # Use uint8 dtype info

        if use_random:
            for i in range(num_frames):
                effective_value = start_value + i
                # Use a combination of random_seed and effective_value for reproducibility
                seed = random_seed + effective_value if random_seed is not None else effective_value
                rng = np.random.default_rng(seed)
                # Generate a single random value for this frame within the dtype range
                random_value = rng.integers(low=dtype_info.min, high=dtype_info.max + 1, size=1, dtype=np.uint8)[0]
                frames[i] = self._create_dummy_frame(random_value, self.frame_size, np.uint8) # Use class attributes
        else:
            for i in range(num_frames):
                # Use start_value + i as the basis for the frame content
                # Ensure value stays within uint8 range by taking modulo
                value = (start_value + i) % (dtype_info.max + 1)
                frames[i] = self._create_dummy_frame(value, self.frame_size, np.uint8) # Use class attributes
        return frames

    # Test case for starting and stopping the encoder
    def test_start_stop(self):
        """Test that the encoder can be started and stopped."""
        encoder = DummyVideoEncoder(self.output_path, self.frame_size, self.buffer_capacity, self.result_queue)
        encoder.start()
        self.assertTrue(encoder._running.is_set())
        self.assertIsNotNone(encoder._worker_process)
        self.assertTrue(encoder._worker_process.is_alive())
        self.assertTrue(encoder._worker_ready.wait(timeout=5)) # Wait for worker to be ready

        encoder.stop()
        self.assertFalse(encoder._running.is_set())
        if encoder._worker_process:
             encoder._worker_process.join(timeout=5)
             self.assertFalse(encoder._worker_process.is_alive())
        self.assertIsNone(encoder._ring_buffer) # Ensure ring buffer is unlinked

    # Test case for submitting frames and verifying they are processed
    def test_submit_frames(self):
        """Test submitting multiple frames and verifying processed hashes."""
        num_frames_to_submit = self.buffer_capacity + 2 # Submit more frames than buffer capacity
        encoder = DummyVideoEncoder(self.output_path, self.frame_size, self.buffer_capacity, self.result_queue)
        encoder.start()
        self.assertTrue(encoder._worker_ready.wait(timeout=5))

        submitted_hashes = []
        frames_to_submit = self._generate_dummy_frames(0, num_frames_to_submit)
        for frame in frames_to_submit:
            submitted_hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
            print(f"Submitting frame with hash: {submitted_hashes[-1][:8]}...")
            print(f"Submitting: Frame shape: {frame.shape}, frame head: {frame[0,0,0]}, frame tail: {frame[-1,-1,-1]}")
            encoder.submit_frame(frame)

        # Wait for all frames to be processed and results sent back
        received_hashes = []
        # Use a timeout to prevent infinite waiting if something goes wrong
        timeout = 15 # seconds
        start_time = time.time()
        while len(received_hashes) < num_frames_to_submit and (time.time() - start_time) < timeout:
            try:
                # Use a short timeout for getting from the queue to avoid blocking indefinitely
                hash_val = self.result_queue.get(timeout=1)
                received_hashes.append(hash_val)
            except:
                # Queue is empty, continue waiting
                pass

        encoder.stop()

        self.assertEqual(len(received_hashes), num_frames_to_submit, "Not all frames were processed")
        self.assertEqual(submitted_hashes, received_hashes, "Processed frame hashes do not match submitted hashes")

    # Test case for buffer full waiting behavior
    def test_buffer_full_waits(self):
        """Test that submit_frame waits when the buffer is full."""
        # Create a pause event for this test case
        self._pause_worker_event = mp.Event()
        # Ensure the event is cleared initially to pause the worker
        self._pause_worker_event.clear()

        encoder = DummyVideoEncoder(self.output_path, self.frame_size, self.buffer_capacity, self.result_queue, pause_event=self._pause_worker_event)
        encoder.start()
        self.assertTrue(encoder._worker_ready.wait(timeout=5))

        # Fill the buffer
        for i in range(self.buffer_capacity):
            # Call _generate_dummy_frames to get a batch, then take the first frame
            frame = self._generate_dummy_frames(i, 1)[0]
            encoder.submit_frame(frame)

        # Try to submit one more frame - this should block because the worker is paused
        # Generate one frame using _generate_dummy_frames
        frame_to_block = self._generate_dummy_frames(self.buffer_capacity, 1)[0]

        # Use a short timeout to verify that it *tries* to wait
        # We expect it to time out because the worker isn't consuming frames yet
        start_time = time.time()
        # submit_frame expects a single frame, not a batch
        success = encoder.submit_frame(frame_to_block)
        end_time = time.time()

        # Set the pause event to allow the worker to resume and the test to clean up
        self._pause_worker_event.set()

        encoder.stop()

        # Check that the submit_frame call took longer than a minimal time,
        # indicating it likely waited, and that it returned False due to timeout
        self.assertFalse(success, "submit_frame should return False on timeout")
        self.assertGreaterEqual(end_time - start_time, 5.0, "submit_frame did not wait when buffer was full") # Expecting at least the timeout duration

    # Test case for buffer empty waiting behavior
    def test_buffer_empty_waits(self):
        """Test that the worker waits when the buffer is empty."""
        encoder = DummyVideoEncoder(self.output_path, self.frame_size, self.buffer_capacity, self.result_queue)
        encoder.start()
        self.assertTrue(encoder._worker_ready.wait(timeout=5))

        # The worker should now be waiting for frames.
        # We can verify this by trying to get a result from the queue with a short timeout.
        # We expect it to time out because no frames have been submitted yet.
        start_time = time.time()
        try:
            # Attempt to get a result with a short timeout
            self.result_queue.get(timeout=1.0)
            # If we get here, it means a result was available, which is not expected
            self.fail("Worker processed a frame when the buffer should have been empty.")
        except:
            # This is the expected behavior - the get call should time out
            pass
        end_time = time.time()

        encoder.stop()

        # Check that the get call in the worker likely waited for the timeout duration
        self.assertGreaterEqual(end_time - start_time, 1.0, "Worker did not wait when buffer was empty.")

    # Test case for cleanup on stop
    def test_cleanup_on_stop(self):
        """Test that resources are properly cleaned up on stop."""
        encoder = DummyVideoEncoder(self.output_path, self.frame_size, self.buffer_capacity, self.result_queue)
        encoder.start()
        self.assertTrue(encoder._worker_ready.wait(timeout=5))

        # Submit a frame to ensure the ring buffer is used
        frame = self._generate_dummy_frames(0, 1)[0]
        encoder.submit_frame(frame)

        encoder.stop()

        # Verify that the ring buffer is unlinked (checked in test_start_stop, but good to be explicit)
        self.assertIsNone(encoder._ring_buffer)

        # Additional checks for resource cleanup can be added here if needed,
        # e.g., trying to access shared memory by name (requires knowing the name, which might be tricky)
        # For now, relying on the ring buffer's unlink method and process termination.

    # Test case for worker initialization error.
    @pytest.mark.skip(reason="Related to specific encoder implementations")
    def test_worker_initialization_error(self):
        """Test that the main process handles worker initialization errors."""
        # Modify DummyVideoEncoder to raise an error during initialization
        class ErrorInitDummyVideoEncoder(DummyVideoEncoder):
            def _initialize_encoder(self):
                raise RuntimeError("Simulated initialization error")

        encoder = ErrorInitDummyVideoEncoder(self.output_path, self.frame_size, self.buffer_capacity, self.result_queue)

        # Starting the encoder should catch the error and stop
        with self.assertRaises(Exception): # Expecting an exception during start
             encoder.start()

        # Verify that the worker process is not alive
        if encoder._worker_process:
             encoder._worker_process.join(timeout=5)
             self.assertFalse(encoder._worker_process.is_alive())

        # Verify that resources are cleaned up
        self.assertIsNone(encoder._ring_buffer)

    # Test case for worker encoding error
    @pytest.mark.skip(reason="Related to specific encoder implementations")
    def test_worker_encoding_error(self):
        """Test that the worker handles encoding errors and continues."""
        # Modify DummyVideoEncoder to raise an error during encoding for a specific frame
        class ErrorEncodeDummyVideoEncoder(DummyVideoEncoder):
            def _encode_frame(self, frame: np.ndarray):
                # Simulate encoding error for a specific frame
                frame_hash = hashlib.sha256(frame.tobytes()).hexdigest()
                if frame_hash == hashlib.sha256(self._generate_dummy_frames(1 % (np.iinfo(np.uint8).max + 1), 1)[0].tobytes()).hexdigest():
                    warnings.warn("Simulated encoding error during _encode_frame")
                    print(f"DummyVideoEncoder worker ({mp.current_process().pid}): Simulated encoding error, skipping frame.")
                    return # Skip the erroneous frame
                super()._encode_frame(frame) # Process other frames normally

        encoder = ErrorEncodeDummyVideoEncoder(self.output_path, self.frame_size, self.buffer_capacity, self.result_queue)
        encoder.start()
        self.assertTrue(encoder._worker_ready.wait(timeout=5))

        num_frames_to_submit = 3
        submitted_hashes = []
        frames_to_submit = self._generate_dummy_frames(0, num_frames_to_submit)
        with self.assertWarnsRegex(UserWarning, "Simulated encoding error during _encode_frame"):
            for frame in frames_to_submit:
                submitted_hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
                encoder.submit_frame(frame)

        # Wait for frames to be processed (excluding the one with the error)
        received_hashes = []
        timeout = 10 # seconds
        start_time = time.time()
        # We expect num_frames_to_submit - 1 successful encodings
        while len(received_hashes) < num_frames_to_submit - 1 and (time.time() - start_time) < timeout:
            try:
                hash_val = self.result_queue.get(timeout=0.1)
                received_hashes.append(hash_val)
            except:
                pass

        encoder.stop()

        # Verify that the worker process is still alive after the error (it should continue)
        if encoder._worker_process:
             self.assertFalse(encoder._worker_process.is_alive(), "Worker process should have terminated after stop.")

        # Verify that the correct number of frames (excluding the error one) were processed
        self.assertEqual(len(received_hashes), num_frames_to_submit - 1, "Incorrect number of frames processed after encoding error.")

        # Verify that the hashes of the successfully processed frames match
        # Need to remove the hash of the frame that caused the error from submitted_hashes
        error_frame_hash = hashlib.sha256(self._generate_dummy_frames(1 % (np.iinfo(np.uint8).max + 1), 1)[0].tobytes()).hexdigest()
        expected_hashes = [h for h in submitted_hashes if h != error_frame_hash]
        self.assertEqual(sorted(expected_hashes), sorted(received_hashes), "Processed frame hashes do not match expected hashes after encoding error.")


if __name__ == '__main__':
    # To run this test file directly, use: python -m unittest tests/test_videoencoder.py
    unittest.main()
