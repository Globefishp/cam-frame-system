import pytest
import numpy as np
import time
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Condition, Event
from typing import Any, Tuple, Optional

# Import the NNAnalyzer class from the parent directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nnanalyzer import NNAnalyzer

# Mock NNAnalyzer subclass for testing
class MockNNAnalyzer(NNAnalyzer):
    def __init__(self, model_path: str, frame_shape: tuple[int, int, int], frame_dtype: np.dtype = np.uint8):
        super().__init__(model_path, frame_shape, frame_dtype)
        self._initialized = mp.Event()
        self._uninitialized = mp.Event()
        self._analysis_count = mp.Value('i', 0) # Counter for analysis calls

    def _initialize_analyzer(self):
        """Mock initialization."""
        print(f"MockNNAnalyzer ({mp.current_process().pid}) initializing...")
        # Simulate some initialization time
        time.sleep(0.1)
        self.model = "MockModel"
        self._initialized.set()
        print(f"MockNNAnalyzer ({mp.current_process().pid}) initialized.")

    def _analyze(self, frame: np.ndarray) -> Any:
        """Mock analysis: returns the sum of the frame data."""
        if self.model is None:
            print(f"Error: Mock model not loaded in worker process ({mp.current_process().pid}).")
            return None
        # Simulate some analysis time
        time.sleep(0.05)
        self._analysis_count.value += 1
        # Return a simple result, e.g., sum of frame elements
        return np.sum(frame)

    def _uninitialize_analyzer(self):
        """Mock uninitialization."""
        print(f"MockNNAnalyzer ({mp.current_process().pid}) uninitializing...")
        # Simulate some cleanup time
        time.sleep(0.1)
        self._uninitialized.set()
        print(f"MockNNAnalyzer ({mp.current_process().pid}) uninitialized.")

# Pytest fixtures
@pytest.fixture
def frame_params():
    return {
        "frame_shape": (100, 100, 3),
        "frame_dtype": np.uint8
    }

@pytest.fixture
def mock_analyzer(frame_params):
    analyzer = MockNNAnalyzer("dummy_model.pth", **frame_params)
    yield analyzer
    # Ensure cleanup in case a test fails before stop() is called
    if analyzer.working.is_set() or analyzer.worker_processes:
        analyzer.stop()
    # Manual cleanup of shared memory if it wasn't unlinked
    if analyzer.shm:
        try:
            analyzer.shm.close()
            analyzer.shm.unlink()
        except FileNotFoundError:
            pass # Already unlinked

# Test cases
def test_analyzer_start_stop(mock_analyzer):
    """Test if the analyzer can be started and stopped."""
    assert not mock_analyzer.working.is_set()
    assert not mock_analyzer.worker_processes

    mock_analyzer.start()

    assert mock_analyzer.working.is_set()
    assert mock_analyzer.worker_processes
    assert mock_analyzer.shm is not None
    assert mock_analyzer.shm_cond is not None
    # Wait for the worker to signal it's ready
    assert mock_analyzer.worker_ready.wait(timeout=5)

    mock_analyzer.stop()

    assert not mock_analyzer.working.is_set()
    assert not mock_analyzer.worker_processes
    assert mock_analyzer.shm is None # Should be unlinked
    assert mock_analyzer.shm_cond is None # Should be reset
    # Check if the mock uninitialization was called
    # This requires the worker process to have exited cleanly
    # We can't directly check the event in the main process after join,
    # but a successful stop implies the worker ran its cleanup.
    # More robust check might involve a file flag or similar IPC,
    # but for now, successful stop() is a good indicator.

def test_submit_and_get_result(mock_analyzer, frame_params):
    """Test submitting a frame and retrieving the analysis result."""
    mock_analyzer.start()
    frame = np.random.randint(0, 256, size=frame_params["frame_shape"], dtype=frame_params["frame_dtype"])
    expected_result = np.sum(frame) # Based on MockNNAnalyzer._analyze

    submission_timestamp = mock_analyzer.submit_frame(frame)
    assert submission_timestamp is not None

    # Give the worker some time to process and put the result
    time.sleep(0.2)

    result_tuple = mock_analyzer.get_result(timeout=5)
    assert result_tuple is not None
    result, received_timestamp, duration = result_tuple

    assert result == expected_result
    assert received_timestamp == submission_timestamp
    assert duration >= 0 # Duration should be non-negative

    mock_analyzer.stop()

def test_concurrent_submissions(mock_analyzer, frame_params):
    """Test submitting multiple frames concurrently."""
    mock_analyzer.start()
    num_frames = 10
    submitted_timestamps = []
    frames = [np.random.randint(0, 256, size=frame_params["frame_shape"], dtype=frame_params["frame_dtype"]) for _ in range(num_frames)]
    expected_results = [np.sum(frame) for frame in frames]

    # Submit frames rapidly
    for frame in frames:
        timestamp = mock_analyzer.submit_frame(frame)
        assert timestamp is not None
        submitted_timestamps.append(timestamp)
        # Add a small sleep to simulate some minimal delay between submissions
        time.sleep(0.01)

    # Give the worker time to process all frames
    time.sleep(num_frames * 0.1) # Adjust sleep based on mock analysis time

    received_results = []
    # Retrieve results
    for _ in range(num_frames):
        result_tuple = mock_analyzer.get_result(timeout=5)
        assert result_tuple is not None
        received_results.append(result_tuple)

    mock_analyzer.stop()

    # Verify the number of analyses performed by the worker
    assert mock_analyzer._analysis_count.value == num_frames

    # Verify results and timestamps
    # Sort received results by timestamp to match submission order
    received_results.sort(key=lambda x: x[1])

    for i in range(num_frames):
        result, received_timestamp, duration = received_results[i]
        assert result == expected_results[i]
        assert received_timestamp == submitted_timestamps[i]
        assert duration >= 0

def test_submit_before_start(mock_analyzer, frame_params):
    """Test submitting a frame before the analyzer is started."""
    frame = np.random.randint(0, 256, size=frame_params["frame_shape"], dtype=frame_params["frame_dtype"])
    submission_timestamp = mock_analyzer.submit_frame(frame)
    assert submission_timestamp is None # Should return None if not started

def test_submit_timeout(mock_analyzer, frame_params):
    """Test submit_frame with a timeout when the worker is not ready."""
    mock_analyzer.start()
    # Stop the worker from being ready by clearing the event
    # In a real scenario, this might happen if initialization takes too long
    mock_analyzer.worker_ready.clear()

    frame = np.random.randint(0, 256, size=frame_params["frame_shape"], dtype=frame_params["frame_dtype"])
    # Use a short timeout
    submission_timestamp = mock_analyzer.submit_frame(frame, timeout=0.1)
    assert submission_timestamp is None # Should timeout and return None

    mock_analyzer.stop()

def test_get_result_timeout(mock_analyzer):
    """Test get_result with a timeout when no results are available."""
    mock_analyzer.start()
    # Do not submit any frames

    # Use a short timeout to check for results
    result = mock_analyzer.get_result(timeout=0.1)
    assert result is None # Should timeout and return None

    mock_analyzer.stop()

