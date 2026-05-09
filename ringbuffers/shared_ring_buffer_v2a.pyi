import ctypes
import multiprocessing.synchronize as mp_sync
import numpy as np
from _typeshed import Incomplete
from typing import overload

METADATA_SIZE: int

class Metadata(ctypes.Structure):
    """
    Metadata structure for the shared ring buffer.
    Using ctypes to maximize performance in pointer operations.
    Need to be explicitly released (del obj) in `close()`.
    """

class ProcessSafeSharedRingBuffer:
    """
    A process-safe high performance circular buffer, implemented using 
    two SharedMemory blocks: one for metadata and one for frame data.
    Uses a pointer lock to synchronize access to metadata.
    Allows one `put` and one `get` concurrently. Optimized for performance.

    v2a supports full serialization across multi-processes in `spawn` mode.
    Attach mode is not preferred. Please pass the initialized buffer to 
    subprocess directly.

    Notes:
        - For detailed instructions and limitation, refer to docstring of `put` and `get`.
    """
    @overload
    def __init__(self, create: bool, buffer_capacity: int, frame_shape: tuple[int, int, int], dtype: np.dtype = ..., source_buffer: None = None) -> None:
        """
        Initialize or attach to the shared ring buffer.

        Args:
            create: (bool), True, create the shared memory blocks and initialize metadata.
            buffer_capacity: (int), the maximum number of frames the buffer can hold.
            frame_shape: (Tuple[int, int, int]), expected frame size (height, width, channel).
            dtype: (np.dtype), data type of the frames. Defaults to np.uint8.
        """
    @overload
    def __init__(self, create: bool, buffer_capacity: None = None, frame_shape: None = None, dtype: None = None, source_buffer: ProcessSafeSharedRingBuffer = None) -> None:
        """
        Initialize or attach to the shared ring buffer.

        Args:
            create: (bool), False, attach to existing shared memory segments using source_buffer.
            source_buffer: (ProcessSafeSharedRingBuffer), the buffer instance created
                           in the main process, used for attaching in worker processes.
        """
    def put(self, frames: np.ndarray, timeout: float | None = None):
        """
        Puts frames into the shared ring buffer. Blocks if the buffer is full.
        Multiple `put` must only be called sequentially, or it will corrupt the buffer.
        Args:
            frames: (np.ndarray), (frame_num, frame_h, frame_w, frame_c), the frames to put.
            timeout: (Optional[float]), seconds to wait for space.
        Returns:
            True if successful, False if timeout occurred.
        """
    def get(self, get_frame_num: int, timeout: float | None = None) -> list[np.ndarray] | None:
        """
        Gets frames from the shared ring buffer. Blocks if the buffer has no enough frames to get.
        Args:
            frame_num: (int), the number of frames to get.
            timeout: (Optional[float]), seconds to wait for data.
        Returns:
            The frames as a list of multiple views of numpy array, or None if timeout occured.
            One or two views are returned, depending on whether the read pointer wraps around.
            The shape of a single array is (frame_num, frame_h, frame_w, frame_c).

        Note:
            The returned arrays are views of the shared memory, the lock state will be released 
            upon next `get` call **despite timeout**. If there're multiple `get` callers, 
            one should consider using a lock to control `get` calls across callers.
        """
    def release_last_got_data(self) -> int:
        """
        Release previously preserved data (returned by last `get` call) from the buffer.
        This is a public method of `_release_last_got_data`, which handles the lock.
        Returns:
            frames released. (int)
        """
    def peek_last_frame(self, timeout: float | None = 0.1) -> np.ndarray | None:
        """
        Gets a copy of the last frame written to the buffer without removing it.

        Args:
            timeout: (Optional[float]), seconds to wait for acquiring the lock. 
                     If None, block indefinitely, default 0.1s.
        Returns:
            A numpy ndarray containing a copy of the last written frame,
            or None if the buffer is currently empty or not initialized 
            or acquiring lock timeout.
        """
    @property
    def dtype(self) -> np.dtype:
        """Returns the dtype of the frame data in the buffer."""
    @property
    def metadata_name(self) -> str | None:
        """Returns the name of the metadata shared memory segment."""
    @property
    def data_name(self) -> str | None:
        """Returns the name of the data shared memory segment."""
    @property
    def unread_count(self) -> int:
        """Returns the number of unread items in the buffer."""
    @property
    def buffer_capacity(self) -> int:
        """Returns the total capacity of the buffer."""
    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """Returns the shape of a single frame in the buffer."""
    @property
    def occupied_count_debug(self) -> int:
        """Debug property: Returns the raw occupied_count metadata."""
    @property
    def last_get_count_debug(self) -> int:
        """Debug property: Returns the raw last_get_count metadata."""
    @property
    def pointer_lock(self) -> mp_sync.Lock | None: ...
    @property
    def data_available_condition(self) -> mp_sync.Condition | None: ...
    @property
    def space_available_condition(self) -> mp_sync.Condition | None: ...
    def reattach(self) -> None:
        """
        Re-attach to the shared memory segments in this process.

        Can be called after the buffer close() but the buffer is not freed globally.

        Noted that on Windows, if all processes have closed the buffer, the 
        shared memory segments will be freed globally.

        Raise:
            RuntimeError: if the buffer was not fully initialized or has unlinked.
            FileNotFoundError: if the shared memory segments are not found.
        """
    def close(self) -> None:
        """
        Detaches the current process from the shared memory segments.

        After calling `close()`, *this process* can no longer access the shared memory.
        The shared memory segments may persist in the system if other processes
        are still attached. This should be called by all processes using the buffer
        when they are finished with it.
        """
    def unlink(self) -> None:
        """
        Marks the shared memory segments for destruction.

        The actual destruction of the segments will occur only after all processes
        that were attached to them have detached (i.e., after all processes have
        called `close()` or terminated). This method should ONLY be called by the
        process that originally created the shared memory. A `close()` should be 
        called before calling `unlink()` in the creator process.
        """

BUFFER_CAPACITY: int
FRAME_SIZE: Incomplete
FRAME_DTYPE: Incomplete

def producer_process(source_buffer: ProcessSafeSharedRingBuffer): ...
def consumer_process(source_buffer: ProcessSafeSharedRingBuffer): ...
def peeker_process(source_buffer: ProcessSafeSharedRingBuffer):
    """A separate process that only peeks."""
