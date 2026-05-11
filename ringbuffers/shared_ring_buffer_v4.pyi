import ctypes
import multiprocessing.synchronize as mp_sync
import numpy as np
from .shared_ring_buffer_v4_types import BufferTicket as BufferTicket, METADATA_SIZE as METADATA_SIZE, Metadata as Metadata
from _typeshed import Incomplete
from loguru._logger import Logger
from numpy.typing import NDArray as NDArray
from typing import Callable, Literal, overload

class ProcessSafeSharedRingBuffer:
    """
    A process-safe high performance circular buffer, implemented using 
    two SharedMemory blocks: one for metadata and one for frame data.
    Uses a pointer lock to synchronize access to metadata.
    Allows one `put` and one `get` concurrently. Optimized for performance.

    Notes:
        - For detailed instructions and limitation, refer to docstring of `put` and `get`.
    """
    @overload
    def __init__(self, create: Literal[True] = ..., *, buffer_capacity: int, frame_shape: tuple[int, int, int], dtype: np.dtype = ..., source_buffer: None = None, inject_logger: Logger | None = None) -> None:
        """
        Initialize a new shared ring buffer.

        Args:
            create: (bool), True, create the shared memory blocks and initialize metadata.
            buffer_capacity: (int), the maximum number of frames the buffer can hold.
            frame_shape: (Tuple[int, int, int]), expected frame size (height, width, channel).
            dtype: (np.dtype), data type of the frames. Defaults to np.uint8.
            inject_logger: (Optional[Logger]), loguru logger to use for logging, 
                           requires `enqueue=True`. If None, logging is disabled.

        Raises:
            ValueError: If `buffer_capacity` or `frame_shape` is not provided.
            RuntimeError: If error occurs during shared memory creation.
        """
    @overload
    def __init__(self, create: Literal[False], *, buffer_capacity: None = None, frame_shape: None = None, dtype: None = None, source_buffer: ProcessSafeSharedRingBuffer = None, inject_logger: Logger | None = None) -> None:
        """
        Attach to a shared ring buffer.

        Args:
            create: (bool), False, attach to existing shared memory segments using source_buffer.
            source_buffer: (ProcessSafeSharedRingBuffer), the buffer instance created
                           in the main process, used for attaching in worker processes.
            inject_logger: (Optional[Logger]), loguru logger to use for logging, 
                           requires `enqueue=True`. If None, logging is disabled.

        Raises:
            ValueError: If `source_buffer` is not provided.
            ValueError: If `source_buffer` shared memory names are not available.
            RuntimeError: If `source_buffer` have smaller data buffer than the metadata 
                specified, or any other error occurs during shared memory attachment.
            FileNotFoundError: If shared memory segment not found during attachment.
        """
    def put(self, frames: np.ndarray, timeout: float | None = None) -> bool:
        """
        Puts frames into the shared ring buffer. Blocks if the buffer is full.
        Multiple `put` must only be called sequentially, or it will corrupt the buffer.
        If putting frames is smaller the declared size, only the head of buffer 
        slots will be filled.

        Args:
            frames: (np.ndarray), (num_frames, H, W, C), the frames to put.
                if ndim==2, will expand to (1, H, W, 1).
                if ndim==3, will expand to (1, H, W, C).
            timeout: (Optional[float]), seconds to wait for space.

        Returns:
            True if successful, False if timeout occurred.

        Raises:
            RuntimeError: If the shared buffer has been uninitialized.
            ValueError: If the frame count exceeds buffer capacity or frame size/dtype mismatch.
        """
    def get(self, get_frame_num: int, timeout: float | None = None) -> tuple[list[np.ndarray], BufferTicket] | None:
        """
        Get unread data for given number of frames. Since V4, the gotten data 
        will no longer be release automatically. Call `release()` to manually 
        release the frames after use. 
        If the frames being put is smaller than the declared size, the consumer
        should slice **the head of EACH frame** and reshape to restore the correct
        data layout.

        Args:
            get_frame_num: (int), the number of frames to get.
            timeout: (Optional[float]), seconds to wait for data.

        Returns:
            The frames as a tuple: (frames_list, got_frame_num), or None if timeout occured.
            The shape of a single NDArray is (num_frames, H, W, C).
        
        Raises:
            RuntimeError: If the shared buffer has been uninitialized.
        """
    @overload
    def release(self, release_num: int, /) -> int:
        """
        Manually release the oldest `release_num` frames by marking them ready 
        to be overwritten. Unread frames can also be released, in which case the
        next `get()` call will read from the first valid frame. So release with
        care.
        
        Args:
            release_num (int): The number of frames intended to release.
            
        Returns:
            int: The number of frames actually released.
        
        Raises:
            RuntimeError: If the shared buffer is not fully initialized.
            TypeError: If the argument type is invalid."""
    @overload
    def release(self, oldest_ticket: BufferTicket, /) -> int:
        """
        Manually release the **oldest** BufferTicket by marking them ready 
        to be overwritten. This API is a wrapper for `release(ticket.read_num)`.
        The buffer state will become unexpected if the passed ticket is not the 
        oldest one.
        
        Args:
            oldest_ticket (BufferTicket): The oldest BufferTicket to release.
            
        Returns:
            int: The number of frames actually released.

        Raises:
            RuntimeError: If the shared buffer is not fully initialized.
            TypeError: If the argument type is invalid.
        """
    @property
    def trigger_release(self) -> Callable[[int], None] | None: ...
    @trigger_release.setter
    def trigger_release(self, gc_func: Callable[[int], None]): ...
    @overload
    def read_from(self, index: int, length: int, /) -> list[np.ndarray]:
        """
        Get a view of data from given `index` and `length` without lock. 
        The data availability is upon whether it has been released by the caller.
        
        Args:
            index (int): The starting index of internal buffer (0 <= index < capacity)
            length (int): The number of frames to read (0 < length <= capacity)
            
        Returns:
            List[np.ndarray]: A list of frame views.
        """
    @overload
    def read_from(self, ticket: BufferTicket, /) -> list[np.ndarray]:
        """
        Get a view of data from given `BufferTicket`. Useful for distributing 
        ticket for later retrival.
        The data availability is upon whether it has been released by the caller.
        
        Args:
            ticket (BufferTicket): The BufferTicket indicating the data to got.
            
        Returns:
            List[np.ndarray]: A list of frame views.

        Raises:
            RuntimeError: If the shared buffer is not fully initialized.
            TypeError: If the argument type is invalid.
            ValueError: If the index or length is invalid.
        """
    @property
    def dtype(self) -> np.dtype:
        """Returns the dtype of the frame data in the buffer."""
    @property
    def metadata_name(self) -> str | None:
        """Returns the name of the metadata shared memory segment. None if already unlinked."""
    @property
    def data_name(self) -> str | None:
        """Returns the name of the data shared memory segment. None if already unlinked."""
    @property
    def buffer_capacity(self) -> int:
        """Returns the total capacity of the buffer."""
    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """Returns the shape of a single frame in the buffer."""
    @property
    def unread_full(self) -> bool:
        """Get whether all the data in buffer is unread."""
    @property
    def unread_full_(self) -> ctypes.c_int16:
        """Get whether all the data in buffer is unread, dirty read without lock."""
    @property
    def read_ptr(self) -> int:
        """The read pointer of the buffer"""
    @property
    def read_ptr_(self) -> ctypes.c_int64:
        """The read pointer of the buffer, dirty read without lock. Safe in 64bit system"""
    @property
    def write_ptr(self) -> int:
        """Returns the write pointer of the buffer"""
    @property
    def write_ptr_(self) -> ctypes.c_int64:
        """Returns the write pointer of the buffer, dirty read without lock. Safe in 64bit system"""
    def occupied_count(self, timeout=None) -> int:
        """
        Get the number of valid items in the buffer.

        Args:
            timeout (float, optional): The timeout in seconds to wait for acquiring the lock.
                Default `None` will block until succeed and will not raise `TimeoutError`.

        Returns:
            int: The number of frames in the buffer.
        
        Raises:
            RuntimeError: If the shared buffer has been uninitialized or unexpected error occurs.
            TimeoutError: If the timeout expires before acquiring the lock.
        """
    @property
    def occupied_count_(self) -> int:
        """
        Read the occupied_count without lock, the value may not be the newest 
        due to memory barrier. Only safe in 64 bit system.
        """
    @property
    def is_full(self) -> bool:
        """Get whether the buffer is full"""
    @property
    def is_full_(self) -> bool:
        """Get whether the buffer is full, dirty read without lock."""
    def unread_count(self, timeout=None) -> int:
        """
        Get the number of unread items in the buffer.

        Args:
            timeout (float, optional): The timeout in seconds to wait for acquiring the lock.
                Default `None` will block until succeed and will not raise `TimeoutError`.

        Returns:
            int: The number of unread items in the buffer.

        Raises:
            RuntimeError: If the shared buffer has been uninitialized.
            TimeoutError: If the timeout expires before acquiring the lock.
        """
    @property
    def pointer_lock(self) -> mp_sync.Lock: ...
    @property
    def data_available_condition(self) -> mp_sync.Condition: ...
    @property
    def space_available_condition(self) -> mp_sync.Condition: ...
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
        The shared memory segments MAY persist in the system if other processes
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

        Raises:
            FileNotFoundError: If the shared memory segment is not found (probably alread unlinked)
            RuntimeError: Unexpected error in unlinking shared memory segment.
        """
    def __del__(self) -> None: ...

BUFFER_CAPACITY: int
FRAME_SIZE: Incomplete
FRAME_DTYPE: Incomplete

def producer_process(source_buffer: ProcessSafeSharedRingBuffer): ...
def consumer_process(source_buffer: ProcessSafeSharedRingBuffer): ...
def peeker_process(source_buffer: ProcessSafeSharedRingBuffer):
    """A separate process that only peeks."""
