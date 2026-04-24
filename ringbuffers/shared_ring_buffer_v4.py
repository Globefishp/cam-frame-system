# ringbuffers/shared_ring_buffer_v4.py
# Sketched by Google Gemini 2.5 Flash exp, 3.1 Pro
# Manually corrected by Haiyun Huang 2026.

# Update log: 
#   - v1: Basic function
#   - v2: Add peek_last_frame (read while blocking put/get), 
#         optimize pointer related func to increase throughput
#   - v3: Refactor peek_last_frame, allowing parallel read/put/get, 
#         useful in peek-intensive applications. (slightly slower put/get 
#         performance than v2)
#         Potential future improvement: split _get/set_pointer_metadata to 
#         smaller functions, return less items (split according to the use case)
#   - v3a: Add a functionally pickle support for (and only for) cross-process
#          serialization. Hook the serialization process by __getstate__ and 
#          __setstate__ (same as v2a), and handle re-attachment automatically. 
#          See v2a for more detail. Not tested, only copy v2a lifecycle
#          management code. (260407)
#   - v4: Ring buffer now do not responsible for flow control
#         (i.e. ensure every `get()` call get the next data, 
#               get() call is sync or async between different consumer, 
#               the timing of data release, the number of data to release)
#         Now it only basic command: `put`, `get`, `read_from` and `release`.
#         The flow control is the responsibility of the upstream FrameServer 
#         which enables multiple consumers.
#         API design: 
#             put() (remains the same)
#             get() (remains the same except won't release last got frames)
#             release(Union[oldest_frame_num, BufferTicket]), release oldest frames
#             read_from(Union[index, BufferTicket], Optional[length]) -> List[NDArray], 
#                 Read from `ticket`.
#         FrameServer issue ticket to consumers and using RingBuffer properties to 
#             do flow control.
#         A GC function can be inject through `trigger_release` property.
#             The buffer can call this when `put()` needs more space.
#         Loguru logging when success. (260421)
#         Clean up unnecessary check, "let it crash".


# for infrastructure, raise instead of logging
# `try` block is for cleanup, not logging or retry. Raise original Exception type. 

# Requirements: python > 3.9

import ctypes
import warnings
import errno
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Any, Optional, Union, List, Callable, overload # Import overload

import multiprocessing as mp
import multiprocessing.shared_memory as mp_shm
import multiprocessing.synchronize as mp_sync

from loguru import logger as file_logger
from loguru._logger import Logger # for type hint only

from .shared_ring_buffer_v4_types import BufferTicket, Metadata, METADATA_SIZE


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
    def __init__(self,
                 create: bool,
                 buffer_capacity: int,
                 frame_shape: Tuple[int, int, int],
                 dtype: np.dtype = np.uint8,
                 source_buffer: None = None,
                 inject_logger: Optional[Logger] = None):
        """
        Initialize or attach to the shared ring buffer.

        Args:
            create: (bool), True, create the shared memory blocks and initialize metadata.
            buffer_capacity: (int), the maximum number of frames the buffer can hold.
            frame_shape: (Tuple[int, int, int]), expected frame size (height, width, channel).
            dtype: (np.dtype), data type of the frames. Defaults to np.uint8.
            inject_logger: (Optional[Logger]), loguru logger to use for logging, 
                           requires `enqueue=True`. If None, logging is disabled.
        """
        ... 

    @overload
    def __init__(self,
                 create: bool,
                 buffer_capacity: None = None,
                 frame_shape: None = None,
                 dtype: None = None,
                 source_buffer: 'ProcessSafeSharedRingBuffer' = None,
                 inject_logger: Optional[Logger] = None):
        """
        Initialize or attach to the shared ring buffer.

        Args:
            create: (bool), False, attach to existing shared memory segments using source_buffer.
            source_buffer: (ProcessSafeSharedRingBuffer), the buffer instance created
                           in the main process, used for attaching in worker processes.
            inject_logger: (Optional[Logger]), loguru logger to use for logging, 
                           requires `enqueue=True`. If None, logging is disabled.
        """
        ... 

    def __init__(self,
                 create: bool,
                 buffer_capacity: Optional[int] = None,
                 frame_shape: Optional[Tuple[int, int, int]] = None,
                 dtype: np.dtype = np.uint8, # Keep default for convenience, but logic handles None
                 source_buffer: Optional['ProcessSafeSharedRingBuffer'] = None,
                 inject_logger: Optional[Logger] = None):
        """
        Initialize or attach to the shared ring buffer.

        Args:
            create: (bool), if True, create the shared memory blocks and initialize metadata.
                    If False, attach to existing shared memory segments using source_buffer.
            buffer_capacity: (Optional[int]), the maximum number of frames the buffer can hold.
                             Required when create is True.
            frame_shape: (Optional[Tuple[int, int, int]]), expected frame size (height, width, channel).
                         Required when create is True.
            dtype: (np.dtype), data type of the frames. Defaults to np.uint8.
                   Used when create is True.
            source_buffer: (Optional[ProcessSafeSharedRingBuffer]), the buffer instance created
                           in the main process, used for attaching in worker processes.
                           Required when create is False.
            inject_logger: (Optional[Logger]), loguru logger to use for logging, 
                           requires `enqueue=True`. If None, logging is disabled.

        Raises:
            ValueError: If `buffer_capacity` or `frame_shape` is not provided when `create` is True.
            ValueError: If `source_buffer` is not provided when `create` is False.
            ValueError: If `source_buffer` shared memory names are not available.
            RuntimeError: If error occurs during shared memory creation or attachment.
            FileNotFoundError: If shared memory segment not found during attachment.
        """
        if inject_logger is not None:
            if isinstance(inject_logger, Logger):
                self._logger = inject_logger.bind(friendly_name="RingBuffer")
            else:
                raise TypeError("inject_logger must be a loguru.Logger instance.")
        else:
            self._logger = None
        logger = self._logger

        self._gc_func: Optional[Callable[[], None]] = None
        
        # Initialize attributes to None or default values, will be set properly later
        self._buffer_capacity: int = 0
        self._frame_shape: Tuple[int, int, int] = (0, 0, 0)
        self._dtype: np.dtype = np.dtype(np.uint8) # Default, will be overwritten
        self._frame_bytes: int = 0
        self._data_buffer_size: int = 0

        self._metadata_shm: Optional[mp_shm.SharedMemory] = None
        self._metadata_ctypes: Optional[Metadata] = None
        self._data_shm: Optional[mp_shm.SharedMemory] = None
        self._data_ndarr: Optional[NDArray] = None
        self._pointer_lock: Optional[mp_sync.Lock] = None
        self._data_available: Optional[mp_sync.Condition] = None
        self._space_available: Optional[mp_sync.Condition] = None


        if ctypes.sizeof(Metadata) > METADATA_SIZE:
             raise ValueError(f"METADATA_SIZE ({METADATA_SIZE}) is too small for metadata dtype ({self._metadata_dtype.itemsize})")

        if create:
            # --- Create Mode ---
            # Validate required parameters for creation
            if buffer_capacity is None or frame_shape is None:
                raise ValueError("buffer_capacity and frame_shape must be provided when create=True")
            # Set default dtype if not provided during creation
            effective_dtype = np.dtype(dtype) if dtype is not None else np.uint8

            # Assign attributes based on input parameters
            self._buffer_capacity = buffer_capacity
            self._frame_shape = frame_shape
            self._dtype = np.dtype(effective_dtype) # Ensure it's a numpy dtype
            self._frame_bytes = int(np.prod(self._frame_shape) * self._dtype.itemsize)
            self._data_buffer_size = self._buffer_capacity * self._frame_bytes

            # Create shared memory blocks and synchronization objects
            self._pointer_lock = mp.Lock()
            self._data_available = mp.Condition(self._pointer_lock)
            self._space_available = mp.Condition(self._pointer_lock)

            try:
                self._metadata_shm = mp_shm.SharedMemory(create=True, size=METADATA_SIZE)
                # Link metadata buffer to a ctypes structure for easier access.
                self._metadata_ctypes = Metadata.from_buffer(self._metadata_shm.buf)
                # Use the calculated _data_buffer_size for data shm creation
                self._data_shm = mp_shm.SharedMemory(create=True, size=self._data_buffer_size)
                self._data_ndarr = np.ndarray(self._data_buffer_size, dtype=np.uint8, buffer=self._data_shm.buf)

                # Initialize metadata in the metadata shared memory
                self._init_metadata()
                if logger: logger.success(f"Shared Ring Buffer created. Metadata SHM: {self._metadata_shm.name}, "
                      f"Data SHM: {self._data_shm.name}, size: {self._data_buffer_size} bytes")
            except OSError as e:
                self.close() # Clean up if creation fails
                raise RuntimeError(f"Error in creating shared memory segment for size: {self._data_buffer_size} bytes"
                    ", system resource has run out." if e.errno == errno.ENOMEM else ".") from e # TODO: Test case
            except Exception as e:
                self.close()
                raise RuntimeError(f"Unexpected error creating Shared Ring Buffer") from e

        else:
            # --- Attach Mode ---
            if source_buffer is None:
                raise ValueError("source_buffer must be provided when create=False")
            if source_buffer.metadata_name is None or source_buffer.data_name is None:
                raise ValueError("Source buffer shared memory names are not available.")

            # Get Synchronization Objects from Source Buffer
            self._pointer_lock = source_buffer.pointer_lock
            self._data_available = source_buffer.data_available_condition
            self._space_available = source_buffer.space_available_condition

            try:
                # Link shared memory, init ctypes and get properties from metadata.
                self._attach_mem_init(source_buffer.metadata_name, source_buffer.data_name)

                if logger: logger.success(f"Attached to Shared Ring Buffer. Metadata SHM: {self._metadata_shm.name}, "
                      f"Data SHM: {self._data_shm.name}, size: {self._data_buffer_size} bytes")

            except FileNotFoundError as e:
                self.close()
                raise FileNotFoundError(f"Shared memory segment not found: Metadata SHM: {self._metadata_shm.name}, "
                      f"Data SHM: {self._data_shm.name}, probably original buffer has expired.") from e
            except Exception as e:
                self.close()
                raise RuntimeError("Unexpected error attaching to Shared Ring Buffer") from e
    
    def _attach_mem_init(self, metadata_shm_name: str, data_shm_name: str):
        """
        Aux function that links SharedMemory and initialize Metadata structure
        and get python properties from source buffer.
        Multiprocessing synchronization objects are not touched here.
        """
        # 1. Attach to Metadata Shared Memory
        self._metadata_shm = mp_shm.SharedMemory(name=metadata_shm_name)
        # Link metadata buffer to a ctypes structure for easier access.
        self._metadata_ctypes = Metadata.from_buffer(self._metadata_shm.buf)

        # Read all metadata using _get_metadata
        capacity, frame_shape, frame_dtype, unread_full, read_ptr, write_ptr, occupied_count = self._get_metadata()

        # Set instance attributes based on read metadata
        self._buffer_capacity = capacity
        self._frame_shape = frame_shape
        self._dtype = frame_dtype
        self._frame_bytes = int(np.prod(self._frame_shape) * self._dtype.itemsize)
        self._data_buffer_size = self._buffer_capacity * self._frame_bytes

        # 3. Attach to Data Shared Memory
        self._data_shm = mp_shm.SharedMemory(name=data_shm_name)
        self._data_ndarr = np.ndarray(self._data_buffer_size, dtype=np.uint8, buffer=self._data_shm.buf)


        # 4. Verify Data Buffer Size
        if self._data_shm.size < self._data_buffer_size:
            raise RuntimeError(f"Attached data buffer size ({self._data_shm.size}) "
                f"is smaller than calculated size based on metadata ({self._data_buffer_size}).")
        # Multiprocessing synchronization objects are not touched here.

    def __getstate__(self) -> dict:
        """
        Custom pickle serialization that exclude process-local pointer exports.
        Metadata ctypes & Data NDArray will be re-attached after pickle.

        Note: ctypes.Structure mapped via .from_buffer(shm.buf) binds to this 
              process's virtual address thus is not available after pickle.
        """
        state = self.__dict__.copy()

        # Remove the process-local ctypes binding entirely; __setstate__ will rebuild it.
        del state['_metadata_ctypes']
        del state['_data_ndarr']

        return state

    def __setstate__(self, state: dict):
        """
        Custom pickle deserialization that will re-attach Metadata ctypes 
        to the shared memory segments.
        """
        # Restore dict
        self.__dict__.update(state)

        # Rebuild the ctypes mapping bound to THIS process's virtual address.
        self._metadata_ctypes = Metadata.from_buffer(self._metadata_shm.buf)
        self._data_ndarr = np.ndarray(self._data_buffer_size, dtype=np.uint8, buffer=self._data_shm.buf)

    def _init_metadata(self):
        """
        Initializes the metadata in the metadata shared memory.
        This method is intended to be called only once during the creation of the buffer
        in the main process to set the initial state of the metadata.
        """
        if self._metadata_shm is None:
            raise RuntimeError("Metadata shared memory not initialized.")

        # This method is only called during creation (under lock in put/get is not needed here)
        if self._metadata_ctypes is None:
            # Initialize metadata ctypes structure directly from buffer
            self._metadata_ctypes = Metadata.from_buffer(self._metadata_shm.buf)

        self._metadata_ctypes.capacity = self._buffer_capacity
        self._metadata_ctypes.frame_h = self._frame_shape[0]
        self._metadata_ctypes.frame_w = self._frame_shape[1]
        self._metadata_ctypes.frame_c = self._frame_shape[2]
        self._metadata_ctypes.dtype_kind = ord(self._dtype.kind) # Store character code (should be V)
        self._metadata_ctypes.dtype_bits = self._dtype.itemsize * 8 # Store bits
        self._metadata_ctypes.unread_full = 0
        self._metadata_ctypes.read_ptr = 0
        self._metadata_ctypes.write_ptr = 0
        self._metadata_ctypes.occupied_count = 0

    def _get_metadata(self):
        """
        Reads metadata from the metadata shared memory. Assumes pointer_lock is held.
        Note: This method involves reading and checking
             and should be AVOIDED in frequently called scenarios.
        """
        if self._metadata_ctypes is None:
            raise RuntimeError("Metadata shared memory not properly initialized.")

        # Access metadata ctypes directly

        capacity =           self._metadata_ctypes.capacity
        frame_h =            self._metadata_ctypes.frame_h
        frame_w =            self._metadata_ctypes.frame_w
        frame_c =            self._metadata_ctypes.frame_c
        dtype_kind =         self._metadata_ctypes.dtype_kind  # 后续需要转换为字符（如 'u'）
        dtype_bits =         self._metadata_ctypes.dtype_bits
        unread_full =        self._metadata_ctypes.unread_full
        read_ptr =           self._metadata_ctypes.read_ptr
        write_ptr =          self._metadata_ctypes.write_ptr
        count =              self._metadata_ctypes.occupied_count

        # Validate and reconstruct dtype
        if not (0 <= dtype_kind <= 127):
            raise ValueError(f"Invalid dtype_kind value read from shared memory: {dtype_kind}")
        read_dtype_kind = chr(dtype_kind)
        frame_dtype = np.dtype(f'{read_dtype_kind}{dtype_bits // 8}')

        return capacity, (frame_h, frame_w, frame_c), frame_dtype, unread_full, read_ptr, write_ptr, count

    def _get_pointers_metadata(self) -> Tuple[int, int, int, int]:
        """
        Reads pointer-related metadata (read_ptr, write_ptr, occupied_count)
        from the metadata shared memory. Assumes pointer_lock is held and 
        metadata_ctypes is not None.
        """

        unread_full =        self._metadata_ctypes.unread_full
        read_ptr =           self._metadata_ctypes.read_ptr
        write_ptr =          self._metadata_ctypes.write_ptr
        occupied_count =     self._metadata_ctypes.occupied_count

        return unread_full, read_ptr, write_ptr, occupied_count

    def _set_pointers_metadata(self, unread_full: int, read_ptr: int, write_ptr: int, occupied_count: int):
        """
        Writes pointer-related metadata (read_ptr, write_ptr, occupied_count)
        to the metadata shared memory. Assumes pointer_lock is held and 
        metadata_ctypes is not None.
        """

        # Access metadata ctypes directly
        self._metadata_ctypes.unread_full    = unread_full
        self._metadata_ctypes.read_ptr       = read_ptr
        self._metadata_ctypes.write_ptr      = write_ptr
        self._metadata_ctypes.occupied_count = occupied_count


    def put(self, frames: np.ndarray, timeout: Optional[float] = None) -> bool:
        """
        Puts frames into the shared ring buffer. Blocks if the buffer is full.
        Multiple `put` must only be called sequentially, or it will corrupt the buffer.

        Args:
            frames: (np.ndarray), (frame_num, frame_h, frame_w, frame_c), the frames to put.
            timeout: (Optional[float]), seconds to wait for space.

        Returns:
            True if successful, False if timeout occurred.

        Raises:
            RuntimeError: If the shared buffer has been uninitialized.
            ValueError: If the frame count exceeds buffer capacity or frame size/dtype mismatch.
        """
        # Validate frame counts to put
        if len(frames) > self._buffer_capacity:
            raise ValueError(f"Frame count {len(frames)} exceeds buffer capacity {self._buffer_capacity}.")

        # Validate frame size and dtype using cached attributes
        if frames.shape[-3:] != self._frame_shape:
            raise ValueError(f"Frame size mismatch. Expected {self._frame_shape}, got {frames.shape[-3:]}")
            
        if frames.dtype != self._dtype:
            raise ValueError(f"Frame dtype mismatch. Expected {self._dtype}, got {frames.dtype}")
        
        # Check remaining space outside lock (cover most calls) and trigger GC if needed.
        put_frame_num = frames.shape[0]
        if self._metadata_ctypes.occupied_count + put_frame_num > self._buffer_capacity:
            if self._gc_func is not None:
                self._gc_func()
        # Reading metadata requires the lock for consistency
        with self._pointer_lock: # Acquire lock for metadata access and synchronization
            # Always use occupied_count to determine if the buffer state, not read_ptr and write_ptr
            if not self._space_available.wait_for(
                lambda: self._metadata_ctypes.occupied_count + put_frame_num <= self._buffer_capacity, 
                timeout=timeout
            ):
                return False # Timeout
            # Re-read mutable metadata after waiting
            unread_full, read_ptr, write_ptr, occupied_count = self._get_pointers_metadata()

        # RELEASE Lock when data is copying
        data_buffer = self._data_ndarr
        # print(data_buffer) # Debugging print

        # Calculate the slots, offsets, handling the case where write_ptr wraps around
        if write_ptr + put_frame_num > self._buffer_capacity: # wraps around
            # Two segment to copy
            first_seg_slots = self._buffer_capacity - write_ptr
            second_seg_slots = put_frame_num - first_seg_slots
            # Calculate the offset in shared memory for the frame data
            first_seg_offset = slice(write_ptr * self._frame_bytes, 
                                     self._buffer_capacity * self._frame_bytes)
            second_seg_offset = slice(0, (write_ptr + put_frame_num) % 
                                         self._buffer_capacity * self._frame_bytes)
            # Execute copy
            first_dest = np.ndarray((first_seg_slots,) + self._frame_shape, # Calculate frame count of each segment
                                    dtype=self._dtype, 
                                    buffer=data_buffer[first_seg_offset])
            second_dest = np.ndarray((second_seg_slots,) + self._frame_shape, 
                                     dtype=self._dtype, 
                                     buffer=data_buffer[second_seg_offset])

            np.copyto(first_dest, frames[:self._buffer_capacity - write_ptr]) # Copy the first segment
            np.copyto(second_dest, frames[self._buffer_capacity - write_ptr:]) # Copy the second segment
        else: # No wrap around
            # Calculate the offset in data shared memory for the frame data
            frame_offset = slice(write_ptr * self._frame_bytes, 
                                 (write_ptr + put_frame_num) * self._frame_bytes)
            # print(f'At put tail: frame_offset: {frame_offset}') # Debugging print
            # print(f'At put tail: data_buffer_size: {self._data_buffer_size}') # Debugging print
            dest = np.ndarray((put_frame_num,) + self._frame_shape, 
                              dtype=self._dtype, 
                              buffer=data_buffer[frame_offset])
            np.copyto(dest, frames) # Data copy without the pointer lock

        with self._pointer_lock: # Acquire lock after data copy
            # Read mutable metadata again, in case `read_ptr` has changed
            unread_full, read_ptr, write_ptr, occupied_count = self._get_pointers_metadata()
            # Update metadata
            new_write_ptr = (write_ptr + put_frame_num) % self._buffer_capacity
            new_count = occupied_count + put_frame_num
            new_unread_full = 1 if new_write_ptr == read_ptr else 0 # chase the read_ptr.
            self._set_pointers_metadata(new_unread_full, read_ptr, new_write_ptr, new_count) # Use read_ptr obtained under the same lock

            # print(f'At put tail: read_ptr: {read_ptr}, write_ptr: {new_write_ptr}, \
            #       occupied_count: {new_count}, next_release_count: {next_release_count}')

            # Notify that data is available (under lock)
            self._data_available.notify_all()
        # Lock released HERE

        return True # Successfully put frame

    def get(self, get_frame_num: int, 
            timeout: Optional[float] = None
            ) -> Optional[Tuple[List[np.ndarray], BufferTicket]]:
        """
        Get unread data for given number of frames. Since V4, the gotten data 
        will no longer be release automatically. Call `release()` to manually 
        release the frames after use.

        Args:
            get_frame_num: (int), the number of frames to get.
            timeout: (Optional[float]), seconds to wait for data.

        Returns:
            The frames as a tuple: (frames_list, got_frame_num), or None if timeout occured.
            The shape of a single array is (frame_num, frame_h, frame_w, frame_c).
        
        Raises:
            RuntimeError: If the shared buffer has been uninitialized.
        """
        with self._pointer_lock: # Acquire lock for metadata access and synchronization
            # Wait for data to become available
            # V4 update: since no auto release, the waiting condition is also changed.
            if not self._data_available.wait_for(
                lambda: self._unread_count() >= get_frame_num, timeout=timeout
            ):
                return None # wait_for() returns False if timeout
            # Read mutable metadata after waiting
            unread_full, read_ptr, write_ptr, occupied_count = self._get_pointers_metadata()

        # A view of data buffer hard-coded to byte(uint8)
        data_buffer = self._data_ndarr
        frames_list = []

        if read_ptr + get_frame_num > self._buffer_capacity: # wraps around
            # Two segment to read
            first_seg_slots = self._buffer_capacity - read_ptr
            second_seg_slots = get_frame_num - first_seg_slots
            first_seg_offset = slice(read_ptr * self._frame_bytes, 
                                     self._buffer_capacity * self._frame_bytes)
            second_seg_offset = slice(0, second_seg_slots * self._frame_bytes)

            # Create views of the data buffer for the frame data
            frames_list.append(np.ndarray((first_seg_slots,) + self._frame_shape, # Calculate the view shape
                                          dtype=self._dtype, 
                                          buffer=data_buffer[first_seg_offset]))
            frames_list.append(np.ndarray((second_seg_slots,) + self._frame_shape, 
                                          dtype=self._dtype, 
                                          buffer=data_buffer[second_seg_offset]))
        else: # No wrap around
            # Calculate the read_ptr and offset similar to `put` method
            frame_offset = slice(read_ptr * self._frame_bytes, 
                                 (read_ptr + get_frame_num) * self._frame_bytes)

            # Read frame data from data shared memory (STILL UNDER LOCK)
            frames = np.ndarray((get_frame_num,) + self._frame_shape, 
                               dtype=self._dtype, 
                               buffer=data_buffer[frame_offset])
            frames_list.append(frames) # Add the view to the list

        with self._pointer_lock: # Acquire lock after data read
            # Move read_ptr
            unread_full, read_ptr, write_ptr, occupied_count = self._get_pointers_metadata()
            new_read_ptr = (read_ptr + get_frame_num) % self._buffer_capacity
            new_unread_full = 0 if get_frame_num > 0 else unread_full
            self._set_pointers_metadata(new_unread_full, new_read_ptr, write_ptr, occupied_count)
        ticket = BufferTicket(read_ptr, get_frame_num)

        return frames_list, ticket # Successfully got frame

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
        ...

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
        ...

    def release(self, obj: Union[int, BufferTicket], /) -> int:
        if isinstance(obj, BufferTicket):
            release_num = obj.read_num
        else: # Assume int or np.integer
            release_num = obj

        if release_num <= 0:
            return 0
            
        with self._pointer_lock: # Acquire lock for metadata access and synchronization
            unread_full, read_ptr, write_ptr, occupied_count = self._get_pointers_metadata()
            actual_release = min(release_num, occupied_count)
            
            # Reduce occupied count
            new_occupied_count = occupied_count - actual_release 
            # Move read_ptr forward if some unread data released.
            new_read_ptr = (read_ptr + max(0, self._unread_count() - new_occupied_count)) % self._buffer_capacity
            # acutal_release == 0 -> occupied_count == 0, buffer is null, otherwise, any release will make some space.
            self._set_pointers_metadata(0, new_read_ptr, write_ptr, new_occupied_count)
            
            if actual_release > 0:
                # Notify all for exported condition (by public property)
                self._space_available.notify_all() 
                
        return actual_release

    @property
    def trigger_release(self):
        return self._gc_func
    @trigger_release.setter
    def trigger_release(self, gc_func: Callable):
        self._gc_func = gc_func

    @overload
    def read_from(self, index: int, length: int, /) -> List[np.ndarray]:
        """
        Get a view of data from given `index` and `length` without lock. 
        The data availability is upon whether it has been released by the caller.
        
        Args:
            index (int): The starting index of internal buffer (0 <= index < capacity)
            length (int): The number of frames to read (0 < length <= capacity)
            
        Returns:
            List[np.ndarray]: A list of frame views.
        """
        ...

    @overload
    def read_from(self, ticket: BufferTicket, /) -> List[np.ndarray]:
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
        ...

    def read_from(self, arg1: Union[int, BufferTicket], arg2: Optional[int] = None) -> List[np.ndarray]:
        if isinstance(arg1, BufferTicket):
            index, length = arg1.read_ptr, arg1.read_num
        elif arg2 is not None: # Assume arg1 is int or np.integer, check arg2.
            index, length = arg1, arg2
        else:
            raise TypeError("Invalid argument composition.")

        if index < 0 or index >= self._buffer_capacity:
            raise ValueError(f"Invalid physical pointer: {index}")
        if length <= 0 or length > self._buffer_capacity:
            raise ValueError(f"Invalid length: {length}")

        data_buffer = self._data_ndarr 
        frames_list = []

        if index + length > self._buffer_capacity: # wraps around
            first_seg_slots = self._buffer_capacity - index
            second_seg_slots = length - first_seg_slots
            first_seg_offset = slice(index * self._frame_bytes, 
                                     self._buffer_capacity * self._frame_bytes)
            second_seg_offset = slice(0, second_seg_slots * self._frame_bytes)

            frames_list.append(np.ndarray((first_seg_slots,) + self._frame_shape, 
                                          dtype=self._dtype, buffer=data_buffer[first_seg_offset]))
            frames_list.append(np.ndarray((second_seg_slots,) + self._frame_shape, 
                                          dtype=self._dtype, buffer=data_buffer[second_seg_offset]))
        else: # No wrap around
            frame_offset = slice(index * self._frame_bytes, (index + length) * self._frame_bytes)
            frames_list.append(np.ndarray((length,) + self._frame_shape, 
                                          dtype=self._dtype, buffer=data_buffer[frame_offset]))
        return frames_list
    
    @property
    def dtype(self) -> np.dtype:
        """Returns the dtype of the frame data in the buffer."""
        return self._dtype

    @property
    def metadata_name(self) -> Optional[str]:
        """Returns the name of the metadata shared memory segment. None if already unlinked."""
        return self._metadata_shm.name if self._metadata_shm else None

    @property
    def data_name(self) -> Optional[str]:
        """Returns the name of the data shared memory segment. None if already unlinked."""
        return self._data_shm.name if self._data_shm else None

    @property
    def buffer_capacity(self) -> int:
        """Returns the total capacity of the buffer."""
        return self._buffer_capacity
    
    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        """Returns the shape of a single frame in the buffer."""
        return self._frame_shape

    @property
    def unread_full(self) -> bool:
        """Get whether all the data in buffer is unread."""
        with self._pointer_lock:
            return bool(self._metadata_ctypes.unread_full)
    @property
    def unread_full_(self) -> ctypes.c_int16:
        """Get whether all the data in buffer is unread, dirty read without lock."""
        return self._metadata_ctypes.unread_full

    @property
    def read_ptr(self) -> int:
        """The read pointer of the buffer"""
        with self._pointer_lock:
            return self._metadata_ctypes.read_ptr
    @property
    def read_ptr_(self) -> ctypes.c_int64:
        """The read pointer of the buffer, dirty read without lock. Safe in 64bit system"""
        # Due to TOCTOU, for single int64, acquire lock here is meaningless.
        return self._metadata_ctypes.read_ptr

    @property
    def write_ptr(self) -> int:
        """Returns the write pointer of the buffer"""
        with self._pointer_lock:
            return self._metadata_ctypes.write_ptr
    @property
    def write_ptr_(self) -> ctypes.c_int64:
        """Returns the write pointer of the buffer, dirty read without lock. Safe in 64bit system"""
        return self._metadata_ctypes.write_ptr

    def occupied_count(self, timeout = None) -> int:
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
        if not self._pointer_lock.acquire(timeout=timeout):
            raise TimeoutError(f"Timeout after waiting {timeout}s in acquiring lock for occupied_count.")
        try:
            count = self.occupied_count_
        except Exception as e:
             raise RuntimeError(f"Unexpected error in reading metadata for occupied_count.") from e
        finally:
            self._pointer_lock.release()
        return count

    # Internal unlocked interface is property.
    @property
    def occupied_count_(self) -> int:
        """
        Read the occupied_count without lock, the value may not be the newest 
        due to memory barrier. Only safe in 64 bit system.
        """
        return self._metadata_ctypes.occupied_count
    
    @property
    def is_full(self) -> bool:
        """Get whether the buffer is full"""
        with self._pointer_lock:
            return self._metadata_ctypes.occupied_count == self._buffer_capacity
    @property
    def is_full_(self) -> bool:
        """Get whether the buffer is full, dirty read without lock."""
        return self._metadata_ctypes.occupied_count == self._buffer_capacity

    def unread_count(self, timeout = None) -> int:
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
        if not self._pointer_lock.acquire(timeout=timeout):
            raise TimeoutError(f"Timeout after waiting {timeout}s in acquiring lock for unread_count.")
        try:
            return self._unread_count()
        finally:
            self._pointer_lock.release()
    
    def _unread_count(self) -> int:
        """
        Returns the number of unread items in the buffer without lock, 
        must acquire lock before calling. Internal use only.
        """
        unread_full, read_ptr, write_ptr, occupied_count = self._get_pointers_metadata()
        # The number of frames available for the *next* get call
        count = (write_ptr - read_ptr) % self._buffer_capacity 
        if (not count) & unread_full: # Equals to `(count==0) and (occupied_count>0)`
            # Corner case: write_ptr chase read_ptr, all frames in buffer is unread.
            count = self._buffer_capacity
        return count

    # Expose synchronization objects for passing to worker process
    @property
    def pointer_lock(self) -> mp_sync.Lock:
        return self._pointer_lock

    @property
    def data_available_condition(self) -> mp_sync.Condition:
        return self._data_available

    @property
    def space_available_condition(self) -> mp_sync.Condition:
        return self._space_available



    def reattach(self):
        """
        Re-attach to the shared memory segments in this process.

        Can be called after the buffer close() but the buffer is not freed globally.

        Noted that on Windows, if all processes have closed the buffer, the 
        shared memory segments will be freed globally.

        Raise:
            RuntimeError: if the buffer was not fully initialized or has unlinked.
            FileNotFoundError: if the shared memory segments are not found.
        """
        # Added in v3a, designed to replace create=False:
        #   Synchronization objects are not touched here, since they are passed 
        #   from the creator process.
        logger = self._logger
        if self._metadata_shm is None or self._data_shm is None or self._metadata_shm.name is None or self._data_shm.name is None:
            raise RuntimeError("Shared ring buffer was not fully initialized or has already unlinked.")
        try:
            # Old segments' name is preserved even after close().
            self._attach_mem_init(self._metadata_shm.name, self._data_shm.name)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error in reattaching to Metadata SHM: {self._metadata_shm.name}, "
                                    f"Data SHM: {self._data_shm.name}") from e
        if logger: logger.success(f"Re-attached to Shared Ring Buffer. Metadata SHM: {self._metadata_shm.name}, "
                                    f"Data SHM: {self._data_shm.name}, size: {self._data_buffer_size} bytes")

    def close(self):
        """
        Detaches the current process from the shared memory segments.

        After calling `close()`, *this process* can no longer access the shared memory.
        The shared memory segments MAY persist in the system if other processes
        are still attached. This should be called by all processes using the buffer
        when they are finished with it.
        """
        logger = self._logger
        if hasattr(self, '_metadata_ctypes'):
            del self._metadata_ctypes # Release exported pointer
        if hasattr(self, '_data_ndarr'):
            # For NDArray, seems no explicit deletion can work, 
            # we still del it for robust lifecycle management
            del self._data_ndarr 
        
        metadata_shm_name = self._metadata_shm.name if self._metadata_shm else None
        data_shm_name = self._data_shm.name if self._data_shm else None
        if self._metadata_shm:
            self._metadata_shm.close()
            # self._metadata_shm = None # Keep reference for unlink
        if self._data_shm:
            self._data_shm.close()
            # self._data_shm = None # Keep reference for unlink
            # Synchronization objects are not closed/unlinked here,
            # they are managed by the creator process.
        if logger: logger.success(f"Shared Ring Buffer closed. Metadata SHM: {metadata_shm_name}, Data SHM: {data_shm_name}")

    def unlink(self):
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
        logger = self._logger
        if self._metadata_shm:
            try:
                self._metadata_shm.unlink()
                if logger: logger.success(f"Metadata shared memory segment '{self._metadata_shm.name}' unlinked.")
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Metadata shared memory segment '{self._metadata_shm.name}' already unlinked.") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error in unlinking metadata shared memory segment '{self._metadata_shm.name}'") from e
            finally:
                 self._metadata_shm = None # Ensure reference is cleared

        if self._data_shm:
            try:
                self._data_shm.unlink()
                if logger: logger.success(f"Data shared memory segment '{self._data_shm.name}' unlinked.")
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Data shared memory segment '{self._data_shm.name}' already unlinked.") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error in unlinking data shared memory segment '{self._data_shm.name}'") from e
            finally:
                 self._data_shm = None # Ensure reference is cleared

            # Note: Synchronization objects are not unlinked here,
            # they are managed by the creator process.

    def __del__(self):
        if hasattr(self, "_metadata_ctypes"):
            del self._metadata_ctypes
        if hasattr(self, "_data_ndarr"):
            del self._data_ndarr


# ------------- Example Usage -------------
BUFFER_CAPACITY = 5
FRAME_SIZE = (1, 1, 3) # Example frame size (height, width, channels)
FRAME_DTYPE = np.dtype('uint8')

def producer_process(source_buffer: ProcessSafeSharedRingBuffer):
    buffer = None
    try:
        buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=source_buffer)
        print(f"Producer process {mp.current_process().pid} started.")
        for i in range(10):
            frame_num = 1 # Put one frame at a time for simpler peek testing
            frame_val = i * 10
            frames = (np.ones((frame_num,) + FRAME_SIZE, dtype=FRAME_DTYPE) * frame_val).astype(FRAME_DTYPE)
            print(f"Producer: Putting frame with value {frame_val}...")
            if buffer.put(frames, timeout=2.0):
                print(f"Producer: Put frame with value {frame_val} successfully.")
            else:
                print(f"Producer: Timeout putting frame with value {frame_val}.")
                break # Stop if timeout occurs
            time.sleep(0.2) # Simulate time between frames
    except Exception as e:
        print(f"Producer error: {e}")
    finally:
        print(f"Producer process {mp.current_process().pid} finished.")
        if buffer:
            buffer.close()

def consumer_process(source_buffer: ProcessSafeSharedRingBuffer):
    buffer = None
    try:
        buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=source_buffer)
        print(f"Consumer process {mp.current_process().pid} started.")
        time.sleep(0.5) # Let producer put some frames first
        for i in range(5): # Get 5 times
            print(f"Consumer: Attempting get...")
            frames_list = buffer.get(1, timeout=2.0) # Get one frame
            if frames_list:
                frame = frames_list[0] # Should be only one view
                print(f"Consumer: Got frame, shape: {frame.shape}, value example: {frame[0, 0, 0, 0]}")
            else:
                print(f"Consumer: Timeout or error getting frame.")

            # Peek after get (before release)
            peeked_frame_after_get = buffer.peek_last_frame()
            if peeked_frame_after_get is not None:
                 print(f"Consumer (peek after get): Last written frame value: {peeked_frame_after_get[0, 0, 0]}")
            else:
                 print("Consumer (peek after get): Buffer empty or error.")

            # Release the gotten frame explicitly (optional, as next get releases implicitly)
            # released = buffer.release_last_got_data()
            # print(f"Consumer: Released {released} frame(s).")

            time.sleep(0.6) # Simulate processing time & allow producer to add more

            # Peek again after some time (and potential release)
            peeked_frame_later = buffer.peek_last_frame()
            if peeked_frame_later is not None:
                 print(f"Consumer (peek later): Last written frame value: {peeked_frame_later[0, 0, 0]}")
            else:
                 print("Consumer (peek later): Buffer empty or error.")


    except Exception as e:
        print(f"Consumer error: {e}")
    finally:
        print(f"Consumer process {mp.current_process().pid} finished.")
        if buffer:
            buffer.close()

def peeker_process(source_buffer: ProcessSafeSharedRingBuffer):
    """A separate process that only peeks."""
    buffer = None
    try:
        buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=source_buffer)
        print(f"Peeker process {mp.current_process().pid} started.")
        for _ in range(15): # Peek multiple times
            peeked_frame = buffer.peek_last_frame()
            unread = buffer.unread_count
            occupied = buffer.occupied_count_debug
            last_get = buffer.next_release_count_debug
            if peeked_frame is not None:
                print(f"Peeker: Last frame value: {peeked_frame[0, 0, 0]}. Unread: {unread}, Occupied: {occupied}, LastGet: {last_get}")
            else:
                print(f"Peeker: Buffer empty or error. Unread: {unread}, Occupied: {occupied}, LastGet: {last_get}")
            time.sleep(0.3)
    except Exception as e:
        print(f"Peeker error: {e}")
    finally:
        print(f"Peeker process {mp.current_process().pid} finished.")
        if buffer:
            buffer.close()


if __name__ == "__main__":
    # Use spawn context for better compatibility across platforms if needed
    # mp.set_start_method('spawn', force=True)
    import time
    shared_buffer = None
    try:
        print("Main process: Creating shared buffer...")
        shared_buffer = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=BUFFER_CAPACITY, frame_shape=FRAME_SIZE, dtype=FRAME_DTYPE)

        # Store original references for unlink if needed (adjust close/unlink logic)
        # setattr(shared_buffer, '_metadata_shm_original_ref', shared_buffer._metadata_shm)
        # setattr(shared_buffer, '_data_shm_original_ref', shared_buffer._data_shm)


        print(f"Main process: Starting processes...")
        p = mp.Process(target=producer_process, args=(shared_buffer,), name="Producer")
        c = mp.Process(target=consumer_process, args=(shared_buffer,), name="Consumer")
        pk = mp.Process(target=peeker_process, args=(shared_buffer,), name="Peeker")


        p.start()
        time.sleep(0.1) # Stagger start slightly
        c.start()
        time.sleep(0.1)
        pk.start()

        p.join()
        c.join()
        pk.join()

        print("Main process: Processes finished.")

    except Exception as e:
        print(f"Main process error: {e}")
    finally:
        if shared_buffer:
            print("Main process: Cleaning up shared memory...")
            # Ensure close is called before unlink in the creator
            try:
                 shared_buffer.close()
            except Exception as e:
                 print(f"Error closing buffer in main: {e}")
            # Unlink should be called by the creator
            try:
                 shared_buffer.unlink()
            except Exception as e:
                 print(f"Error unlinking buffer in main: {e}")

    print("Main process: Shutdown complete.")