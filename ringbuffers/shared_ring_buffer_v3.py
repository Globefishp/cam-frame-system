# Sketched by Google Gemini 2.5 Flash exp, 
# Manually corrected by Haiyun Huang 2025.

# Update log: 
#   - v1: Basic function
#   - v2: Add peek_last_frame (read while blocking put/get), 
#         optimize pointer related func to increase throughput
#   - v3: Refactor peek_last_frame, allowing parallel read/put/get, 
#         useful in peek-intensive applications. (slightly slower put/get 
#         performance than v2)
#         Potential future improvement: split _get/set_pointer_metadata to smaller functions,
#         return less items (split according to the use case)

# Requirements: python > 3.9

import multiprocessing as mp
import multiprocessing.shared_memory as mp_shm
import multiprocessing.synchronize as mp_sync
import numpy as np
import ctypes
import warnings
from typing import Tuple, Any, Optional, List, overload # Import overload
import time # Import time for timeouts


# Define the metadata structure size
# Using a simple structure for demonstration: capacity, frame_size (h, w, c), dtype_kind, dtype_bits, read_ptr, write_ptr, count
# Using int64 for safety, adjust if needed. Let's ensure enough space.
# 10 fields + 2 new fields * 8 bytes/field (for int64) = 96 bytes. Let's use 128 bytes to be safe.

METADATA_SIZE = 128 # Keep 128 for alignment and future additions
class Metadata(ctypes.Structure):
    '''
    Metadata structure for the shared ring buffer.
    Using ctypes to maximize performance in pointer operations.
    Need to be explicitly released (del obj) in `close()`.
    '''
    _fields_ = [
        ("capacity", ctypes.c_int64),
        ("frame_h", ctypes.c_int64),
        ("frame_w", ctypes.c_int64),
        ("frame_c", ctypes.c_int64),
        ("dtype_kind", ctypes.c_int64),
        ("dtype_bits", ctypes.c_int64),
        ("read_ptr", ctypes.c_int64),
        ("write_ptr", ctypes.c_int64),
        ("occupied_count", ctypes.c_int64),
        ("next_release_count", ctypes.c_int64),
        ("protected_count", ctypes.c_int64), # Added: Frames protected by peek
        ("eager_release_count", ctypes.c_int64), # Added: Frames to release after peek
    ]

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
                 source_buffer: None = None):
        """
        Initialize or attach to the shared ring buffer.

        Args:
            create: (bool), True, create the shared memory blocks and initialize metadata.
            buffer_capacity: (int), the maximum number of frames the buffer can hold.
            frame_shape: (Tuple[int, int, int]), expected frame size (height, width, channel).
            dtype: (np.dtype), data type of the frames. Defaults to np.uint8.
        """
        ... 

    @overload
    def __init__(self,
                 create: bool,
                 buffer_capacity: None = None,
                 frame_shape: None = None,
                 dtype: None = None,
                 source_buffer: 'ProcessSafeSharedRingBuffer' = None):
        """
        Initialize or attach to the shared ring buffer.

        Args:
            create: (bool), False, attach to existing shared memory segments using source_buffer.
            source_buffer: (ProcessSafeSharedRingBuffer), the buffer instance created
                           in the main process, used for attaching in worker processes.
        """
        ... 

    def __init__(self,
                 create: bool,
                 buffer_capacity: Optional[int] = None,
                 frame_shape: Optional[Tuple[int, int, int]] = None,
                 dtype: np.dtype = np.uint8, # Keep default for convenience, but logic handles None
                 source_buffer: Optional['ProcessSafeSharedRingBuffer'] = None):
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
        """
        # Initialize attributes to None or default values, will be set properly later
        self._buffer_capacity: int = 0
        self._frame_shape: Tuple[int, int, int] = (0, 0, 0)
        self._dtype: np.dtype = np.dtype(np.uint8) # Default, will be overwritten
        self._frame_bytes: int = 0
        self._data_buffer_size: int = 0

        self._metadata_shm: Optional[mp_shm.SharedMemory] = None
        self._metadata_ctypes: Optional[Metadata] = None
        self._data_shm: Optional[mp_shm.SharedMemory] = None
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
            try:
                self._metadata_shm = mp_shm.SharedMemory(create=True, size=METADATA_SIZE)
                # Link metadata buffer to a ctypes structure for easier access.
                self._metadata_ctypes = Metadata.from_buffer(self._metadata_shm.buf)
                # Use the calculated _data_buffer_size for data shm creation
                self._data_shm = mp_shm.SharedMemory(create=True, size=self._data_buffer_size)


                # Create synchronization objects
                self._pointer_lock = mp.Lock()
                self._data_available = mp.Condition(self._pointer_lock)
                self._space_available = mp.Condition(self._pointer_lock)

                # Initialize metadata in the metadata shared memory
                self._init_metadata()
                print(f"Shared Ring Buffer created. Metadata SHM: {self._metadata_shm.name}, "
                      f"Data SHM: {self._data_shm.name}")

            except Exception as e:
                print(f"Error creating Shared Ring Buffer: {e}")
                self.close() # Clean up if creation fails
                raise # Re-raise the exception

        else:
            # --- Attach Mode ---
            if source_buffer is None:
                raise ValueError("source_buffer must be provided when create=False")
            if source_buffer.metadata_name is None or source_buffer.data_name is None:
                raise ValueError("Source buffer shared memory names are not available.")

            try:
                # 1. Attach to Metadata Shared Memory
                self._metadata_shm = mp_shm.SharedMemory(name=source_buffer.metadata_name)
                # Link metadata buffer to a ctypes structure for easier access.
                self._metadata_ctypes = Metadata.from_buffer(self._metadata_shm.buf)

                # 2. Read Metadata and Set Instance Attributes
                # Ensure metadata dtype is defined before accessing buffer
                if not hasattr(self, '_metadata_ctypes'):
                    raise ValueError("Metadata structure is not defined.")
                # Should not reach here, but just in case.

                # Read all metadata using _get_metadata
                capacity, frame_shape, frame_dtype, read_ptr, write_ptr, \
                occupied_count, next_release_count, protected_count, eager_release_count = self._get_metadata()

                # Set instance attributes based on read metadata
                self._buffer_capacity = capacity
                self._frame_shape = frame_shape
                self._dtype = frame_dtype
                self._frame_bytes = int(np.prod(self._frame_shape) * self._dtype.itemsize)
                self._data_buffer_size = self._buffer_capacity * self._frame_bytes

                # 3. Attach to Data Shared Memory
                self._data_shm = mp_shm.SharedMemory(name=source_buffer.data_name)

                # 4. Verify Data Buffer Size
                if self._data_shm.size != self._data_buffer_size:
                    import warnings
                    warnings.warn(f"Attached data buffer size ({self._data_shm.size}) mismatches "
                                  f"calculated size based on metadata ({self._data_buffer_size}). \n"
                                  f"This might be due to system-specific shared memory allocation "
                                  f"behavior. Will use calculated size.")

                # 5. Get Synchronization Objects from Source Buffer
                self._pointer_lock = source_buffer.pointer_lock
                self._data_available = source_buffer.data_available_condition
                self._space_available = source_buffer.space_available_condition

                print(f"Attached to Shared Ring Buffer. Metadata SHM: {self._metadata_shm.name}, "
                      f"Data SHM: {self._data_shm.name}")

            except FileNotFoundError as e:
                print(f"Error: Shared memory segment not found: {e}")
                self.close()
                raise
            except Exception as e:
                print(f"Error attaching to Shared Ring Buffer: {e}")
                self.close()
                raise

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
        self._metadata_ctypes.read_ptr = 0
        self._metadata_ctypes.write_ptr = 0
        self._metadata_ctypes.occupied_count = 0
        self._metadata_ctypes.next_release_count = 0
        self._metadata_ctypes.protected_count = 0 # Initialize new field
        self._metadata_ctypes.eager_release_count = 0 # Initialize new field

    def _get_metadata(self):
        """
        Reads metadata from the metadata shared memory. Assumes pointer_lock is held.
        Note: This method involves reading and checking
             and should be AVOIDED in frequently called scenarios.
        """
        if self._metadata_ctypes is None:
            raise RuntimeError("Metadata shared memory not properly initialized.")

        # # Access metadata ctypes directly

        capacity =           self._metadata_ctypes.capacity
        frame_h =            self._metadata_ctypes.frame_h
        frame_w =            self._metadata_ctypes.frame_w
        frame_c =            self._metadata_ctypes.frame_c
        dtype_kind =         self._metadata_ctypes.dtype_kind  # 后续需要转换为字符（如 'u'）
        dtype_bits =         self._metadata_ctypes.dtype_bits
        read_ptr =           self._metadata_ctypes.read_ptr
        write_ptr =          self._metadata_ctypes.write_ptr
        count =              self._metadata_ctypes.occupied_count
        next_release_count = self._metadata_ctypes.next_release_count
        protected_count =    self._metadata_ctypes.protected_count # Read new field
        eager_release_count= self._metadata_ctypes.eager_release_count # Read new field

        # Validate and reconstruct dtype
        if not (0 <= dtype_kind <= 127):
            raise ValueError(f"Invalid dtype_kind value read from shared memory: {dtype_kind}")
        read_dtype_kind = chr(dtype_kind)
        frame_dtype = np.dtype(f'{read_dtype_kind}{dtype_bits // 8}')

        return capacity, (frame_h, frame_w, frame_c), frame_dtype, read_ptr, write_ptr, count, next_release_count, protected_count, eager_release_count

    def _get_pointers_metadata(self):
        """
        Reads pointer-related metadata (read_ptr, write_ptr, occupied_count, next_release_count)
        from the metadata shared memory. Assumes pointer_lock is held.
        This method is intended for frequent access to mutable metadata.
        """
        if self._metadata_ctypes is None:
            raise RuntimeError("Metadata shared memory not properly initialized.")

        read_ptr =           self._metadata_ctypes.read_ptr
        write_ptr =          self._metadata_ctypes.write_ptr
        occupied_count =     self._metadata_ctypes.occupied_count
        next_release_count = self._metadata_ctypes.next_release_count
        protected_count =    self._metadata_ctypes.protected_count # Read new field
        eager_release_count= self._metadata_ctypes.eager_release_count # Read new field

        return read_ptr, write_ptr, occupied_count, next_release_count, protected_count, eager_release_count

    def _set_pointers_metadata(self, read_ptr: int, write_ptr: int, occupied_count: int, next_release_count: int, protected_count: int, eager_release_count: int):
        """
        Writes pointer-related metadata (read_ptr, write_ptr, occupied_count, next_release_count)
        to the metadata shared memory. Assumes pointer_lock is held.
        This method is intended for updating mutable metadata.
        """
        if self._metadata_ctypes is None:
            raise RuntimeError("Metadata shared memory not properly initialized.")

        # Access metadata ctypes directly

        self._metadata_ctypes.read_ptr = read_ptr
        self._metadata_ctypes.write_ptr = write_ptr
        self._metadata_ctypes.occupied_count = occupied_count
        self._metadata_ctypes.next_release_count = next_release_count
        self._metadata_ctypes.protected_count = protected_count # Write new field
        self._metadata_ctypes.eager_release_count = eager_release_count # Write new field


    def put(self, frames: np.ndarray, timeout: Optional[float] = None):
        """
        Puts frames into the shared ring buffer. Blocks if the buffer is full.
        Multiple `put` must only be called sequentially, or it will corrupt the buffer.
        Args:
            frames: (np.ndarray), (frame_num, frame_h, frame_w, frame_c), the frames to put.
            timeout: (Optional[float]), seconds to wait for space.
        Returns:
            True if successful, False if timeout occurred.
        """
        if self._metadata_shm is None or self._data_shm is None or self._pointer_lock is None or self._space_available is None or self._data_available is None:
            print("Error: Shared buffer not fully initialized.")
            return False
        
        # Validate requests, too many checks may reduce performance
        # Validate frame counts to put
        if len(frames) > self._buffer_capacity:
            print(f"Error: Frame count {len(frames)} exceeds buffer capacity {self._buffer_capacity}.")
            return False

        # Validate frame size and dtype using cached attributes
        if frames.shape[-3:] != self._frame_shape:
            print(f"Error: Frame size mismatch. Expected {self._frame_shape}, got {frames.shape[-3:]}")
            return False
            
        if frames.dtype != self._dtype:
            print(f"Error: Frame dtype mismatch. Expected {self._dtype}, got {frames.dtype}")
            return False # Return False as frame is invalid
        
        # Validate frame size and dtype
        # Reading metadata requires the lock for consistency
        with self._pointer_lock: # Acquire lock for metadata access and synchronization
            # Read mutable metadata under lock
            read_ptr, write_ptr, occupied_count, next_release_count, _, _ = self._get_pointers_metadata()
            # print(f'At put head: read_ptr: {read_ptr}, write_ptr: {write_ptr}, \
            #       occupied_count: {occupied_count}, next_release_count: {next_release_count}') # Debugging print

            put_frame_num = frames.shape[0]
            # Wait for space to become available
            while occupied_count + put_frame_num > self._buffer_capacity: # The buffer is full
                # Always use occupied_count to determine if the buffer state, not read_ptr and write_ptr
                # see `get` for details
                if not self._space_available.wait(timeout=timeout):
                    return False # Timeout
                # Re-read mutable metadata after waiting
                read_ptr, write_ptr, occupied_count, next_release_count, _, _ = self._get_pointers_metadata()
        # RELEASE Lock when data is copying
        # Access the data buffer directly
        data_buffer = np.ndarray(self._data_buffer_size, dtype=self._dtype, buffer=self._data_shm.buf)
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
            (read_ptr, write_ptr, occupied_count, next_release_count, 
             protected_count, eager_release_count) = self._get_pointers_metadata()
            # Update metadata
            new_write_ptr = (write_ptr + put_frame_num) % self._buffer_capacity
            new_count = occupied_count + put_frame_num
            # If peeking, update protected_count correspondingly
            new_protected_count = (protected_count + put_frame_num) if protected_count > 0 else 0 
            self._set_pointers_metadata(read_ptr, new_write_ptr, new_count, next_release_count, 
                                        new_protected_count, eager_release_count) # Use read_ptr obtained under the same lock

            # print(f'At put tail: read_ptr: {read_ptr}, write_ptr: {new_write_ptr}, \
            #       occupied_count: {new_count}, next_release_count: {next_release_count}')

            # Notify that data is available (under lock)
            self._data_available.notify()
        # Lock released HERE

        return True # Successfully put frame

    def get(self, get_frame_num: int, timeout: Optional[float] = None) -> Optional[List[np.ndarray]]:
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
        # To achieve "locking" the data read last time, the occupied_count is introduced.
        # It includes frames in buffer and frames have been read but not released.
        # So, occupied_count always >= write_ptr - read_ptr. That is, 
        # [write_ptr - occupied_count : write_ptr] is occupied. (should consider wrap when implementing)

        # [write_ptr - occupied_count : read_ptr] is intentionally preserved until 
        # next `get` or `_release_last_got_data` call.
        if self._metadata_shm is None or self._data_shm is None or self._pointer_lock is None or self._space_available is None or self._data_available is None:
            print("Error: Shared buffer not fully initialized.")
            return None

        with self._pointer_lock: # Acquire lock for metadata access and synchronization
            # Release previously preserved data (if any)
            released_count = self._release_last_got_data()
            if released_count > 0:
                self._space_available.notify() # Some space is available, the lock will be released at `wait` (see after)
            # Read mutable metadata
            read_ptr, write_ptr, occupied_count, next_release_count, protected_count, eager_release_count = self._get_pointers_metadata()
            # Wait for data to become available
            # print(f'At get head: read_ptr: {read_ptr}, write_ptr: {write_ptr}, \
            #       occupied_count: {occupied_count}, next_release_count: {next_release_count}') # Debugging print
            while (occupied_count - next_release_count) < get_frame_num: # Buffer has no enough data to get
                if not self._data_available.wait(timeout=timeout):
                    return None # Timeout
                # Re-read mutable metadata after waiting
                read_ptr, write_ptr, occupied_count, next_release_count, protected_count, eager_release_count = self._get_pointers_metadata()

        # Create a view of data buffer where further slicing occurs.
        data_buffer = np.ndarray(self._data_buffer_size, dtype=self._dtype, buffer=self._data_shm.buf)
        # print(data_buffer) # Debugging print
        frames_list = []

        if read_ptr + get_frame_num > self._buffer_capacity: # wraps around
            # Two segment to read
            first_seg_slots = self._buffer_capacity - read_ptr
            second_seg_slots = get_frame_num - first_seg_slots
            first_seg_offset = slice(read_ptr * self._frame_bytes, 
                                     self._buffer_capacity * self._frame_bytes)
            second_seg_offset = slice(0, (read_ptr + get_frame_num) % 
                                          self._buffer_capacity * self._frame_bytes)

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
            # Read mutable metadata again, in case `write_ptr` has changed
            read_ptr, write_ptr, occupied_count, next_release_count, protected_count, eager_release_count = self._get_pointers_metadata()
            # Update metadata (read_ptr and next_release_count) (under lock)
            new_read_ptr = (read_ptr + get_frame_num) % self._buffer_capacity
            self._set_pointers_metadata(new_read_ptr, write_ptr, occupied_count, get_frame_num, protected_count, eager_release_count) # Use write_ptr obtained under the same lock
            # the occupied_count is unchanged, the returned frames are not released here.
            # the `next_release_count` is updated. For the design concept, see `_release_last_got_data`.

            # Since the frames are not released, do not notify that space is available here.
            # self._space_available.notify()
        # Lock released HERE

        return frames_list # Successfully got frame

    def peek_frames(self, offset: int, num_frames: int, timeout: Optional[float] = 0.1) -> Optional[np.ndarray]:
        """
        Gets a copy of `num_frames` frames without removing them, starting from 
        last `offset` frame (i.e. offset relative to newest frame). Frames are
        read from oldest to newest.

        Args:
            offset: (int), the offset from the write pointer (0 for the last frame, 1 for the second last, etc.).
            num_frames: (int), the number of frames to read starting from the offset.
                        Should >= 1, <= offset + 1
            timeout: (Optional[float]), seconds to wait for acquiring the lock.
                     If None, block indefinitely, default 0.1s.
        Returns:
            A numpy ndarrays containing copies of the requested frames, (f, h, w, c)
            or None if the buffer is currently empty or the offset/num_frames is invalid
            or acquiring lock timeout.
        """
        if self._metadata_shm is None or self._data_shm is None or self._pointer_lock is None or self._metadata_ctypes is None:
            print("Error: Shared buffer not fully initialized for peek_frames.")
            return None

        acquired_lock = False

        # Stage 1 (with lock): Acquire lock and update metadata for protection
        acquired_lock = self._pointer_lock.acquire(timeout=timeout)
        if not acquired_lock:
                print("Timeout acquiring lock for peek_frames (Stage 1).")
                return None

        try:
            # Read mutable metadata
            read_ptr, write_ptr, occupied_count, next_release_count, protected_count, eager_release_count = self._get_pointers_metadata()

            # Validate request: Check if the offset and num_frames are valid and if there are enough frames.
            # Also check if protected_count is 0, assuming non-concurrent peek operations.
            # use a looser validation: allow accessing unreleased frames
            available_frames = occupied_count # - next_release_count
            if offset < 0 or offset >= available_frames or num_frames <= 0 or num_frames > offset + 1:
                 print(f"Error: Invalid offset ({offset}) or num_frames ({num_frames}). Available frame count is {available_frames}.")
                 return None # Invalid request

            if protected_count > 0:
                 print(f"Error: Another peek operation is in progress. protected_count is {protected_count}.")
                 return None # Non-concurrent peek assumption violated


            # Calculate new protected_count and eager_release_count
            new_protected_count = offset + 1
            # eager_release_count is for frames that were supposed to be released by get
            # but are now protected by peek.
            new_eager_release_count = max(0, new_protected_count + next_release_count - occupied_count)

            # Update metadata: set protected_count and eager_release_count
            self._set_pointers_metadata(read_ptr, write_ptr, occupied_count, next_release_count, new_protected_count, new_eager_release_count)

            # Calculate the index of the first frame to peek
            start_index_to_peek = (write_ptr - offset - 1) % self._buffer_capacity
            # Calculate the index of the last frame to peek,
            end_index_to_peek = (start_index_to_peek + num_frames) % self._buffer_capacity

            # Handling wrap-around for frames to copy
            slices_to_copy = [] # Store slices to copy
            if end_index_to_peek <= start_index_to_peek: # wraps around
                # Two segments to copy
                first_seg_slots = self._buffer_capacity - start_index_to_peek
                second_seg_slots = num_frames - first_seg_slots

                first_seg_start_bytes = start_index_to_peek * self._frame_bytes
                first_seg_end_bytes = self._buffer_capacity * self._frame_bytes
                second_seg_start_bytes = 0
                second_seg_end_bytes = (end_index_to_peek + 1) * self._frame_bytes

                # (start_bytes, end_bytes)
                slices_to_copy.append((slice(first_seg_start_bytes, first_seg_end_bytes), first_seg_slots))
                slices_to_copy.append((slice(second_seg_start_bytes, second_seg_end_bytes), second_seg_slots))

            else: # No wrap around
                # One segment to copy
                slices_to_copy.append((slice(start_index_to_peek * self._frame_bytes, 
                                             (end_index_to_peek + 1) * self._frame_bytes), num_frames))

        except Exception as e:
             print(f"Error during peek_frames Stage 1: {e}")
             return None # Indicate failure
        finally:
            self._pointer_lock.release() # Release lock after Stage 1

        # Stage 2 (no lock): Copy data
        try:
            data_buffer_views = []
            for slice_to_copy, slots in slices_to_copy:
                data_buffer_views.append(np.ndarray((slots, ) + self._frame_shape, 
                                                    dtype=self._dtype, 
                                                    buffer=self._data_shm.buf[slice_to_copy]))

            # np.concat always return a copy.
            frame_copies = np.concat(data_buffer_views, axis=0, dtype=self._dtype)

        except Exception as e:
            # Handle potential errors during view creation or copying
            print(f"Error accessing or copying frame data in peek_frames Stage 2: {e}")
            return None # Return None on error
        
        finally:
            # Stage 3 (with lock): Acquire lock and update metadata for release
            try:
                with self._pointer_lock:
                    # Must acquire lock again to ensure correct protected_count (important) and
                    # eager_release_count update.

                    # Read mutable metadata
                    read_ptr, write_ptr, occupied_count, next_release_count, protected_count, eager_release_count = self._get_pointers_metadata()

                    # Release eager frames and reset protected_count and eager_release_count
                    eager_released_now = eager_release_count
                    new_occupied_count = min(occupied_count - eager_released_now, occupied_count)
                    self._set_pointers_metadata(read_ptr, write_ptr, new_occupied_count, next_release_count, 0, 0)

                    if eager_released_now > 0:
                        self._space_available.notify() # Notify that space is available

            except Exception as e:
                print(f"Error during peek_frames Stage 3: {e}")
                # Data was copied, but metadata couldn't be updated.
                # The protected_count will remain set, blocking future peeks until a get/release happens.
                # Consider adding a more robust cleanup mechanism or logging.
                return None

        return frame_copies # Successfully peeked and copied frames

    def _release_last_got_data(self) -> int:
        """
        Release previously preserved data (returned by last `get` call) from the buffer.
        Operating metadata block directly, the `pointer_lock` should be handled by caller.
        Returns:
            frames released. (int)
        """
        # A simple implementation is setting `occupied_count` to 
        # (write_ptr - read_ptr) % capacity.
        # This is correct except for one corner case: 
        # when read_ptr == write_ptr and occupied_count == capacity
        # Is the buffer empty? or full? Depends on whether a `get` is called 
        # after a `put`.
        # So unfortunately we need to introduce one more metadata: `next_release_count`
        # It's maintained here and in `get`. It now also considers `protected_count`.
        read_ptr, write_ptr, occupied_count, next_release_count, protected_count, eager_release_count = self._get_pointers_metadata()

        # Calculate frames that can actually be released, respecting protected frames.
        # protected_count represents the N newest frames that cannot be released.
        # We can only release frames from the oldest ones (next_release_count).
        # Logically, protected_count should not exceed occupied_count.
        actual_release = min(next_release_count, occupied_count - protected_count)

        # Calculate frames that were intended for release but couldn't be due to protection.
        pending_release = next_release_count - actual_release

        new_occupied_count = occupied_count - actual_release
        new_eager_release_count = eager_release_count + pending_release
        # Update metadata: adjust occupied_count, set next_release_count to 0, add pending to eager.
        # Note: protected_count is managed by peek_frames.
        self._set_pointers_metadata(read_ptr, write_ptr, new_occupied_count, 0, protected_count, new_eager_release_count)
        return actual_release # Return the number of frames actually released
    
    def release_last_got_data(self) -> int:
        """
        Release previously preserved data (returned by last `get` call) from the buffer.
        This is a public method of `_release_last_got_data`, which handles the lock.
        Returns:
            frames released. (int)
        """
        if self._metadata_shm is None or self._data_shm is None or self._pointer_lock is None:
            # TODO: TOO MANY things to check for a correct initialization states, 
            #       considering wrap a function to do this.
            #       But this is a rare corner case.
            raise RuntimeError("Error: Shared buffer not fully initialized.")
        with self._pointer_lock: # Acquire lock for metadata access and synchronization
            released_count = self._release_last_got_data()
            self._space_available.notify() # Notify that space is available

        return released_count # Successfully released occupied frames
    
    def peek_last_frame(self, timeout: Optional[float] = 0.1) -> Optional[np.ndarray]:
        """
        Gets a copy of the last frame written to the buffer without removing it.
        Deceperated, use `peek_frames` instead.

        Args:
            timeout: (Optional[float]), seconds to wait for acquiring the lock. 
                     If None, block indefinitely, default 0.1s.
        Returns:
            A numpy ndarray containing a copy of the last written frame,
            or None if the buffer is currently empty or not initialized 
            or acquiring lock timeout.
        """
        if self._metadata_shm is None or self._data_shm is None or self._pointer_lock is None or self._metadata_ctypes is None:
            print("Error: Shared buffer not fully initialized for peek_last_frame.")
            return None

        frame_copy = None
        acquired_lock = False
        try:
            # Use the provided timeout for acquiring the lock.
            acquired_lock = self._pointer_lock.acquire(timeout=timeout)
            if not acquired_lock:
                 print("Timeout acquiring lock for peek_last_frame.")
                 return None

            # Read mutable metadata
            read_ptr, write_ptr, occupied_count, next_release_count, _ ,_ = self._get_pointers_metadata()

            # Check if there's at least one frame physically in the buffer slots.
            # occupied_count > 0 indicates something is conceptually in the buffer.
            if occupied_count <= 0:
                 # Buffer is logically empty, no frame to peek.
                 return None

            # Calculate the index of the last written frame slot using Python's modulo behavior
            last_frame_index = (write_ptr - 1) % self._buffer_capacity

            # Calculate the byte offset for the last frame in the data buffer
            offset = last_frame_index * self._frame_bytes

            # --- Create View and Copy (Still Under Lock) ---
            try:
                # Ensure the offset and size are within the bounds of the data buffer
                if offset < 0 or offset + self._frame_bytes > self._data_shm.size:
                     print(f"Error: Calculated offset {offset} or size is invalid for buffer size {self._data_shm.size}")
                     return None # Prevent potential memory access violation

                # Use the data shared memory buffer directly
                data_buffer_view = np.ndarray(
                    self._frame_shape,
                    dtype=self._dtype,
                    buffer=self._data_shm.buf,
                    offset=offset
                )
                # Create a copy *while holding the lock*
                frame_copy = np.copy(data_buffer_view)

            except Exception as e:
                # Handle potential errors during view creation or copying
                print(f"Error accessing or copying frame data in peek: {e}")
                return None # Return None on error

        except Exception as e:
             print(f"Error during peek_last_frame operation: {e}")
             return None # Indicate failure
        finally:
            if acquired_lock:
                self._pointer_lock.release() # Release lock HERE

        return frame_copy


    @property
    def metadata_name(self) -> Optional[str]:
        """Returns the name of the metadata shared memory segment."""
        return self._metadata_shm.name if self._metadata_shm else None

    @property
    def data_name(self) -> Optional[str]:
        """Returns the name of the data shared memory segment."""
        return self._data_shm.name if self._data_shm else None

    @property
    def unread_count(self) -> int:
        """Returns the number of unread items in the buffer."""
        if self._metadata_shm is None or self._pointer_lock is None:
            raise RuntimeError("Shared buffer not fully initialized.")

        with self._pointer_lock:
            read_ptr, write_ptr, occupied_count, next_release_count, _, _ = self._get_pointers_metadata()
            # The number of frames available for the *next* get call
            count = occupied_count - next_release_count
            count = max(0, count) # Ensure non-negative
            return count
        
    @property
    def buffer_capacity(self) -> int:
        """Returns the total capacity of the buffer."""
        return self._buffer_capacity
    
    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        """Returns the shape of a single frame in the buffer."""
        return self._frame_shape

    @property
    def occupied_count_debug(self) -> int:
        """Debug property: Returns the raw occupied_count metadata."""
        if self._metadata_shm is None or self._pointer_lock is None or self._metadata_ctypes is None:
             print("Warning: occupied_count_debug called on uninitialized buffer.")
             return -1 # Indicate error state

        count = -1
        acquired_lock = False
        try:
            acquire_timeout = 0.1
            acquired_lock = self._pointer_lock.acquire(timeout=acquire_timeout)
            if not acquired_lock:
                 print("Timeout acquiring lock for occupied_count_debug.")
                 return -1

            # Directly read the value
            count = self._metadata_ctypes.occupied_count

        except Exception as e:
             print(f"Error reading metadata for occupied_count_debug: {e}")
             count = -1
        finally:
            if acquired_lock:
                self._pointer_lock.release()
        return count

    @property
    def next_release_count_debug(self) -> int:
        """Debug property: Returns the raw next_release_count metadata."""
        if self._metadata_shm is None or self._pointer_lock is None or self._metadata_ctypes is None:
             print("Warning: next_release_count_debug called on uninitialized buffer.")
             return -1

        count = -1
        acquired_lock = False
        try:
            acquire_timeout = 0.1
            acquired_lock = self._pointer_lock.acquire(timeout=acquire_timeout)
            if not acquired_lock:
                 print("Timeout acquiring lock for next_release_count_debug.")
                 return -1

            count = self._metadata_ctypes.next_release_count

        except Exception as e:
             print(f"Error reading metadata for next_release_count_debug: {e}")
             count = -1
        finally:
            if acquired_lock:
                self._pointer_lock.release()
        return count


    # Expose synchronization objects for passing to worker process
    @property
    def pointer_lock(self) -> Optional[mp_sync.Lock]:
        return self._pointer_lock

    @property
    def data_available_condition(self) -> Optional[mp_sync.Condition]:
        return self._data_available

    @property
    def space_available_condition(self) -> Optional[mp_sync.Condition]:
        return self._space_available


    def close(self):
        """
        Detaches the current process from the shared memory segments.

        After calling `close()`, *this process* can no longer access the shared memory.
        The shared memory segments may persist in the system if other processes
        are still attached. This should be called by all processes using the buffer
        when they are finished with it.
        """
        if hasattr(self, '_metadata_ctypes'):
            del self._metadata_ctypes # Set to None to release the reference
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
        print(f"Shared Ring Buffer closed. Metadata SHM: {metadata_shm_name}, Data SHM: {data_shm_name}")

    def unlink(self):
        """
        Marks the shared memory segments for destruction.

        The actual destruction of the segments will occur only after all processes
        that were attached to them have detached (i.e., after all processes have
        called `close()` or terminated). This method should ONLY be called by the
        process that originally created the shared memory. A `close()` should be 
        called before calling `unlink()` in the creator process.
        """
        if self._metadata_shm:
            try:
                self._metadata_shm.unlink()
                print(f"Metadata shared memory segment '{self._metadata_shm.name}' unlinked.")
            except FileNotFoundError:
                print(f"Metadata shared memory segment '{self._metadata_shm.name}' already unlinked.")
            except Exception as e:
                print(f"Error unlinking metadata shared memory segment '{self._metadata_shm.name}': {e}")
            finally:
                 self._metadata_shm = None # Ensure reference is cleared

        if self._data_shm:
            try:
                self._data_shm.unlink()
                print(f"Data shared memory segment '{self._data_shm.name}' unlinked.")
            except FileNotFoundError:
                print(f"Data shared memory segment '{self._data_shm.name}' already unlinked.")
            except Exception as e:
                print(f"Error unlinking data shared memory segment '{self._data_shm.name}': {e}")
            finally:
                 self._data_shm = None # Ensure reference is cleared

            # Note: Synchronization objects are not unlinked here,
            # they are managed by the creator process.


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