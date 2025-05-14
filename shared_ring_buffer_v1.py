# Sketched by Google Gemini 2.5 Flash exp, 
# Manually corrected by Haiyun Huang 2025. 

# Update log: 
#   - v1: Basic function

# Requirements: python > 3.9

import multiprocessing as mp
import multiprocessing.shared_memory as mp_shm
import multiprocessing.synchronize as mp_sync
import numpy as np
from typing import Tuple, Any, Optional, List, overload # Import overload
import time # Import time for timeouts

# Define the metadata structure size
# Using a simple structure for demonstration: capacity, frame_size (h, w, c), dtype_kind, dtype_bits, read_ptr, write_ptr, count
# Using int64 for safety, adjust if needed. Let's ensure enough space.
# 10 fields * 8 bytes/field (for int64) = 80 bytes. Let's use 128 bytes to be safe and allow for potential future additions.
METADATA_SIZE = 128

class ProcessSafeSharedRingBuffer:
    """
    A process-safe circular buffer implemented using two SharedMemory blocks:
    one for metadata and one for frame data.
    Allows multiple processes to safely put and get numpy arrays (frames).
    Uses a pointer lock to synchronize access to metadata.
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
        self._metadata_array: Optional[np.ndarray] = None # Must connect to metadata_shm later
        self._data_shm: Optional[mp_shm.SharedMemory] = None
        self._pointer_lock: Optional[mp_sync.Lock] = None
        self._data_available: Optional[mp_sync.Condition] = None
        self._space_available: Optional[mp_sync.Condition] = None


        # Define metadata dtype
        self._metadata_dtype = np.dtype([
            ('capacity', np.int64),
            ('frame_h', np.int64),
            ('frame_w', np.int64),
            ('frame_c', np.int64),
            ('dtype_kind', np.int64), # e.g., 'u' for unsigned int
            ('dtype_bits', np.int64), # e.g., 8 for uint8
            ('read_ptr', np.int64),
            ('write_ptr', np.int64),
            ('occupied_count', np.int64), # Key concept, see `get` for details
            ('last_get_count', np.int64) # A patch concept, see `_release_last_got_data`.
        ])
        if self._metadata_dtype.itemsize > METADATA_SIZE:
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
                # Initialize numpy object to access shared memory
                self._metadata_array = np.ndarray((1,), dtype=self._metadata_dtype, buffer=self._metadata_shm.buf[:METADATA_SIZE])
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
                # Connect numpy object to shared memory
                self._metadata_array = np.ndarray((1,), dtype=self._metadata_dtype, buffer=self._metadata_shm.buf[:METADATA_SIZE])

                # 2. Read Metadata and Set Instance Attributes
                # Ensure metadata dtype is defined before accessing buffer
                if not hasattr(self, '_metadata_dtype'):
                    raise ValueError("Metadata dtype is not defined.")
                    # Should not reach here, but just in case.
                

                # Read values from metadata
                read_capacity =       int(self._metadata_array[0]['capacity'])
                read_frame_h =        int(self._metadata_array[0]['frame_h'])
                read_frame_w =        int(self._metadata_array[0]['frame_w'])
                read_frame_c =        int(self._metadata_array[0]['frame_c'])
                read_dtype_kind_ord = int(self._metadata_array[0]['dtype_kind'])
                read_dtype_bits =     int(self._metadata_array[0]['dtype_bits'])

                # Validate and reconstruct dtype
                if not (0 <= read_dtype_kind_ord <= 127):
                    raise ValueError(f"Invalid dtype_kind value read from shared memory: {read_dtype_kind_ord}")
                read_dtype_kind = chr(read_dtype_kind_ord)
                read_dtype = np.dtype(f'{read_dtype_kind}{read_dtype_bits // 8}')

                # Set instance attributes based on read metadata
                self._buffer_capacity = read_capacity
                self._frame_shape = (read_frame_h, read_frame_w, read_frame_c)
                self._dtype = read_dtype
                self._frame_bytes = int(np.prod(self._frame_shape) * self._dtype.itemsize)
                self._data_buffer_size = self._buffer_capacity * self._frame_bytes

                # 3. Attach to Data Shared Memory
                self._data_shm = mp_shm.SharedMemory(name=source_buffer.data_name)

                # 4. Verify Data Buffer Size
                if self._data_shm.size != self._data_buffer_size:
                    import warnings
                    warnings.warn(f"Attached data buffer size ({self._data_shm.size}) mismatches "
                                  f"calculated size based on metadata ({self._data_buffer_size}). "
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
        """Initializes the metadata in the metadata shared memory."""
        if self._metadata_shm is None:
            raise RuntimeError("Metadata shared memory not initialized.")

        # This method is only called during creation (under lock in put/get is not needed here)
        # Not sure if self._metadata_array is initialized, 
        if self._metadata_array is None:
            # Initialize metadata attribute in case it's not set.
            self._metadata_array = np.ndarray((1,), dtype=self._metadata_dtype, buffer=self._metadata_shm.buf[:METADATA_SIZE])

        self._metadata_array[0]['capacity'] = self._buffer_capacity
        self._metadata_array[0]['frame_h'] = self._frame_shape[0]
        self._metadata_array[0]['frame_w'] = self._frame_shape[1]
        self._metadata_array[0]['frame_c'] = self._frame_shape[2]
        self._metadata_array[0]['dtype_kind'] = ord(self._dtype.kind) # Store character code (should be V)
        self._metadata_array[0]['dtype_bits'] = self._dtype.itemsize * 8 # Store bits
        self._metadata_array[0]['read_ptr'] = 0
        self._metadata_array[0]['write_ptr'] = 0
        self._metadata_array[0]['occupied_count'] = 0
        self._metadata_array[0]['last_get_count'] = 0

    def _get_metadata(self):
        """Reads metadata from the metadata shared memory. Assumes pointer_lock is held."""
        if self._metadata_shm is None or self._metadata_array is None:
            raise RuntimeError("Metadata shared memory not properly initialized.")

        # Access metadata array directly

        capacity =       self._metadata_array[0]['capacity']
        frame_h =        self._metadata_array[0]['frame_h']
        frame_w =        self._metadata_array[0]['frame_w']
        frame_c =        self._metadata_array[0]['frame_c']
        dtype_kind = chr(self._metadata_array[0]['dtype_kind'])
        dtype_bits =     self._metadata_array[0]['dtype_bits']
        read_ptr =       self._metadata_array[0]['read_ptr']
        write_ptr =      self._metadata_array[0]['write_ptr']
        count =          self._metadata_array[0]['occupied_count']
        last_get_count = self._metadata_array[0]['last_get_count']

        # Reconstruct dtype
        frame_dtype = np.dtype(f'{dtype_kind}{dtype_bits // 8}')

        return capacity, (frame_h, frame_w, frame_c), frame_dtype, read_ptr, write_ptr, count, last_get_count

    def _set_metadata(self, read_ptr: int, write_ptr: int, count: int, last_get_count: int):
        """Writes metadata to the metadata shared memory. Assumes pointer_lock is held."""
        if self._metadata_shm is None:
            raise RuntimeError("Metadata shared memory not initialized.")

        # Access metadata array directly

        self._metadata_array[0]['read_ptr'] = read_ptr
        self._metadata_array[0]['write_ptr'] = write_ptr
        self._metadata_array[0]['occupied_count'] = count
        self._metadata_array[0]['last_get_count'] = last_get_count


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

        # Validate frame size and dtype
        # Reading metadata requires the lock for consistency
        with self._pointer_lock: # Acquire lock for metadata access and synchronization
            # Read metadata under lock
            capacity, expected_frame_shape, expected_dtype, read_ptr, write_ptr, occupied_count, last_get_count = self._get_metadata()
            # print(f'At put head: read_ptr: {read_ptr}, write_ptr: {write_ptr}, \
            #       occupied_count: {occupied_count}, last_get_count: {last_get_count}') # Debugging print

            # Validate frame size and dtype
            if frames.shape[-3:] != expected_frame_shape:
                print(f"Error: Frame size mismatch. Expected {expected_frame_shape}, got {frames.shape[-3:]}")
                return False
                
            if frames.dtype != expected_dtype:
                print(f"Error: Frame dtype mismatch. Expected {expected_dtype}, got {frames.dtype}")
                return False # Return False as frame is invalid
            
            put_frame_num = frames.shape[0]
            # Wait for space to become available
            while occupied_count + put_frame_num > capacity: # The buffer is full
                # Always use occupied_count to determine if the buffer state, not read_ptr and write_ptr
                # see `get` for details
                if not self._space_available.wait(timeout=timeout):
                    return False # Timeout
                # Re-read metadata after waiting
                capacity, expected_frame_shape, expected_dtype, read_ptr, write_ptr, occupied_count, last_get_count = self._get_metadata()
        # RELEASE Lock when data is copying
        # Access the data buffer directly
        data_buffer = np.ndarray(self._data_buffer_size, dtype=np.uint8, buffer=self._data_shm.buf)
        # print(data_buffer) # Debugging print

        # Calculate the slots, offsets, handling the case where write_ptr wraps around
        if write_ptr + put_frame_num > capacity: # wraps around
            # Two segment to copy
            first_seg_slots = capacity - write_ptr
            second_seg_slots = put_frame_num - first_seg_slots
            # Calculate the offset in shared memory for the frame data
            first_seg_offset = slice(write_ptr * self._frame_bytes, capacity * self._frame_bytes)
            second_seg_offset = slice(0, (write_ptr + put_frame_num) % capacity * self._frame_bytes)
            # Execute copy
            first_dest = np.ndarray((first_seg_slots,) + expected_frame_shape, # Calculate frame count of each segment
                                    dtype=self._dtype, 
                                    buffer=data_buffer[first_seg_offset])
            second_dest = np.ndarray((second_seg_slots,) + expected_frame_shape, 
                                     dtype=self._dtype, 
                                     buffer=data_buffer[second_seg_offset])

            np.copyto(first_dest, frames[:capacity - write_ptr]) # Copy the first segment
            np.copyto(second_dest, frames[capacity - write_ptr:]) # Copy the second segment
        else: # No wrap around
            # Calculate the offset in data shared memory for the frame data
            frame_offset = slice(write_ptr * self._frame_bytes, 
                                 (write_ptr + put_frame_num) * self._frame_bytes)
            # print(f'At put tail: frame_offset: {frame_offset}') # Debugging print
            # print(f'At put tail: data_buffer_size: {self._data_buffer_size}') # Debugging print
            dest = np.ndarray((put_frame_num,) + expected_frame_shape, 
                              dtype=self._dtype, 
                              buffer=data_buffer[frame_offset])
            np.copyto(dest, frames) # Data copy without the pointer lock

        with self._pointer_lock: # Acquire lock after data copy
            # Read metadata again, in case `read_ptr` has changed
            capacity, expected_frame_shape, expected_dtype, read_ptr, write_ptr, occupied_count, last_get_count = self._get_metadata()
            # Update metadata
            new_write_ptr = (write_ptr + put_frame_num) % capacity
            new_count = occupied_count + put_frame_num
            self._set_metadata(read_ptr, new_write_ptr, new_count, last_get_count) # Use read_ptr obtained under the same lock

            # print(f'At put tail: read_ptr: {read_ptr}, write_ptr: {new_write_ptr}, \
            #       occupied_count: {new_count}, last_get_count: {last_get_count}')

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
            # Read metadata
            capacity, frame_shape, frame_dtype, read_ptr, write_ptr, occupied_count, last_get_count = self._get_metadata()
            # Wait for data to become available
            # print(f'At get head: read_ptr: {read_ptr}, write_ptr: {write_ptr}, \
            #       occupied_count: {occupied_count}, last_get_count: {last_get_count}') # Debugging print
            while occupied_count - get_frame_num < 0: # Buffer has no enough data to get
                if not self._data_available.wait(timeout=timeout):
                    return None # Timeout
                # Re-read metadata after waiting
                capacity, frame_shape, frame_dtype, read_ptr, write_ptr, occupied_count, last_get_count = self._get_metadata()

        # Create a view of data buffer where further slicing occurs.
        data_buffer = np.ndarray(self._data_buffer_size, dtype=np.uint8, buffer=self._data_shm.buf)
        # print(data_buffer) # Debugging print
        frames_list = []

        if read_ptr + get_frame_num > capacity: # wraps around
            # Two segment to read
            first_seg_slots = capacity - read_ptr
            second_seg_slots = get_frame_num - first_seg_slots
            first_seg_offset = slice(read_ptr * self._frame_bytes, capacity * self._frame_bytes)
            second_seg_offset = slice(0, (read_ptr + get_frame_num) % capacity * self._frame_bytes)

            # Create views of the data buffer for the frame data
            frames_list.append(np.ndarray((first_seg_slots,) + frame_shape, # Calculate the view shape
                                          dtype=frame_dtype, 
                                          buffer=data_buffer[first_seg_offset]))
            frames_list.append(np.ndarray((second_seg_slots,) + frame_shape, 
                                          dtype=frame_dtype, 
                                          buffer=data_buffer[second_seg_offset]))
        else: # No wrap around
            # Calculate the read_ptr and offset similar to `put` method
            frame_offset = slice(read_ptr * self._frame_bytes, 
                                 (read_ptr + get_frame_num) * self._frame_bytes)

            # Read frame data from data shared memory (STILL UNDER LOCK)
            frames = np.ndarray((get_frame_num,) + frame_shape, 
                               dtype=frame_dtype, 
                               buffer=data_buffer[frame_offset])
            frames_list.append(frames) # Add the view to the list

        with self._pointer_lock: # Acquire lock after data read
            # Read metadata again, in case `write_ptr` has changed
            capacity, frame_shape, frame_dtype, read_ptr, write_ptr, occupied_count, last_get_count = self._get_metadata()
            # Update metadata (read_ptr and last_get_count) (under lock)
            new_read_ptr = (read_ptr + get_frame_num) % capacity
            self._set_metadata(new_read_ptr, write_ptr, occupied_count, get_frame_num) # Use write_ptr obtained under the same lock
            # the occupied_count is unchanged, the returned frames are not released here.
            # the `last_get_count` is updated. For the design concept, see `_release_last_got_data`.

            # Since the frames are not released, do not notify that space is available here.
            # self._space_available.notify()
        # Lock released HERE

        return frames_list # Successfully got frame
    
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
        # So unfortunately we need to introduce one more metadata: `last_get_count`
        # It's maintained here and in `get`.
        capacity, frame_size, frame_dtype, read_ptr, write_ptr, occupied_count, last_get_count = self._get_metadata()
        new_occupied_count = occupied_count - last_get_count 
        # New occupied_count should 'release' the frame count of last `get` call.
        if new_occupied_count < 0: # This should never happen, but just in case.
            new_occupied_count = 0
        self._set_metadata(read_ptr, write_ptr, new_occupied_count, 0) # Reset last_get_count to 0
        return last_get_count


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
            capacity, frame_shape, frame_dtype, read_ptr, write_ptr, occupied_count, last_get_count = self._get_metadata()
            return occupied_count - last_get_count

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
        """Closes the shared memory connections."""
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
        Unlinks the shared memory segments.
        This should only be called by the process that created the shared memory.
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


# ------------- Here after is the example usage of the shared ring buffer. -------------
BUFFER_CAPACITY = 5
FRAME_SIZE = (1, 1, 3) # Example frame size (height, width, channels)
FRAME_DTYPE = np.dtype('uint8')
# --- Producer Process (Simulated) ---
def producer_process(source_buffer: ProcessSafeSharedRingBuffer):
    # Attach to the existing shared buffer using the source_buffer object
    # The source_buffer object itself contains the names and sync objects
    buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=source_buffer)

    print("Producer process started.")
    for i in range(10): # Put 10 frames
        frame_num = 3
        frames = np.arange(i*frame_num, (i+1)*frame_num, dtype=FRAME_DTYPE).reshape(-1, 1, 1, 1) * np.ones((frame_num,) + FRAME_SIZE, dtype=FRAME_DTYPE) # Create dummy frames
        print(f"Producer: Putting frame loop {i}...")
        if buffer.put(frames, timeout=5.0):
            print(f"Producer: Put frame loop {i} successfully.")
            pass
        else:
            print(f"Producer: Timeout putting frame loop {i}.")
        time.sleep(0.1) # Simulate time between frames

    print("Producer process finished.")
    buffer.close() # Close the shared memory connections

# --- Consumer Process (Simulated) ---
def consumer_process(source_buffer: ProcessSafeSharedRingBuffer):
    # Attach to the existing shared buffer using the source_buffer object
    buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=source_buffer)

    print("Consumer process started.")
    for i in range(10): # Get 10 frames
        print(f"Consumer: Getting frame loop {i}...")
        frames_list = buffer.get(3, timeout=5.0)
        if frames_list is not None:
            for frames in frames_list:
                # assert frames[0, 0, 0, 0] == i
                print(f"Consumer: Got frame loop {i}, shape: {frames.shape}, value example: {frames[:, 0, 0, 0]}")
        else:
            print(f"Consumer: Timeout getting frame {i}.")
        time.sleep(0.1) # Simulate processing time

    print("Consumer process finished.")
    buffer.close() # Close the shared memory connections
    return None

# Example Usage (for testing the buffer independently)
if __name__ == "__main__":

    # --- Main Process ---
    print("Main process: Creating shared buffer and synchronization objects...")
    # Create the shared buffer (which creates SHM blocks, Lock, Conditions)
    shared_buffer = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=BUFFER_CAPACITY, frame_shape=FRAME_SIZE, dtype=FRAME_DTYPE)

    print(f"Main process: Starting producer and consumer processes...")
    # Create and start the producer and consumer processes
    # Pass the shared_buffer object itself to the processes
    p = mp.Process(target=producer_process, args=(shared_buffer,))
    c = mp.Process(target=consumer_process, args=(shared_buffer,))

    p.start()
    c.start()

    # Wait for processes to finish
    p.join()
    c.join()

    print("Main process: Processes finished. Cleaning up shared memory...")
    # Clean up the shared memory segments (only the creator process should unlink)
    shared_buffer.unlink()
    shared_buffer.close() # Close the connections in the main process as well

    print("Main process: Shutdown complete.")
