# frameserver_v2.py
# Scratch by Google Gemini 3.1 Pro, reviewed & cleaned up by Haiyun Huang (260421)

# Require 64bit system to safely read int64 atomically.
# gc_lock is a small grain lock, if lock nesting is needed, gc_lock must be acquired 
#   AFTER holding cid_lock/reg_lock to prevent ABBA dead lock.

import time
import ctypes
import errno
import multiprocessing as mp
import multiprocessing.shared_memory as mp_shm
import numpy as np
from numpy.typing import NDArray
from typing import Optional, List
from loguru._logger import Logger # for type hint only
from ringbuffers.shared_ring_buffer_v4 import ProcessSafeSharedRingBuffer
from frameserver_v2_types import (FrameTicket, TicketExpireException, 
                                  FSMetadata, _METADATA_VER_HASH, 
                                  MAX_CONSUMERS, MAX_TICKETS, _INT64_MAX)


class FrameServer:
    """
    FrameServer V2 enables zero-copy frame distribution to multiple consumers 
    from a shared ring buffer. Process-safe design, allowing concurrently get() 
    calls.
    Limitation: can only be passed to subprocesses when creating them. 
    (Common limitation for mp objects, standard picklization won't work)
    """
    def __init__(self, ring_buffer: ProcessSafeSharedRingBuffer, 
                 create: bool = True, frameserver: 'FrameServer' = None, 
                 inject_logger: Optional[Logger] = None):
        """
        Args:
            ring_buffer (ProcessSafeSharedRingBuffer): The ring buffer instance to use. 
                Once injected, it should be exclusively read by this FrameServer instance.
                Or internal status will be corrupted.
            create (bool): If True, create a new FrameServer. 
            frameserver (FrameServer): The FrameServer instance to link.
            inject_logger (Optional[Logger]): The loguru logger to use for logging, 
                           requires `enqueue=True`. If None, logging is disabled.
        
        Raises:
            TypeError: If inject_logger is not a loguru.Logger instance.
            ValueError: If create=False and shm_name is None.
            FileNotFoundError: If create=False and shm_name is not found.
            OSError: If the shared memory segment for metadata cannot be created or attached to.
            RuntimeError: If the shared memory segment for metadata is not a valid metadata 
                memory block for this FrameServer, or unexpected error occurs.
        """
        if inject_logger is not None:
            if isinstance(inject_logger, Logger):
                self._logger = inject_logger.bind(friendly_name="FrameServer")
            else:
                raise TypeError("inject_logger must be a loguru.Logger instance.")
        else:
            self._logger = None
        logger = self._logger

        self.buffer = ring_buffer
        self._reg_lock = mp.Lock()
        self._cid_locks: List[mp.Lock] = [mp.Lock() for _ in range(MAX_CONSUMERS)]
        self._gc_lock = mp.Lock() # gc lock is for blocking GC during get_async_copy()

        self._metadata: Optional[FSMetadata] = None
        self._enable_mask: Optional[NDArray] = None
        self._next_frame_ids: Optional[NDArray] = None
        self._tickets_arr: Optional[NDArray] = None
        self._gc_view: Optional[NDArray] = None # _gc_view is a flattened view for _next_frame_ids + _tickets_arr
        
        meta_size = ctypes.sizeof(FSMetadata)
        if create:
            try:
                self._shm = mp_shm.SharedMemory(create=True, size=meta_size)
                self._init_metadata()

            except OSError as e:
                self.close()
                raise OSError(f"Error in creating shared memory segment for metadata with size {meta_size}"
                    ", system resource has run out." if e.errno == errno.ENOMEM else ".") from e
            except Exception as e:
                self.close()
                raise RuntimeError(f"Unexpected error when creating FrameServer.") from e

            logger.success(f"FrameServer created with ShM name: {self._shm.name}, size: {meta_size} bytes")
        else: # link to exist mp FrameServer instance by name.
            if frameserver is None:
                raise ValueError("`frameserver` must be provided when create=False.")
            try:
                # Link to syncronization objects
                self._reg_lock = frameserver.reg_lock
                self._cid_locks = frameserver.cid_locks
                self._gc_lock = frameserver.gc_lock

                self._shm = mp_shm.SharedMemory(name=frameserver.shm_name)
                self._link_np_view()
            except AttributeError as e:
                self.close()
                raise AttributeError(f"FrameServer with ShM name: {frameserver.shm_name} is not a valid FrameServer instance.") from e
            except FileNotFoundError as e:
                self.close()
                raise FileNotFoundError(f"FrameServer ShM name {frameserver.shm_name} is invalid or expired.") from e
            except OSError as e:
                self.close()
                raise OSError(f"Failed to attach to FrameServer with ShM name: {frameserver.shm_name}.") from e
            except Exception as e:
                self.close()
                raise RuntimeError(f"Unexpected error when attaching to FrameServer with ShM name: {frameserver.shm_name}.") from e
            
            if self._metadata.fs_protocol_ver != _METADATA_VER_HASH:
                self.close()
                raise RuntimeError(f"ShM name: {frameserver.shm_name} is not a valid metadata memory block for this FrameServer."
                    f"Expected: {_METADATA_VER_HASH}, received: {self._metadata.fs_protocol_ver}")
            logger.success(f"Attached to FrameServer with ShM name: {self._shm.name}")

    @property
    def shm_name(self) -> str:
        return self._shm.name
    @property
    def reg_lock(self) -> mp.Lock:
        return self._reg_lock
    @property
    def cid_locks(self) -> List[mp.Lock]:
        return self._cid_locks
    @property
    def gc_lock(self) -> mp.Lock:
        return self._gc_lock
    

    def __getstate__(self) -> dict:
        """Picklization protocol that is compatible in multiprocess environment."""
        state = self.__dict__.copy()
        # Clear memory reference by ctypes and numpy.
        for k in ['_metadata', '_enable_mask', '_next_frame_ids', '_tickets_arr']:
            state.pop(k, None)
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        # Recreate memory reference.
        self._link_np_view()

    def _init_metadata(self):
        self._link_np_view()
        self._metadata.fs_protocol_ver = _METADATA_VER_HASH
        self._metadata.fs_oldest_frame_id = 0
        self._metadata.rb_offset = self.buffer.read_ptr
        self._enable_mask.fill(0)
        self._next_frame_ids.fill(0)
        self._tickets_arr.fill(_INT64_MAX)

    def _link_np_view(self):
        """Use numpy view to have easy access to array fields."""
        self._metadata: FSMetadata = FSMetadata.from_buffer(self._shm.buf)
        buf = self._shm.buf
        
        self._enable_mask: NDArray[np.bool_] = np.ndarray((MAX_CONSUMERS,), dtype=np.bool_, buffer=buf, offset=FSMetadata.enable_mask.offset)
        self._next_frame_ids: NDArray[np.int64] = np.ndarray((MAX_CONSUMERS,), dtype=np.int64, buffer=buf, offset=FSMetadata.next_frame_ids.offset)
        self._tickets_arr: NDArray[np.int64] = np.ndarray((MAX_CONSUMERS, MAX_TICKETS), dtype=np.int64, buffer=buf, offset=FSMetadata.tickets.offset)
        self._gc_view: NDArray[np.int64] = np.ndarray((MAX_CONSUMERS + MAX_CONSUMERS * MAX_TICKETS,), dtype=np.int64, buffer=buf, offset=FSMetadata.next_frame_ids.offset)

    def register_consumer(self) -> int:
        """
        Register a new consumer and return the assigned ID for further access.

        Returns:
            cid (int): consumer id, for future FrameServer API call.

        Raises:
            RuntimeError: If the number of consumers exceeds the maximum supported number.
        """
        logger = self._logger
        with self._reg_lock:
            zeros = np.where(self._enable_mask == False)[0]
            if len(zeros) == 0:
                raise RuntimeError(f"Exceed the max consumer number supported ({MAX_CONSUMERS}).")
            # Use the first available slot.
            cid = zeros[0]
            
            # Initialize metadata for the consumer.
            with self._gc_lock:
                self._next_frame_ids[cid] = self._metadata.fs_oldest_frame_id + self.buffer.occupied_count_
            self._tickets_arr[cid, :] = _INT64_MAX # Use _INT64_MAX to mark unused tickets.
            self._enable_mask[cid] = True # set flag at the end does NOT provide any order guarantee as it is shared memory.
        
        logger.info(f"Consumer {cid} registered.")
        return cid

    def unregister_consumer(self, cid: int):
        """
        Unregister consumer of given cid, release related resources.
        Do nothing if a consumer has already unregistered.

        Args:
            cid (int): consumer id to be unregistered.

        Raises:
            ValueError: If the consumer id is invalid.
        """
        logger = self._logger
        if not (0 <= cid < MAX_CONSUMERS):
            raise ValueError(f"Invalid consumer id: {cid}")
        with self._reg_lock:
            with self._cid_locks[cid]:
                if self._enable_mask[cid] == True:
                    self._enable_mask[cid] = False
                    self._next_frame_ids[cid] = _INT64_MAX
                    self._tickets_arr[cid, :] = _INT64_MAX

                    logger.info(f"Consumer {cid} unregistered.")
                else:
                    logger.debug(f"Consumer {cid} is already unregistered.")

        self._gc()

    def get_sync(self, cid: int, size: int, timeout: Optional[float] = None) -> Optional[FrameTicket]:
        """
        Get the next ticket for a consumer for `size` frames. 
        
        Can also guarenteed to get sequential data (no overlap, no skip) under 
        concurrent `get_sync` call for the same cid.

        Args:
            cid (int): consumer id.
            size (int): number of frames to get in this ticket.
            timeout (Optional[float]): time to wait for data to be available. None = blocking.

        Returns:
            FrameTicket: ticket for the consumer. None if timeout waiting race condition.

        Raises:
            ValueError: If the consumer id is invalid.
            RuntimeError: If the consumer is not activated.
        
        """
        if not (0 <= cid < MAX_CONSUMERS):
            raise ValueError(f"Invalid consumer id: {cid}")

        if self._enable_mask[cid] == False: # Check don't need lock.
            raise RuntimeError(f"Consumer ID ({cid}) is not activated.")

        end_time = time.monotonic() + timeout if timeout is not None else 0.0
        while True:
            insufficient_tickets: bool = False
            with self._cid_locks[cid]:
                # Check available slot.
                available_ticket_slots = np.where(self._tickets_arr[cid] == _INT64_MAX)[0]
                if len(available_ticket_slots) > 0:

                    with self._gc_lock: # Use _gc_lock to prevent fs_oldest_frame_id mismatch with occupied_count_.
                        buffer_frontier_id = self._metadata.fs_oldest_frame_id + self.buffer.occupied_count_

                    # Check whether buffer has enough data.
                    next_frame_id: int = self._next_frame_ids[cid]
                    if buffer_frontier_id >= next_frame_id + size: # Possibility for false-negative.
                        # Have enough data, issue ticket.
                        available_ticket_slot = available_ticket_slots[0]
                        self._tickets_arr[cid, available_ticket_slot] = next_frame_id
                        self._next_frame_ids[cid] = next_frame_id + size
                    
                        return FrameTicket(head_id=next_frame_id, size=size)
                else:
                    insufficient_tickets = True

            if insufficient_tickets:
                time.sleep(0.001) # time slice rotation, 16ms punishment on Windows.
                continue # retry by polling rather than waiting condition.

            # insufficient buffer, wait buffer with timeout.
            if timeout is not None:
                timeout = end_time - time.monotonic()
                if timeout <= 0: return None # timeout, return.
            else: timeout = None

            with self.buffer.pointer_lock: 
                # Check within lock, memory barrier.
                # wait_for exec predicate in lock, avoid using locked .occupied_count() (dead lock)
                if not self.buffer.data_available_condition.wait(timeout=timeout):
                    # Wait without predicate, let next cycle to check false positive.
                    return None # timeout, return.
                    

    def get_from_ticket(self, ticket: FrameTicket) -> List[np.ndarray]:
        """
        Extract the data view from the Ticket. Zero-copy.
        
        Args:
            ticket (FrameTicket): The ticket to extract data from.
        
        Returns:
            data (List[np.ndarray]): The data view from the Ticket. 
                                     len(data) can be 1 or 2.
        
        Raises:
            TicketExpireException: If the ticket is invalid or expired.
        """
        if ticket.head_id < self._metadata.fs_oldest_frame_id:
            raise TicketExpireException(ticket, self._metadata.fs_oldest_frame_id)
            
        relative_ptr = (ticket.head_id + self._metadata.rb_offset) % self.buffer.buffer_capacity
        return self.buffer.read_from(relative_ptr, ticket.size)

    def release_sync(self, cid: int, ticket: FrameTicket):
        """
        Release a ticket for a named consumer. Currently will also trigger GC.
        It's safe if a ticket is already released.

        Args:
            cid (int): consumer id.
            ticket (FrameTicket): The ticket to release.
        
        Raises:
            ValueError: If the consumer id is invalid.
        """
        if not (0 <= cid < MAX_CONSUMERS):
            raise ValueError(f"Invalid consumer id: {cid}")

        with self._cid_locks[cid]:
            match_idx = np.where(self._tickets_arr[cid] == ticket.head_id)[0]
            if len(match_idx) > 0:
                self._tickets_arr[cid, match_idx[0]] = _INT64_MAX
                # Ticket available condition now is checked by polling
            
        self._gc() # avoid unnecessary lock nesting

    def _gc(self) -> int:
        """
        GC function scans the FrameServer metadata and release the expired data from the RingBuffer.
        `_gc()` will acquire `_gc_lock`, please avoid unnecessary lock nesting to prevent dead lock.

        Returns:
            release_num (int): number of the oldest frames released.
        """
        # _gc() maintains the coupling of buffer release state and write fs_oldest_frame_id.
        logger = self._logger
        if self._metadata is None or self._enable_mask is None or self._next_frame_ids is None or self._tickets_arr is None:
            # May run in daemon thread, raise is not suitable here.
            logger.warning("GC is called when metadata not fully initialized. Release nothing.")
            return 0

        # Find the lagging most frame id (occupied for tickets, need to be preserved for next_frame_ids).
        global_occupied_from_id = np.min(self._gc_view) 
        # GC can run concurrently with other get_sync/release_sync. 
        # in 64 bit system, int64 is atomic, the data inconsistency will only make 
        # global_occupied_from_id lag behind the actual value. (release less, safe)

        # === Critical zone (protect fs_oldest_frame_id, no concurrent _gc()) ===
        if not self._gc_lock.acquire(block=False):
            return 0 # Let next gc to do works.
        
        # Has consumers, and has something to release.
        try:
            if global_occupied_from_id != _INT64_MAX and global_occupied_from_id > self._metadata.fs_oldest_frame_id:
                to_release = global_occupied_from_id - self._metadata.fs_oldest_frame_id
                release_num = self.buffer.release(to_release)
                # Maintain the couple of FS metadata and buffer status.
                self._metadata.fs_oldest_frame_id += release_num 
                return release_num
            return 0
        finally: # use `finally` to release lock before leaving current scope.
            self._gc_lock.release()

    def get_async(self, size: int) -> Optional[FrameTicket]:
        """
        Get the latest `size` frames from the buffer.

        Returns:
            Optional[FrameTicket]: The ticket for the latest `size` frames, 
                or None if the buffer do not have enough frames to return.
        """
        # Due to memory barrier, not guaranteed to be the latest.
        if self.buffer.occupied_count_ < size:
            return None # Buffer do not have enough data.
        with self._gc_lock:
            buffer_frontier_id = self._metadata.fs_oldest_frame_id + self.buffer.occupied_count_
        
        head_id = buffer_frontier_id - size
        return FrameTicket(head_id=head_id, size=size)

    def get_async_copy(self, size: int) -> Optional[List[NDArray]]:
        """
        Get the deep copy of the latest `size` frames.

        Returns:
            Optional[List[NDArray]]: The deep copy of the latest `size` frames, 
                or None if the buffer do not have enough frames to return.
        """
        if self.buffer.occupied_count_ < size:
            return None # Buffer do not have enough data.

        with self._gc_lock: # TODO: Anyway to make copy outside?? probably not except mod ring buffer.
            # Block GC ensures consistent `fs_oldest_frame_id` and `occupied_count_` and data integrity.
            buffer_frontier_id = self._metadata.fs_oldest_frame_id + self.buffer.occupied_count_
            head_id = buffer_frontier_id - size

            relative_ptr = (head_id + self._metadata.rb_offset) % self.buffer.buffer_capacity
            data_views = self.buffer.read_from(relative_ptr, size)
            return [np.copy(data_view) for data_view in data_views]


    def close(self):
        """Close the metadata shared memory."""
        if hasattr(self, '_metadata'):
            del self._metadata
        if hasattr(self, '_enable_mask'):
            del self._enable_mask
        if hasattr(self, '_next_frame_ids'):
            del self._next_frame_ids
        if hasattr(self, '_tickets_arr'):
            del self._tickets_arr

        if self._shm:
            self._shm.close()
        # The ring buffer is injected, the lifecycle should be managed outside.

    def unlink(self):
        """Unlink the metadata shared memory, should be called by the creator process."""
        if self._shm:
            try:
                self._shm.unlink()
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Metadata shared memory segment '{self._shm.name}' already unlinked.") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error in unlinking metadata shared memory segment '{self._shm.name}'") from e
            finally:
                self._shm = None # Ensure reference is cleared
