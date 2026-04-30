# frameserver_v3.py
# Scratch by Google Gemini 3.1 Pro, reviewed & cleaned up by Haiyun Huang (260425)

# Concurrent code must be modified with extreme care, make sure you have considered 
#   most concurrent cases and race conditions (HW out-of-order load/store etc.)

# Require 64bit system to safely read uint64 atomically.
# gc_lock is a fine-grained lock, if lock nesting is needed, gc_lock must be acquired 
#   AFTER holding cid_lock/reg_lock to prevent ABBA dead lock.
# Lock order (inner -> outer): gc_lock -> cid_lock -> reg_lock -> link_lock.

from typing import overload, Literal
import time
import ctypes
import errno
import hashlib
from contextlib import ExitStack
import multiprocessing as mp
import multiprocessing.shared_memory as mp_shm
import numpy as np
from numpy.typing import NDArray
from typing import Optional, List
from loguru._logger import Logger # for type hint only
from ringbuffers.shared_ring_buffer_v4 import ProcessSafeSharedRingBuffer

from .frameserver_v3_types import (FrameTicket, TicketExpireException, 
                                  FSMetadata, _METADATA_VER_HASH, 
                                  MAX_CONSUMERS, MAX_TICKETS, _UINT64_MAX,
                                  MAX_LINKED_BUFFERS)


class FrameServer:
    """
    FrameServer V3 enables zero-copy frame distribution to multiple consumers 
    from a shared ring buffer. Process-safe design, allowing concurrent `get()` 
    calls. V3 supports multiple ring buffers ("streams") synchronized in a unified 
    ticket space.

    Get ticket from any fs 
    -> get data views from streams in needed
    -> release from all (or any, in lazy GC mode, see below) streams.

    Note: Once released on ANY stream, the data views of **ALL** streams are not 
    guaranteed to be integral. Release only after all data views are not used 
    anymore. 

    Note: Currently event triggered GC is used. To avoid call `release_sync(tickets)`
    for all streams, enable lazy GC for attached buffers by explicitly set 
    `ring_buffer.trigger_release = frameserver._gc`, so that you can release 
    the ticket only in a single stream without blocking others (reduce gc call).
    
    Limitation: can only be passed to subprocesses when creating them. 
    (Common limitation for mp objects, standard picklization won't work)
    """
    @overload
    def __init__(self, 
                 create: Literal[True] = ..., *,
                 ring_buffer: ProcessSafeSharedRingBuffer, 
                 frameserver: None = None, 
                 inject_logger: Optional[Logger] = None):
        """
        Create a new FrameServer to manage flow control for the given ring buffer.

        Args:
            create (Literal[True]): Must be True for creation mode.
            ring_buffer (ProcessSafeSharedRingBuffer): The target ring buffer to manage.
                It should be exclusively read by this FrameServer instance, or 
                internal status will be corrupted. Existing read data in the 
                ring buffer will be released, unread data will be preserved.
            frameserver (None): Unused when `create=True`.
            inject_logger (Optional[Logger]): Loguru logger instance.

        Raises:
            TypeError: If `inject_logger` is not a loguru.Logger instance.
            ValueError: If `ring_buffer` is not provided correctly.
            OSError: If the shared memory segment for metadata cannot be created.
            RuntimeError: If any unexpected error occurs.
        """
        ...

    @overload
    def __init__(self, 
                 create: Literal[False], *,
                 ring_buffer: None = None, 
                 frameserver: 'FrameServer', 
                 inject_logger: Optional[Logger] = None):
        """
        Attach to an existing FrameServer forking its internal resources.

        Usually this attachment is not needed. Directly using the frameserver copy 
        inherited from parent process is recommended.

        Args:
            create (Literal[False]): Must be False for attach mode.
            ring_buffer (None): Leave None to inherit the buffer from the source `frameserver`.
            frameserver (FrameServer): The source FrameServer instance to attach to.
            inject_logger (Optional[Logger]): Loguru logger instance.

        Raises:
            TypeError: If `inject_logger` is not a loguru.Logger instance.
            ValueError: If `frameserver` is not provided correctly.
            FileNotFoundError: If the shared memory segment of the parent `frameserver` cannot be found.
            OSError: If the shared memory segment for metadata cannot be attached to.
            RuntimeError: If the shared memory segment for metadata is not a valid metadata 
                memory block for this FrameServer, or unexpected error occurs.
        """
        ...

    @overload
    def __init__(self, 
                 create: Literal[False], *,
                 ring_buffer: ProcessSafeSharedRingBuffer, 
                 frameserver: 'FrameServer', 
                 inject_logger: Optional[Logger] = None):
        """
        Attach a new buffer to an existing FrameServer's unified ticket space to
        create a new stream. This stream will start reading data from the `read_ptr`
        when attaching. 

        Args:
            create (Literal[False]): Must be False for attach mode.
            ring_buffer (ProcessSafeSharedRingBuffer): The new ring buffer to manage.
            frameserver (FrameServer): The source FrameServer instance to join.
            inject_logger (Optional[Logger]): Loguru logger instance.

        Raises:
            TypeError: If `inject_logger` is not a loguru.Logger instance.
            ValueError: If `frameserver` is not provided correctly.
            FileNotFoundError: If the shared memory segment of the parent `frameserver` cannot be found.
            OSError: If the shared memory segment for metadata cannot be attached to.
            RuntimeError: If exceed the max number of linked buffers supported, 
                or ring_buffer provided has already been linked to the frameserver, 
                or unexpected error occurs.
        """
        ...

    def __init__(self, create: bool = True, *,
                 ring_buffer: Optional[ProcessSafeSharedRingBuffer] = None, 
                 frameserver: Optional['FrameServer'] = None, 
                 inject_logger: Optional[Logger] = None):
        if inject_logger is not None:
            if isinstance(inject_logger, Logger):
                self._logger = inject_logger.bind(friendly_name="FrameServer")
            else:
                raise TypeError("inject_logger must be a loguru.Logger instance.")
        else:
            self._logger = None
        logger = self._logger

        # === Syncronization objects ===
        # Lock for register/unregister consumer (`c_enable_mask`)
        self._reg_lock: mp.Lock        =  mp.Lock()
        # Consumer lock for get/release (`tickets`)
        self._cid_locks: List[mp.Lock] = [mp.Lock() for _ in range(MAX_CONSUMERS)] 
        # Independent GC locks per buffer (`buffer.occupied_count_` & `tickets[:, :MAX_TICKETS]`)
        self._gc_locks: List[mp.Lock]  = [mp.Lock() for _ in range(MAX_LINKED_BUFFERS)]
        # lock for linking buffers and ref counting (`rb_*`)
        self._link_lock: mp.Lock       =  mp.Lock() 

        # === Buffer ===
        self.buffer: Optional[ProcessSafeSharedRingBuffer] = None # committed later.
        self.buf_id: Optional[int] = None # Only set to actual id when all validation passed.

        # === Metadata ===
        self._shm: Optional[mp_shm.SharedMemory] = None
        self._metadata:     Optional[FSMetadata] = None
        self._rb_metadata_name_hashes:  Optional[NDArray] = None
        self._rb_linked_fs_count:  Optional[NDArray] = None
        self._rb_oldest_frame_ids: Optional[NDArray] = None # pad to 64 bytes, use [buf_id, 0].
        self._rb_offsets:          Optional[NDArray] = None
        self._c_enable_mask:  Optional[NDArray] = None
        self._next_frame_ids: Optional[NDArray] = None
        self._tickets_arr:    Optional[NDArray] = None
        self._gc_view:        Optional[NDArray] = None
        
        meta_size = ctypes.sizeof(FSMetadata)

        # ====== Initialization ======
        # Sync obj -> metadata(views) -> buffer & metadata init
        if create:
            if not isinstance(ring_buffer, ProcessSafeSharedRingBuffer):
                raise ValueError("`ring_buffer` must be provided as a instance of "
                                 "`ProcessSafeSharedRingBuffer` when create=True.")
            try:

                self._shm = mp_shm.SharedMemory(create=True, size=meta_size)
                self._link_np_view()

                # Set buffer and initialize related properties
                self.buffer = ring_buffer; self.buf_id = 0
                self._init_buffer_gc()
                self._init_metadata()

            except OSError as e:
                self.close()
                raise OSError(f"Error in creating shared memory segment for metadata with size {meta_size}"
                    ", system resource has run out." if e.errno == errno.ENOMEM else ".") from e
            except Exception as e:
                self.close()
                raise RuntimeError(f"Unexpected error when creating FrameServer.") from e

            if logger: logger.success(f"FrameServer created with ShM name: {self._shm.name}, size: {meta_size} bytes")
        else: # link to exist mp FrameServer instance by name.
            if not isinstance(frameserver, FrameServer):
                raise ValueError("`frameserver` must be provided as a instance of "
                                 "`FrameServer` when create=False.")
            try:
                # Link to syncronization objects
                self._reg_lock = frameserver.reg_lock
                self._cid_locks = frameserver.cid_locks
                self._gc_locks = frameserver.gc_locks
                self._link_lock = frameserver.link_lock

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

            with ExitStack() as stack:
                if not self._link_lock.acquire(timeout=0.1):
                    raise TimeoutError("Acquire link_lock timeout.")
                stack.callback(self._link_lock.release)

                if not isinstance(ring_buffer, ProcessSafeSharedRingBuffer):
                    # Link to buffer inside the provided FrameServer
                    self.buffer = frameserver.buffer; self.buf_id = frameserver.buf_id
                    self._rb_linked_fs_count[self.buf_id] += 1
                else:
                    # Link to a external buffer but use the same flow control with master server.
                    buffer_hash = int.from_bytes(hashlib.blake2b(ring_buffer.metadata_name.encode(), digest_size=8).digest(), 'little', signed=False)

                    # Prevent duplicate buffer linking
                    if buffer_hash in self._rb_metadata_name_hashes[self._rb_linked_fs_count > 0]:
                        raise RuntimeError(f"Buffer {ring_buffer.metadata_name} is already linked to this FrameServer.")

                    available_buf_ids = np.where(self._rb_linked_fs_count == 0)[0]
                    if len(available_buf_ids) == 0:
                        raise RuntimeError(f"Exceed the max number of linked buffers supported ({MAX_LINKED_BUFFERS}).")
                    self.buffer = ring_buffer # validation pass, commit property.
                    self.buf_id = int(available_buf_ids[0]) # Assign the first available slots.
                    self._init_buffer_gc()

                    self._rb_linked_fs_count[self.buf_id] = 1
                    self._rb_metadata_name_hashes[self.buf_id] = buffer_hash
                    
                    # Super heavy lock sequence will block all actions:
                    if not self._reg_lock.acquire(timeout=0.1): # _c_enable_mask
                        raise TimeoutError("Acquire reg_lock timeout.")
                    stack.callback(self._reg_lock.release)

                    enabled_consumer_cids = np.where(self._c_enable_mask == True)[0]

                    for i in enabled_consumer_cids: 
                        if not self._cid_locks[i].acquire(timeout=0.02): # _next_frame_ids
                            raise TimeoutError(f"Acquire cid_locks[{i}] timeout.")
                        stack.callback(self._cid_locks[i].release)
                    
                    read_frontier_id: np.uint64
                    if len(enabled_consumer_cids) > 0:
                        # Link to current consumers' read frontier.
                        read_frontier_id = np.max(self._next_frame_ids[enabled_consumer_cids])
                    else:
                        # Inherit current frameserver's frontier.
                        # The same logic as the first consumer registration.
                        read_frontier_id = frameserver._buf_write_frontier_gc_locked()

                    if not self._gc_locks[self.buf_id].acquire(timeout=0.1): # _rb_oldest_frame_ids
                        raise TimeoutError("Acquire gc_locks timeout.")
                    try:
                        self._rb_oldest_frame_ids[self.buf_id] = read_frontier_id
                        # Align `read_frontier_id` to the new buffer `read_ptr` by the offset.
                        # The first `get` will get (read_frontier_id + rb_offset) % capacity === read_ptr.
                        self._rb_offsets[self.buf_id] = (self.buffer.read_ptr_ - int(read_frontier_id)) % self.buffer.buffer_capacity
                    finally:
                        self._gc_locks[self.buf_id].release()

            if logger: logger.success(f"Attached to FrameServer with ShM name: {self._shm.name}")

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
    def gc_locks(self) -> List[mp.Lock]:
        return self._gc_locks
    @property
    def link_lock(self) -> mp.Lock:
        return self._link_lock
    

    def __getstate__(self) -> dict:
        """Picklization protocol that is compatible in multiprocess environment."""
        state = self.__dict__.copy()
        # Clear memory reference by ctypes and numpy.
        for k in ['_metadata', 
                  '_rb_linked_fs_count', '_rb_metadata_name_hashes', 
                  '_rb_oldest_frame_ids', '_rb_offsets', 
                  '_c_enable_mask', '_next_frame_ids', '_tickets_arr', '_tickets_arr_full']:
            state.pop(k, None)
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        # Recreate memory reference.
        self._link_np_view()
        # Increase ref count for the inherited buffer slot safely
        with self._link_lock:
            self._rb_linked_fs_count[self.buf_id] += 1

    def _init_buffer_gc(self):
        # Initial buffer GC, preserve unread frames.
        with self.buffer.pointer_lock:
            initial_release_num = self.buffer.occupied_count_ - self.buffer._unread_count()
        self.buffer.release(initial_release_num)

    def _init_metadata(self):
        self._metadata.fs_protocol_ver = _METADATA_VER_HASH

        self._rb_metadata_name_hashes.fill(0)
        self._rb_linked_fs_count.fill(0)
        self._rb_oldest_frame_ids.fill(_UINT64_MAX) # _UINT64_MAX = uninit.
        self._rb_offsets.fill(_UINT64_MAX)

        self._rb_linked_fs_count[0] = 1 # reference count
        self._rb_metadata_name_hashes[0] = int.from_bytes(
            hashlib.blake2b(self.buffer.metadata_name.encode(), digest_size=8).digest(), 'little', signed=False)
        # A transformation from relative counting(rb) to absolute counting(fs).
        # Anchor: for the first buffer, let initial frame to be 0, mapped to `rb_offset`.
        self._rb_oldest_frame_ids[0] = 0 
        self._rb_offsets[0] = self.buffer.read_ptr_

        self._c_enable_mask.fill(0)
        self._gc_view.fill(_UINT64_MAX)

    def _link_np_view(self):
        """Use numpy view to have easy access to array fields."""
        self._metadata: FSMetadata = FSMetadata.from_buffer(self._shm.buf)
        buf = self._shm.buf
        
        self._rb_metadata_name_hashes: NDArray[np.uint64] = np.ndarray((MAX_LINKED_BUFFERS,), dtype=np.uint64, buffer=buf, offset=FSMetadata.rb_metadata_name_hashes.offset)
        self._rb_linked_fs_count: NDArray[np.uint8] = np.ndarray((MAX_LINKED_BUFFERS,), dtype=np.uint8, buffer=buf, offset=FSMetadata.rb_linked_fs_count.offset)
        self._rb_oldest_frame_ids: NDArray[np.uint64] = np.ndarray((MAX_LINKED_BUFFERS, 8), dtype=np.uint64, buffer=buf, offset=FSMetadata.rb_oldest_frame_ids.offset)[:, 0] # the first element is valid.
        self._rb_offsets: NDArray[np.uint64] = np.ndarray((MAX_LINKED_BUFFERS,), dtype=np.uint64, buffer=buf, offset=FSMetadata.rb_offsets.offset)
        
        self._c_enable_mask: NDArray[np.bool_] = np.ndarray((MAX_CONSUMERS,), dtype=np.bool_, buffer=buf, offset=FSMetadata.c_enable_mask.offset)
        self._gc_view:        NDArray[np.uint64] = np.ndarray((MAX_CONSUMERS, MAX_TICKETS + 1), dtype=np.uint64, buffer=buf, offset=FSMetadata.tickets.offset)
        self._tickets_arr:    NDArray[np.uint64] = self._gc_view[:, :MAX_TICKETS]
        self._next_frame_ids: NDArray[np.uint64] = self._gc_view[:, MAX_TICKETS]

    def _buf_write_frontier_gc_locked(self) -> np.uint64:
        """Get the absolute index of the current buffer's write frontier with the buffer's gc_lock."""
        with self._gc_locks[self.buf_id]: # Use gc_lock to prevent rb_oldest_frame_id mismatch with occupied_count_.
            return self._rb_oldest_frame_ids[self.buf_id] + self.buffer.occupied_count_

    def register_consumer(self, historical_data: bool = False) -> int:
        """
        Register a new consumer and return the assigned ID for further access.

        Args:
            historical_data (bool): If True, the consumer will start from the 
                oldest frame that hasn't been released by any consumer.
                If False, the consumer will start from the current write frontier,
                thus only data produced after registration can be read.
                Default is False.

        Returns:
            cid (int): consumer id, for future FrameServer API call.

        Raises:
            RuntimeError: If the number of consumers exceeds the maximum supported number.
        """
        logger = self._logger
        with self._reg_lock:
            available_cids = np.where(self._c_enable_mask == False)[0]
            if len(available_cids) == 0:
                raise RuntimeError(f"Exceed the max consumer number supported ({MAX_CONSUMERS}).")
            # Use the first available slot.
            cid = available_cids[0]
            
            # Initialize metadata for the consumer.
            if historical_data:
                self._next_frame_ids[cid] = self._rb_oldest_frame_ids[self.buf_id]
            else:
                self._next_frame_ids[cid] = self._buf_write_frontier_gc_locked()
            self._tickets_arr[cid, :] = _UINT64_MAX # Use _INT64_MAX to mark unused tickets.
            self._c_enable_mask[cid] = True # set flag at the end does NOT provide any order guarantee as it is shared memory.
        
        if logger: logger.info(f"Consumer {cid} registered.")
        return int(cid)

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
            # if cid is the last consumer, no need for gc, to align with previous logics
            need_gc = (self._c_enable_mask.sum() > 1)
            with self._cid_locks[cid]:
                if self._c_enable_mask[cid] == True:
                    self._c_enable_mask[cid] = False
                    self._next_frame_ids[cid] = _UINT64_MAX
                    self._tickets_arr[cid, :] = _UINT64_MAX

                    if logger: logger.info(f"Consumer {cid} unregistered.")
                else:
                    if logger: logger.debug(f"Consumer {cid} is already unregistered.")

        if need_gc: self._gc()

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

        if self._c_enable_mask[cid] == False: # Check don't need lock.
            raise RuntimeError(f"Consumer ID ({cid}) is not activated.")

        end_time = time.monotonic() + timeout if timeout is not None else 0.0
        while True:
            insufficient_tickets: bool = False
            with self._cid_locks[cid]:
                # Check available slot.
                available_ticket_slots = np.where(self._tickets_arr[cid] == _UINT64_MAX)[0]
                if len(available_ticket_slots) > 0:

                    buffer_frontier_id = self._buf_write_frontier_gc_locked()

                    # Check whether buffer has enough data.
                    next_frame_id: int = self._next_frame_ids[cid]
                    if buffer_frontier_id >= next_frame_id + size: # Possibility for false-negative.
                        # Have enough data, issue ticket.
                        available_ticket_slot = available_ticket_slots[0]
                        self._tickets_arr[cid, available_ticket_slot] = next_frame_id
                        self._next_frame_ids[cid] = next_frame_id + size
                    
                        return FrameTicket(head_id=int(next_frame_id), size=size, buf_id=self.buf_id, cid=cid)
                else:
                    insufficient_tickets = True

            if insufficient_tickets:
                time.sleep(0.001) # time slice rotation, 16ms punishment on Windows.
                continue # retry by polling rather than waiting condition.

            # insufficient buffer, wait buffer with timeout.
            if timeout is not None:
                timeout = end_time - time.monotonic()
                if timeout <= 0: return None # timeout, return.

            with self.buffer.pointer_lock: 
                # Check within lock, memory barrier.
                # wait_for exec predicate in lock, avoid using locked .occupied_count() (dead lock)
                if not self.buffer.data_available_condition.wait(timeout=timeout):
                    # Wait without predicate, let next cycle to check false positive.
                    return None # timeout, return.
                    

    def get_from_ticket(self, ticket: FrameTicket, 
                        timeout: Optional[float] = None) -> Optional[List[np.ndarray]]:
        """
        Extract the data view from the Ticket. Zero-copy.
        Lock-free if the ticket is issued from the same buffer.
        Can also handle cross-buffer ticket safely.
        
        Args:
            ticket (FrameTicket): The ticket to extract data from.
            timeout (Optional[float]): time to wait if ticket is issued from
                another buffer and is requesting future data. None = blocking.
                Will not block if the ticket is issued from the same buffer.
        
        Returns:
            data (Optional[List[np.ndarray]]): The data view from the Ticket. 
                len(data) can be 1 or 2. None only if the ticket is from 
                another buffer and timeout requesting future data.
        
        Raises:
            TicketExpireException: If the ticket is from a registered consumer 
                and is expired (released).
        
        Notes:
            - Will also block async tickets from another fs stream to make data 
              syncronized. If this coupling is not preferred, one should call 
              `get_async` directly from the other fs stream to get newest data.
        """
        if ticket.cid >= 0 and ticket.head_id < self._rb_oldest_frame_ids[self.buf_id]:
            raise TicketExpireException(ticket, self._rb_oldest_frame_ids[self.buf_id])

        if ticket.buf_id != self.buf_id:
            end_time = time.monotonic() + timeout if timeout is not None else 0.0
            while True: # timeout loop
                buf_write_frontier = self._buf_write_frontier_gc_locked()
                if buf_write_frontier >= ticket.head_id + ticket.size:
                    break
                
                if timeout is not None:
                    timeout = end_time - time.monotonic()
                    if timeout <= 0: return None # timeout, return.
                
                with self.buffer.pointer_lock:
                    if not self.buffer.data_available_condition.wait(timeout=timeout):
                        return None # timeout, return.

        relative_ptr = (ticket.head_id + self._rb_offsets[self.buf_id]) % self.buffer.buffer_capacity
        return self.buffer.read_from(relative_ptr, ticket.size)

    def release_sync(self, ticket: FrameTicket):
        """
        Release the ticket acquired from a named consumer. Currently will also 
        trigger GC. Do nothing if a ticket is already released or consumer is 
        unregistered.

        Args:
            ticket (FrameTicket): The ticket acquired from a named consumer to 
                release.
        
        Raises:
            ValueError: If the ticket is from `get_async` or has invalid cid.
        """
        cid = ticket.cid
        if not (0 <= cid < MAX_CONSUMERS):
            raise ValueError(f"Invalid ticket with consumer id: {cid}")

        with self._cid_locks[cid]:
            match_idx = np.where(self._tickets_arr[cid] == ticket.head_id)[0]
            if len(match_idx) > 0:
                self._tickets_arr[cid, match_idx[0]] = _UINT64_MAX
                # Ticket available condition now is checked by polling
            
        self._gc() # avoid unnecessary lock nesting

    def _gc(self, request_num: int = _UINT64_MAX) -> int:
        """
        GC function scans the FrameServer metadata and release the expired data 
        from the RingBuffer.
        `_gc()` will acquire `_gc_locks[self.buf_id]`, please avoid unnecessary 
        lock nesting to prevent dead lock.

        Args:
            request_num (int): number of frames request to release, default is
                _UINT64_MAX which will release as much frames as possible.

        Returns:
            released_num (int): number of the oldest frames actually released.
        """
        # _gc() maintains the coupling of buffer release state and write fs_oldest_frame_id.
        logger = self._logger
        if self._metadata is None or self._c_enable_mask is None or self._gc_view is None:
            # May run in daemon thread, raise is not suitable here.
            if logger: logger.warning("GC is called when metadata not fully initialized. Release nothing.")
            return 0

        # Find the lagging most frame id (occupied for tickets, need to be preserved for next_frame_ids).
        global_occupied_from_id = np.min(self._gc_view) 
        # GC can run concurrently with other get_sync/release_sync. 
        # in 64 bit system, uint64 is atomic, the data inconsistency will only make 
        # global_occupied_from_id lag behind the actual value. (release less, safe)

        # === Critical zone (protect fs_oldest_frame_id, no concurrent _gc()) ===
        if not self._gc_locks[self.buf_id].acquire(block=False):
            return 0 # Let next gc to do works.

        try:
            # Has something to release.
            if global_occupied_from_id > self._rb_oldest_frame_ids[self.buf_id]:
                # If no active consumer, can_release will be very large, near _UINT64_MAX
                can_release = global_occupied_from_id - self._rb_oldest_frame_ids[self.buf_id]
                release_num = self.buffer.release(min(can_release, request_num))
                # Maintain the couple of FS metadata and buffer status.
                self._rb_oldest_frame_ids[self.buf_id] += release_num 
                return release_num
            return 0
        finally: # use `finally` to release lock before leaving current scope.
            self._gc_locks[self.buf_id].release()

    def get_async(self, size: int) -> Optional[FrameTicket]:
        """
        Get the latest `size` frames from the buffer. Due to TOCTOU, when 
        `get_from_ticket` is called to get data view, it may not be the newest
        and may be corrupted or overwritten by producer. Thus this is a 
        dirty-read API. If data integrity should be guaranteed, use 
        `get_async_copy` instead.

        Returns:
            Optional[FrameTicket]: The ticket for the latest `size` frames, 
                or None if the buffer do not have enough frames to return.
        """
        # Due to memory barrier, not guaranteed to be the latest.
        if self.buffer.occupied_count_ < size:
            return None # Buffer do not have enough data.
        with self._gc_locks[self.buf_id]:
            buffer_frontier_id = self._rb_oldest_frame_ids[self.buf_id] + self.buffer.occupied_count_
        
        head_id = buffer_frontier_id - size
        return FrameTicket(head_id=head_id, size=size, buf_id=self.buf_id, cid=-1)

    def get_async_copy(self, size: int) -> Optional[List[NDArray]]:
        """
        Get the deep copy of the latest `size` frames.

        Returns:
            Optional[List[NDArray]]: The deep copy of the latest `size` frames, 
                or None if the buffer do not have enough frames to return.
        """
        if self.buffer.occupied_count_ < size:
            return None # Buffer do not have enough data.

        with self._gc_locks[self.buf_id]: # TODO: Anyway to make copy outside?? probably not except mod ring buffer.
            # Block GC ensures consistent `fs_oldest_frame_id` and `occupied_count_` and data integrity.
            buffer_frontier_id = self._rb_oldest_frame_ids[self.buf_id] + self.buffer.occupied_count_
            head_id = buffer_frontier_id - size

            relative_ptr = (head_id + self._rb_offsets[self.buf_id]) % self.buffer.buffer_capacity
            data_views = self.buffer.read_from(relative_ptr, size)
            return [np.copy(data_view) for data_view in data_views]


    def close(self):
        """Close the metadata shared memory and release buffer link."""
        buf_id = getattr(self, 'buf_id', None)
        if buf_id is not None:
            with self._link_lock:
                if self._rb_linked_fs_count[buf_id] > 0:
                    self._rb_linked_fs_count[buf_id] -= 1
                    if self._rb_linked_fs_count[buf_id] == 0:
                        self._rb_metadata_name_hashes[buf_id] = 0
            self.buf_id = None # Let it crash philosophy for subsequent uses
            
        if hasattr(self, '_metadata'):
            del self._metadata
        if hasattr(self, '_c_enable_mask'):
            del self._c_enable_mask
        if hasattr(self, '_next_frame_ids'):
            del self._next_frame_ids
        if hasattr(self, '_tickets_arr'):
            del self._tickets_arr
        if hasattr(self, '_gc_view'):
            del self._gc_view
        if hasattr(self, '_rb_metadata_name_hashes'):
            del self._rb_metadata_name_hashes
        if hasattr(self, '_rb_linked_fs_count'):
            del self._rb_linked_fs_count
        if hasattr(self, '_rb_oldest_frame_ids'):
            del self._rb_oldest_frame_ids
        if hasattr(self, '_rb_offsets'):
            del self._rb_offsets

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

    def __del__(self):
        buf_id = getattr(self, 'buf_id', None)
        if buf_id is not None:
            if self._link_lock.acquire(block=True, timeout=0.1):
                try:
                    if self._rb_linked_fs_count[buf_id] > 0:
                        self._rb_linked_fs_count[buf_id] -= 1
                        if self._rb_linked_fs_count[buf_id] == 0:
                            self._rb_metadata_name_hashes[buf_id] = 0
                finally:
                    self._link_lock.release()
                
        if hasattr(self, '_metadata'):
            del self._metadata
        if hasattr(self, '_rb_linked_fs_count'):
            del self._rb_linked_fs_count
        if hasattr(self, '_rb_metadata_name_hashes'):
            del self._rb_metadata_name_hashes
        if hasattr(self, '_rb_oldest_frame_ids'):
            del self._rb_oldest_frame_ids
        if hasattr(self, '_rb_offsets'):
            del self._rb_offsets
        if hasattr(self, '_c_enable_mask'):
            del self._c_enable_mask
        if hasattr(self, '_next_frame_ids'):
            del self._next_frame_ids
        if hasattr(self, '_tickets_arr'):
            del self._tickets_arr
        if hasattr(self, '_gc_view'):
            del self._gc_view