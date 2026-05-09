import multiprocessing as mp
import numpy as np
from .frameserver_v3_types import FSMetadata as FSMetadata, FrameTicket as FrameTicket, MAX_CONSUMERS as MAX_CONSUMERS, MAX_LINKED_BUFFERS as MAX_LINKED_BUFFERS, MAX_TICKETS as MAX_TICKETS, TicketExpireException as TicketExpireException
from _typeshed import Incomplete
from loguru._logger import Logger
from numpy.typing import NDArray as NDArray
from ringbuffers.shared_ring_buffer_v4 import ProcessSafeSharedRingBuffer
from typing import Literal, overload

class FrameServer:
    '''
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
    (Common limitation for mp objects, standard picklization won\'t work)
    '''
    @overload
    def __init__(self, create: Literal[True] = ..., *, ring_buffer: ProcessSafeSharedRingBuffer, frameserver: None = None, inject_logger: Logger | None = None) -> None:
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
    @overload
    def __init__(self, create: Literal[False], *, ring_buffer: None = None, frameserver: FrameServer, inject_logger: Logger | None = None) -> None:
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
    @overload
    def __init__(self, create: Literal[False], *, ring_buffer: ProcessSafeSharedRingBuffer, frameserver: FrameServer, inject_logger: Logger | None = None) -> None:
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
    @property
    def shm_name(self) -> str: ...
    @property
    def reg_lock(self) -> mp.Lock: ...
    @property
    def cid_locks(self) -> list[mp.Lock]: ...
    @property
    def gc_locks(self) -> list[mp.Lock]: ...
    @property
    def link_lock(self) -> mp.Lock: ...
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
    def unregister_consumer(self, cid: int):
        """
        Unregister consumer of given cid, release related resources.
        Do nothing if a consumer has already unregistered.

        Args:
            cid (int): consumer id to be unregistered.

        Raises:
            ValueError: If the consumer id is invalid.
        """
    def get_sync(self, cid: int, size: int, timeout: float | None = None) -> FrameTicket | None:
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
    def get_from_ticket(self, ticket: FrameTicket, timeout: float | None = None) -> list[np.ndarray] | None:
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
    def get_async(self, size: int) -> FrameTicket | None:
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
    def get_async_copy(self, size: int) -> list[NDArray] | None:
        """
        Get the deep copy of the latest `size` frames.

        Returns:
            Optional[List[NDArray]]: The deep copy of the latest `size` frames, 
                or None if the buffer do not have enough frames to return.
        """
    buf_id: Incomplete
    def close(self) -> None:
        """Close the metadata shared memory and release buffer link."""
    def unlink(self) -> None:
        """Unlink the metadata shared memory, should be called by the creator process."""
    def __del__(self) -> None: ...
