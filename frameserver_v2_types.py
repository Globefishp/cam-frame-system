# frameserver_v2_types.py

import hashlib
import ctypes
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class FrameTicket:
    """
    FrameTicket is issued by FrameServer and use absolute index to identify frame batches.
    """
    head_id: int    # The global index maintained by FrameServer
    size: int       # The continuous frames to get started from head_id.

class TicketExpireException(Exception):
    """Ticket is expired."""
    def __init__(self, ticket: FrameTicket, oldest_retained: int):
        self.ticket = ticket
        self.oldest_retained = oldest_retained
        super().__init__(f"Ticket ID {ticket.head_id} is expired. Oldest available frame id: {oldest_retained}.")

MAX_CONSUMERS = 32
MAX_TICKETS = 32
_INT64_MAX = np.iinfo(np.int64).max
version = 2
_METADATA_VER_HASH = int.from_bytes(hashlib.blake2b(version.to_bytes(8, byteorder='big', signed=False), digest_size=8).digest())

class FSMetadata(ctypes.Structure):
    """
    FrameServer Metadata structure for cross-process frame distribution.
    SOA layout for performance and easier numpy manipulation.
    """
    fs_protocol_ver: int
    fs_oldest_frame_id: int
    rb_offset: int
    enable_mask: NDArray[np.bool_]
    next_frame_ids: NDArray[np.int64]
    tickets_arr: NDArray[np.int64]

    _fields_ = [
        ("fs_protocol_ver", ctypes.c_uint64),    # mark the Metadata version. See _METADATA_VER_HASH
        ("fs_oldest_frame_id", ctypes.c_int64), # The newest frame id of the data that has got from ring buffer by FrameServer.
        ("rb_offset", ctypes.c_int64), # The offset of the ring buffer read_ptr when initialize the FrameServer.
        # For each consumer:
        ("enable_mask", ctypes.c_bool * MAX_CONSUMERS), # If a consumer is enabled.
        ("next_frame_ids", ctypes.c_int64 * MAX_CONSUMERS), # The first frame id to be got in the next `get_sync()` call.
        ("tickets", (ctypes.c_int64 * MAX_TICKETS) * MAX_CONSUMERS), # The first frame id of each batch (define as a `ticket`) that IS BEING occupied by consumers.
        # tickets use Tickets x Consumers, to avoid False-sharing. Usually, different consumer's tickets will operate in the same process.
    ]