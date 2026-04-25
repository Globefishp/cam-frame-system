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
    head_id: np.int64  # The global index maintained by FrameServer
    size: int          # The continuous frames to get started from head_id.

class TicketExpireException(Exception):
    """Ticket is expired."""
    def __init__(self, ticket: FrameTicket, oldest_retained: int):
        self.ticket = ticket
        self.oldest_retained = oldest_retained
        super().__init__(f"Ticket ID {ticket.head_id} is expired. Oldest available frame id: {oldest_retained}.")

MAX_LINKED_BUFFERS = 16
MAX_CONSUMERS = 32
MAX_TICKETS = 32
_UINT64_MAX = np.iinfo(np.uint64).max
version = 3
_METADATA_VER_HASH = int.from_bytes(hashlib.blake2b(version.to_bytes(8, byteorder='little', signed=False), digest_size=8).digest())

class FSMetadata(ctypes.Structure):
    """
    FrameServer Metadata structure for cross-process frame distribution.
    SOA layout for performance and easier numpy manipulation.
    """
    fs_protocol_ver: int
    
    rb_shm_name_hashes:  NDArray[np.uint64]
    rb_linked_fs_count:  NDArray[np.uint8]
    rb_oldest_frame_ids: NDArray[np.uint64]
    rb_offsets:          NDArray[np.uint64]

    c_enable_mask:  NDArray[np.bool_]
    next_frame_ids: NDArray[np.uint64]
    tickets:        NDArray[np.uint64]

    _fields_ = [
        ("fs_protocol_ver", ctypes.c_uint64),    # mark the Metadata version. See _METADATA_VER_HASH
        # for each linked ring buffer:
        ("rb_shm_name_hashes", ctypes.c_uint64 * MAX_LINKED_BUFFERS), # Hash of the linked ring buffer's shm_name to prevent duplicate binding.
        ("rb_linked_fs_count", ctypes.c_uint8 * MAX_LINKED_BUFFERS), # The reference count of each ring buffer linked by frameserver instances. 0 = free slot.
        ("rb_oldest_frame_ids", (ctypes.c_uint64 * 8) * MAX_LINKED_BUFFERS), # The newest frame id of the data that has got from ring buffer by FrameServer. Padded to 64 bytes to prevent false sharing.
        ("rb_offsets", ctypes.c_uint64 * MAX_LINKED_BUFFERS), # The offset of the ring buffer read_ptr when initialize the FrameServer.
        # For each consumer:
        ("c_enable_mask", ctypes.c_bool * MAX_CONSUMERS), # If a consumer is enabled.
        ("tickets", (ctypes.c_uint64 * (MAX_TICKETS + 1)) * MAX_CONSUMERS), # The tickets array, organized as Tickets x Consumers, to avoid False-sharing. 
        # Usually, tickets of the same consumer will be operated in the same process, and different consumers can use different cache-lines.
        # 1. tickets[:, :MAX_TICKETS]: The first frame id of each batch (define as a `ticket`) that IS BEING occupied by consumers.
        # 2. tickets[:, -1]: `next_frame_ids`, the first frame id to be got in the next `get_sync()` call.
    ]