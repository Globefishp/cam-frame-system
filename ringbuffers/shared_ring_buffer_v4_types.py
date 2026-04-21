# ringbuffers/shared_ring_buffer_v4_types.py

from dataclasses import dataclass
import ctypes

@dataclass(frozen=True, slots=True) 
# Immutable is important for the data integrity. Ring buffer can skip data validation.
# Produced by ring buffer, used by ring buffer.
class BufferTicket:
    read_ptr: int  # The absolute position for the ring buffer API to get data from.
    read_num: int  # The data slots to read start from read_ptr.


# Define the metadata structure size

METADATA_SIZE = 128 # Use 128 for alignment and future additions
class Metadata(ctypes.Structure):
    '''
    Metadata structure for the shared ring buffer.
    Using ctypes to maximize performance in pointer operations.
    Need to be explicitly released (del obj) in `close()`.
    '''
    capacity: int
    frame_h: int
    frame_w: int
    frame_c: int
    dtype_kind: int
    dtype_bits: int
    read_ptr: int
    write_ptr: int
    occupied_count: int
    
    _fields_ = [
        ("capacity", ctypes.c_int64),
        ("frame_h", ctypes.c_int32),
        ("frame_w", ctypes.c_int32),
        ("frame_c", ctypes.c_int32),
        ("dtype_kind", ctypes.c_int16),
        ("dtype_bits", ctypes.c_int16), # 8 bytes aligned
        ("read_ptr", ctypes.c_int64),
        ("write_ptr", ctypes.c_int64),
        ("occupied_count", ctypes.c_int64),
    ]