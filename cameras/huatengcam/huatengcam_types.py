# huatengcam_types.py
# Enums and Exceptions

from ..abstractcamera import CamException
from cameras.huatengcam.mvsdk_mod import CameraException as MvCamException
from typing import Optional
from enum import IntEnum

class HuatengSDKException(CamException):
    """Huateng Camera Exception"""
    def __init__(self, exception: MvCamException,
                 src_func: str = "Unspecified",
                 extra_info: str = ""):
        self.error_code = exception.error_code
        self.extra_info = extra_info
        extra_info = extra_info + ":"
        message: str
        if exception.error_code is not None:
            message = f"[SDK_ERR {exception.error_code}] {extra_info} {exception.message}"
        else:
            message = exception.message
        super().__init__(message, src_func=src_func)

class TriggerMode(IntEnum):
    """Trigger Mode"""
    FREERUN = 0
    SOFT_TRIGGER = 1

class BitDepth(IntEnum):
    """Bit Depth"""
    _8 = 0
    _12 = 1

class BayerPattern(IntEnum):
    """Bayer Pattern"""
    BGGR = 0
    RGGB = 1

