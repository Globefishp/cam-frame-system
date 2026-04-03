# abstractcamera_types.py

# Public types for abstract camera class, including Exceptions and CameraFeatures.

# Add extra CameraFeatures here if needed (e.g. a new camera is added).

from enum import IntFlag

__all__ = [
    "CamException",
    "CameraFeatures"
]

class CamException(Exception):
    """Base exception class for camera error."""
    def __init__(self, msg: str, 
                 src_func: str = "UnspecifiedFunction"):
        message: str = f"[{src_func}] {msg}"
        super().__init__(message)

class CameraFeatures(IntFlag):
    """CameraFeature class enumerate common camera features."""
    NONE          = 0      # Default capability.
    TIMECODE      = 1 << 0 # Hardware timecode support/enabled.
    # TIMECODE mode differs at: data transfer, downstream(analyzer/encoder) parsing.
    # An AbstractCamera obj will be passed to downstream to use staticmethod to parse metadata(timecode).
    # Downstream abstract class should add the ability to handle extended info dict.
    HW_TRIGGER    = 1 << 1 # Hardware trigger support/enabled.
    # HW_TRIGGER mode differs at: Camera configuration, grab frame logic (more likely blocking).
    # Timeout part may need careful consideration.
    SW_BLC        = 1 << 2 # Software Black level correction support/enabled. Needs a BLCManager.
    # SW_BLC Currently specific to PCOEdge42Camera.
    GAIN          = 1 << 3 # HW or SW Gain control. Need override `gain` & `gain_range` property in base class.
    # GAIN is purely a Camera status, just use feature() to check and draw UI.