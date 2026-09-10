from typing import Protocol, runtime_checkable
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

# Note: Python's Protocol doesn't perfectly validate class attributes like PySide Signals at runtime via isinstance(),
# because Signals are descriptors (class attributes), not instance attributes until bound.
# Thus, MainWindow uses `hasattr(widget, 'signal_name')` for runtime checking.

# These Protocols serve as structural documentation and Type Hinting for developers.
# Last update: 260910

@runtime_checkable
class NeedBackend(Protocol):
    """
    Contract for components that require the HeadlessBackend for initialization.
    
    If a Widget class implements this, it must accept `backend` as an argument
    in its `__init__` constructor. This is automatically handled by the 
    MainWindow's initialization.
    """
    def __init__(self, backend, *args, **kwargs):
        ...

@runtime_checkable
class SendCaptureState(Protocol):
    """
    Contract for components that control and broadcast the global capture state.
    
    If a Widget has a `capture_toggled` Signal(bool), MainWindow will:
    1. Connect to it to manage internal VSync and rendering states.
    2. Broadcast its emitted state to all components implementing `SenseCaptureState`.
    
    Warning: Only the first component found with this capability will be connected.
    """
    capture_toggled: Signal  # Signal(bool)

@runtime_checkable
class SenseCaptureState(Protocol):
    """
    Contract for components that need to respond to global capture state changes.
    
    If a Widget implements `set_capture_active`, MainWindow will actively call this 
    method whenever a component with `SendCaptureState` emits a capture state change.
    """
    def set_capture_active(self, active: bool) -> None:
        ...

@runtime_checkable
class SendBBoxDrawing(Protocol):
    """
    Contract for components that generate bounding boxes for rendering.
    
    If a Widget has a `bboxes_to_draw` Signal(list), MainWindow will automatically
    connect it to the GL Display Widget's overlay rendering slot.
    
    Warning: Only the first component found with this capability will be connected 
    to avoid flickering from multiple data sources.
    """
    bboxes_to_draw: Signal  # Signal(list)
