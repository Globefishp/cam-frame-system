import sys
from typing import Optional, List, Type, Callable, Any
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QFileDialog, QLineEdit, 
                             QSizePolicy, QGroupBox)
from PySide6.QtCore import Qt, QTimer

from frontend.gl_widget import CameraDisplayWidget
from frontend.gl_upload_thread import GLTextureUploadThread

from loguru._logger import Logger # for type hinting only

class MainWindow(QMainWindow):
    """
    Main Application Window. Manages UI layout and interactions.
    """
    def __init__(self, backend, 
                 panel_classes: Optional[List[Type[QWidget]]] = None, 
                 panel_kwargs: Optional[List[dict]] = None,
                 post_panel_init: Optional[Callable[[List[QWidget]], None]] = None,
                 inject_logger: Optional[Logger] = None):
        super().__init__()
        self.backend = backend
        self._is_capturing = False # For deciding if vsync should be sent to render thread
        
        # Logger Setup
        if inject_logger is not None:
            if isinstance(inject_logger, Logger):
                self._logger = inject_logger.bind(friendly_name="Backend")
            else:
                raise TypeError("inject_logger must be a loguru.Logger instance.")
        else:
            self._logger = None

        self.setWindowTitle("Camera System - PySide6 + ModernGL")
        self.setGeometry(100, 100, 1000, 700)

        # Main widget & layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left: OpenGL Display ---
        self.display_widget = CameraDisplayWidget()
        self.display_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.display_widget, 4)

        # --- Right: Controls Panel ---
        controls_widget = QWidget()
        controls_widget.setMinimumWidth(280)
        controls_layout = QVBoxLayout(controls_widget)

        # Initialize injected panels
        panel_classes = panel_classes or []
        panel_kwargs = panel_kwargs or []
        self.panel_widgets: List[QWidget] = []
        
        _capture_toggled_registered = False
        _bboxes_to_draw_registered = False

        for idx, WidgetClass in enumerate(panel_classes):
            kwargs = panel_kwargs[idx] if idx < len(panel_kwargs) else {}
            kwargs.update({'backend': self.backend})
            # Fail-fast: Init widget, expecting it supports `NeedBackend` protocol
            widget = WidgetClass(**kwargs)

            # Contract check: Lifecycle method `stop`
            if not hasattr(widget, 'stop') or not callable(getattr(widget, 'stop')):
                err_msg = f"QWidget {WidgetClass.__name__} missing lifecycle method '.stop()'"
                raise TypeError(err_msg)

            # Duck-typing wiring: SendCaptureState
            if hasattr(widget, 'capture_toggled'):
                if _capture_toggled_registered:
                    if self._logger:
                        self._logger.warning("Multiple widgets has the capability of "
                            f"'SendCaptureState', only the first one will take effect, skipping {WidgetClass.__name__}")
                else:
                    widget.capture_toggled.connect(self._on_capture_toggled)
                    _capture_toggled_registered = True

            # Duck-typing wiring: SendBBoxDrawing
            if hasattr(widget, 'bboxes_to_draw'):
                if _bboxes_to_draw_registered:
                    if self._logger:
                        self._logger.warning("Multiple widgets has the capability of "
                            f"'SendBBoxDrawing', only the first one will take effect, skipping {WidgetClass.__name__}")
                else:
                    widget.bboxes_to_draw.connect(self.display_widget.update_overlay_lines)
                    _bboxes_to_draw_registered = True

            self.panel_widgets.append(widget)
            controls_layout.addWidget(widget)

        # Custom post-init operations from caller
        if post_panel_init is not None:
            post_panel_init(self.panel_widgets)

        controls_layout.addStretch(1)

        # 4. Status Display (Bottom)
        status_group = QGroupBox("Statistics")
        status_vbox = QVBoxLayout(status_group)
        
        self.buffer_label = QLabel(f"Buffer load: 0 / {backend.frame_server.buffer.buffer_capacity} frames (0.0 %)")
        status_vbox.addWidget(self.buffer_label)
        
        controls_layout.addWidget(status_group)

        # Status Update Timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(500)

        main_layout.addWidget(controls_widget, 1)

        # --- Render Thread & VSync Setup ---
        # Force context creation for the main widget so we can share it
        self.display_widget.grabFramebuffer()
        share_context = self.display_widget.context()
        
        if not share_context:
            print("Fatal Error: Could not get shared context from QOpenGLWidget.")
            sys.exit(1)

        # Start the Background Rendering Thread
        self.render_thread = GLTextureUploadThread(self.backend, share_context)
        self.render_thread.frame_ready.connect(self.display_widget.on_frame_ready, Qt.QueuedConnection)
        
        # Connect vsync to trigger backend frame fetch and schedule next render
        self.display_widget.frameSwapped.connect(self._on_frame_swapped)

        self.render_thread.start()

    def _update_status(self):
        """Update status in UI"""
        # buffer load
        buffer_count = self.backend.frame_server.buffer.occupied_count_
        buffer_capacity = self.backend.frame_server.buffer.buffer_capacity
        self.buffer_label.setText(f"Buffer load: {buffer_count} / {buffer_capacity} frames "
                                  f"({buffer_count / buffer_capacity * 100:.1f}%)")
        
        # TODO: can add color warning based on buffer load

    def _on_frame_swapped(self):
        """Triggered on VSync. Wakes up render thread and requests next frame draw."""
        if self._is_capturing:
            # Do not send vsync signal to save CPU.
            self.render_thread.vsync_event.set()
            self.display_widget.update()

    def _on_capture_toggled(self, checked: bool):
        """Handle capture state changes and broadcast to SenseCaptureState widgets."""
        self._is_capturing = checked
        for widget in self.panel_widgets:
            method: Optional[Callable[[bool], None]] = getattr(widget, 'set_capture_active', None)
            if method:
                method(checked)

    def closeEvent(self, event):
        """Handle cleanup on exit."""
        # Notify all injected widgets to cleanup their own resources
        for widget in self.panel_widgets:
            try:
                widget.stop()
            except Exception as e:
                if self._logger:
                    self._logger.opt(exception=e).error(f"Error closing widget when calling {widget.__class__.__name__}.stop()")
        
        if getattr(self, 'render_thread', None):
            self.render_thread.stop()
            
        event.accept()
