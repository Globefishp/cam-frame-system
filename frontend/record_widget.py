import os
import time
import threading
from typing import Optional, Any
from PySide6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QWidget,
                               QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
                               QCheckBox, QSpinBox)
from PySide6.QtCore import Qt, QTimer, Slot, Signal, QThread
from .record_handling_thread import RecordingThread

class RecordWidget(QGroupBox):
    """
    Widget to configure and control video recording.
    Encapsulates path selection, recording toggle, and encoding status display.
    """
    def __init__(self, backend: Any, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the recording widget.

        :param backend: The backend system object providing recording APIs.
        :param parent: Optional parent widget.
        """
        super().__init__("Recording Controls", parent)
        self.backend: Any = backend
        self.rotation_thread: Optional[RecordingThread] = None
        self._base_path: str = ""
        self._capture_active: bool = False
        
        self._init_ui()
        
        # Status polling timer
        self.status_timer: QTimer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(500)

    def _init_ui(self) -> None:
        """Initialize and layout the UI components."""
        layout = QVBoxLayout(self)

        # 1. Output Path Selection
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Output:"))
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        path_layout.addWidget(self.path_edit)
        self.btn_browse = QPushButton("...")
        self.btn_browse.setFixedWidth(30)
        self.btn_browse.clicked.connect(self._select_output_path)
        path_layout.addWidget(self.btn_browse)
        layout.addLayout(path_layout)

        # 1. Rotation Options
        rotate_layout = QHBoxLayout()
        self.cb_auto_rotate = QCheckBox("Auto Rotate")
        rotate_layout.addWidget(self.cb_auto_rotate)
        
        rotate_layout.addWidget(QLabel("Every"))
        self.spin_rotate_interval = QSpinBox()
        self.spin_rotate_interval.setRange(1, 1440)
        self.spin_rotate_interval.setValue(10)
        self.spin_rotate_interval.setSuffix(" min")
        rotate_layout.addWidget(self.spin_rotate_interval)
        layout.addLayout(rotate_layout)

        # 3. Recording Controls (Start/Stop and Rotate)
        btns_layout = QHBoxLayout()
        
        self.btn_toggle = QPushButton("Start Recording")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setEnabled(False) # Enable only when capture is active and path is set
        self.btn_toggle.toggled.connect(self._toggle_recording)
        btns_layout.addWidget(self.btn_toggle)

        self.btn_rotate_now = QPushButton("Rotate Now")
        self.btn_rotate_now.setEnabled(False)
        self.btn_rotate_now.clicked.connect(self._trigger_manual_rotation)
        btns_layout.addWidget(self.btn_rotate_now)
        
        layout.addLayout(btns_layout)

        # 4. Status Display
        status_layout = QVBoxLayout()
        self.encoded_label = QLabel("Encoded: 0 frames")
        self.speed_label = QLabel("Encoding Speed: 0.0 FPS")
        status_layout.addWidget(self.encoded_label)
        status_layout.addWidget(self.speed_label)
        layout.addLayout(status_layout)

        # Connect path change to button state and base path tracking
        self.path_edit.textChanged.connect(self._on_path_changed)
        self.path_edit.textChanged.connect(self._update_button_state)

    def stop(self) -> None:
        """
        Cleanup and stop any active recording and timers.
        Should be called when the application or parent window is closing.
        """
        if self.btn_toggle.isChecked():
            self.btn_toggle.setChecked(False)
        self.status_timer.stop()

    def set_capture_active(self, active: bool) -> None:
        """
        Notify the widget whether the camera capture is currently active.
        This affects the availability of the recording controls.

        :param active: True if camera capture is active, False otherwise.
        """
        self._capture_active = active
        self._update_button_state()
        # Auto stop recording if capture is deactivated
        if not active and self.btn_toggle.isChecked():
            self.btn_toggle.setChecked(False)

    def _update_button_state(self) -> None:
        """Update the enabled/disabled state of the recording toggle button."""
        # Enable recording only if camera capture is active and an output path is selected
        can_record = getattr(self, '_capture_active', False) and bool(self.path_edit.text())
        self.btn_toggle.setEnabled(can_record)

    @Slot(str)
    def _on_path_changed(self, text: str) -> None:
        """Handle manual changes to the output path text."""
        self._base_path = text

    def _select_output_path(self) -> None:
        """Open a file dialog to select the recording output path."""
        path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "MP4 Files (*.mp4)")
        if path:
            self.path_edit.setText(path)

    def _toggle_recording(self, checked: bool) -> None:
        """Handle the recording start/stop toggle action."""
        if checked:
            # Prepare interval and timestamp settings
            interval = self.spin_rotate_interval.value() * 60 if self.cb_auto_rotate.isChecked() else None
            file_ts = self.cb_auto_rotate.isChecked()
            
            try:
                # Update UI to recording state
                self.btn_toggle.setText("Stop Recording")
                self.btn_toggle.setStyleSheet("background-color: #f44336; color: white;")
                
                # Start the background thread which manages the actual recording lifecycle
                self.rotation_thread = RecordingThread(self.backend, self._base_path, interval, file_ts)
                self.rotation_thread.disable_ui.connect(self._disable_ui)
                self.rotation_thread.enable_ui.connect(self._enable_ui)
                self.rotation_thread.error.connect(self._on_rotation_thread_error)
                self.rotation_thread.path_updated.connect(self.path_edit.setText)
                self.rotation_thread.start()
                
                self.cb_auto_rotate.setEnabled(False)
                self.spin_rotate_interval.setEnabled(False)
                self.btn_rotate_now.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to initialize recording:\n{e}")
                self.btn_toggle.setChecked(False)
        else:
            # Stop the thread and wait for it to finish (this will trigger backend.stop_recording)
            if self.rotation_thread:
                self.rotation_thread.stop()
                self.rotation_thread.wait()
                self.rotation_thread = None
                
            # Reset UI to idle state
            self.btn_toggle.setText("Start Recording")
            self.btn_toggle.setStyleSheet("")
            self.cb_auto_rotate.setEnabled(True)
            self.spin_rotate_interval.setEnabled(True)
            self.btn_rotate_now.setEnabled(False)

    def _trigger_manual_rotation(self) -> None:
        """Trigger an immediate file rotation through the background thread."""
        if self.rotation_thread:
            self.rotation_thread.trigger_rotation()

    @Slot()
    def _disable_ui(self) -> None:
        """Disable recording controls during rotation transition."""
        self.btn_toggle.setEnabled(False)
        self.btn_rotate_now.setEnabled(False)

    @Slot()
    def _enable_ui(self) -> None:
        """Enable recording controls after rotation transition is complete."""
        self.btn_toggle.setEnabled(True)
        self.btn_rotate_now.setEnabled(True)

    def _on_rotation_thread_error(self, err_msg: str) -> None:
        """Handle errors reported by the rotation thread."""
        QMessageBox.warning(self, "Recording Failed", err_msg)
        if self.btn_toggle.isChecked():
            self.btn_toggle.setChecked(False)

    def _update_status(self) -> None:
        """Poll and update the recording status display from the backend."""
        encoder = getattr(self.backend, 'encoder', None)
        if encoder:
            status = encoder.status
            frame_count = status.get('frame_count', 0)
            fps = status.get('fps', 0.0)
            self.encoded_label.setText(f"Encoded: {frame_count} frames")
            self.speed_label.setText(f"Encoding Speed: {fps:.1f} FPS")
        elif not self.btn_toggle.isChecked():
            # Reset labels if not recording
            self.encoded_label.setText("Encoded: 0 frames")
            self.speed_label.setText("Encoding Speed: 0.0 FPS")

