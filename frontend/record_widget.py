import os
from PySide6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt, QTimer, Slot

class RecordWidget(QGroupBox):
    """
    Widget to configure and control video recording.
    Encapsulates path selection, recording toggle, and encoding status display.
    """
    def __init__(self, backend, parent=None):
        super().__init__("Recording Controls", parent)
        self.backend = backend
        self._init_ui()
        
        # Status polling timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(500)

    def _init_ui(self):
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

        # 2. Status Display
        status_layout = QVBoxLayout()
        self.encoded_label = QLabel("Encoded: 0 frames")
        self.speed_label = QLabel("Encoding Speed: 0.0 FPS")
        status_layout.addWidget(self.encoded_label)
        status_layout.addWidget(self.speed_label)
        layout.addLayout(status_layout)

        # 3. Start/Stop Button
        self.btn_toggle = QPushButton("Start Recording")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setEnabled(False) # Enable only when capture is active and path is set
        self.btn_toggle.toggled.connect(self._toggle_recording)
        layout.addWidget(self.btn_toggle)

        # Connect path change to button state
        self.path_edit.textChanged.connect(self._update_button_state)

    def stop(self):
        """Cleanup: stop recording if active and stop timer."""
        if self.btn_toggle.isChecked():
            self.btn_toggle.setChecked(False)
        self.status_timer.stop()

    def set_capture_active(self, active: bool):
        """External call to notify if capture is active, affecting recording availability."""
        self._capture_active = active
        self._update_button_state()
        # Auto stop recording if capture is deactivated
        if not active and self.btn_toggle.isChecked():
            self.btn_toggle.setChecked(False)

    def _update_button_state(self):
        can_record = getattr(self, '_capture_active', False) and bool(self.path_edit.text())
        self.btn_toggle.setEnabled(can_record)

    def _select_output_path(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "MP4 Files (*.mp4)")
        if path:
            self.path_edit.setText(path)

    def _toggle_recording(self, checked):
        if checked:
            output_path = self.path_edit.text()
            try:
                self.backend.start_recording(output_path)
                self.btn_toggle.setText("Stop Recording")
                self.btn_toggle.setStyleSheet("background-color: #f44336; color: white;")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start recording:\n{e}")
                self.btn_toggle.blockSignals(True)
                self.btn_toggle.setChecked(False)
                self.btn_toggle.blockSignals(False)
        else:
            self.backend.stop_recording()
            self.btn_toggle.setText("Start Recording")
            self.btn_toggle.setStyleSheet("")

    def _update_status(self):
        """Poll encoder status from backend."""
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
