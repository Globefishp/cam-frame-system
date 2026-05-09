import os
import time
import threading
from PySide6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
                               QCheckBox, QSpinBox)
from PySide6.QtCore import Qt, QTimer, Slot, Signal, QThread

class RecordWidget(QGroupBox):
    """
    Widget to configure and control video recording.
    Encapsulates path selection, recording toggle, and encoding status display.
    """
    def __init__(self, backend, parent=None):
        super().__init__("Recording Controls", parent)
        self.backend = backend
        self.rotation_thread = None
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
                
                # Setup rotation thread if enabled or manual rotate is desired
                # Even if auto-rotate is off, the thread only handle manual rotate button
                interval = self.spin_rotate_interval.value() * 60 if self.cb_auto_rotate.isChecked() else None
                self.rotation_thread = RotationThread(self.backend, output_path, interval)
                self.rotation_thread.disable_ui.connect(self._disable_ui)
                self.rotation_thread.enable_ui.connect(self._enable_ui)
                self.rotation_thread.error.connect(self._on_rotation_error)
                self.rotation_thread.path_updated.connect(self.path_edit.setText)
                self.rotation_thread.start()
                
                self.cb_auto_rotate.setEnabled(False)
                self.spin_rotate_interval.setEnabled(False)
                self.btn_rotate_now.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start recording:\n{e}")
                self.btn_toggle.blockSignals(True)
                self.btn_toggle.setChecked(False)
                self.btn_toggle.blockSignals(False)
        else:
            if self.rotation_thread:
                self.rotation_thread.stop()
                self.rotation_thread.wait()
                self.rotation_thread = None
                
            self.backend.stop_recording()
            self.btn_toggle.setText("Start Recording")
            self.btn_toggle.setStyleSheet("")
            self.cb_auto_rotate.setEnabled(True)
            self.spin_rotate_interval.setEnabled(True)
            self.btn_rotate_now.setEnabled(False)

    def _trigger_manual_rotation(self):
        if self.rotation_thread:
            self.rotation_thread.trigger_rotation()

    @Slot()
    def _disable_ui(self):
        self.btn_toggle.setEnabled(False)
        self.btn_rotate_now.setEnabled(False)

    @Slot()
    def _enable_ui(self):
        self.btn_toggle.setEnabled(True)
        self.btn_rotate_now.setEnabled(True)

    def _on_rotation_error(self, err_msg):
        QMessageBox.warning(self, "Rotation Failed", err_msg)

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

class RotationThread(QThread):
    """
    Background thread to handle rotation timing and execution.
    Name it 'handle_rotation' conceptually.
    """
    disable_ui = Signal()
    enable_ui = Signal()
    path_updated = Signal(str)
    error = Signal(str)

    def __init__(self, backend, base_path, interval_sec=None):
        super().__init__()
        self.backend = backend
        self.base_path = base_path
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._manual_trigger = threading.Event()

    def stop(self):
        self._stop_event.set()
        self._manual_trigger.set() # wake up

    def trigger_rotation(self):
        self._manual_trigger.set()

    def run(self):
        last_rotate_time = time.time()
        while not self._stop_event.is_set():
            if self.interval_sec is not None:
                wait_time = max(0, self.interval_sec - (time.time() - last_rotate_time))
                triggered = self._manual_trigger.wait(timeout=wait_time)
            else:
                # Only manual trigger
                triggered = self._manual_trigger.wait()
            
            if self._stop_event.is_set():
                break
            
            # Reset event and perform rotation
            self._manual_trigger.clear()
            self.disable_ui.emit()
            
            # Generate next path using POSIX timestamp
            base, ext = os.path.splitext(self.base_path)
            new_path = f"{base}_{int(time.time())}{ext}"
            
            try:
                self.backend.rotate_recording(new_path)
                self.path_updated.emit(new_path)
            except Exception as e:
                cur_time = time.strftime("%H:%M:%S", time.localtime())
                self.error.emit(f"[{cur_time}] Rotation failed: {e}")
            
            self.enable_ui.emit()
            last_rotate_time = time.time()
