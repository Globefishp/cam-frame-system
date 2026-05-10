from typing import Optional, Any
from PySide6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QWidget,
                               QPushButton, QLabel, QSlider, QCheckBox)
from PySide6.QtCore import Qt, Signal, Slot

class CaptureWidget(QGroupBox):
    """
    Widget to control camera capture settings and state.
    Includes Exposure, FPS control and Start/Stop toggle.
    """
    capture_toggled = Signal(bool)

    def __init__(self, backend: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__("Capture Controls", parent)
        self.backend = backend
        self._is_capturing = False
        self._init_ui()
        self._load_initial_values()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # 1. Exposure Control
        layout.addWidget(QLabel("Exposure (ms)"))
        exposure_layout = QHBoxLayout()
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(1, 100)
        exposure_layout.addWidget(self.exposure_slider)
        self.exposure_label = QLabel("0 ms")
        exposure_layout.addWidget(self.exposure_label)
        layout.addLayout(exposure_layout)

        # 2. FPS Control
        layout.addWidget(QLabel("FPS"))
        fps_layout = QHBoxLayout()
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setRange(1, 100)
        fps_layout.addWidget(self.fps_slider)
        self.fps_label = QLabel("0 FPS")
        fps_layout.addWidget(self.fps_label)
        layout.addLayout(fps_layout)

        # 3. Burn Timestamp Option
        self.cb_burn_ts = QCheckBox("Burn Timestamp")
        layout.addWidget(self.cb_burn_ts)

        # 4. Capture Button
        self.btn_toggle_capture = QPushButton("Start Capture")
        self.btn_toggle_capture.setCheckable(True)
        layout.addWidget(self.btn_toggle_capture)

        # Connect signals
        self.exposure_slider.valueChanged.connect(self._on_exposure_changed)
        self.fps_slider.valueChanged.connect(self._on_fps_changed)
        self.cb_burn_ts.toggled.connect(self._on_burn_ts_toggled)
        self.btn_toggle_capture.clicked.connect(self._on_btn_capture_clicked)

    def _load_initial_values(self) -> None:
        """Fetch current settings from backend."""
        exposure = self.backend.get_exposure_time()
        fps = self.backend.get_fps()
        burn_ts = self.backend.get_burn_ts_enabled()
        
        self.exposure_slider.setValue(int(exposure))
        self.fps_slider.setValue(int(fps))
        self.cb_burn_ts.setChecked(burn_ts)
        
        self._update_ui_state()

    def _on_exposure_changed(self, value: int) -> None:
        self.backend.set_exposure_time(float(value))
        self._update_ui_state()

    def _on_fps_changed(self, value: int) -> None:
        self.backend.set_fps(float(value))
        self._update_ui_state()

    def _on_burn_ts_toggled(self, checked: bool) -> None:
        self.backend.set_burn_ts_enabled(checked)

    def _on_btn_capture_clicked(self, checked: bool) -> None:
        """Handle user intent to start/stop capture."""
        try:
            if checked:
                self.backend.start_capture()
            else:
                self.backend.stop_capture()
            
            self._is_capturing = checked
            self.capture_toggled.emit(checked)
        finally:
            self._update_ui_state()

    def _update_ui_state(self) -> None:
        """
        Centralized UI state management.
        Updates control appearances based on the current state.
        """
        # Update labels
        self.exposure_label.setText(f"{self.exposure_slider.value()} ms")
        self.fps_label.setText(f"{self.fps_slider.value()} FPS")

        # Sync button state
        self.btn_toggle_capture.setChecked(self._is_capturing)
        
        if self._is_capturing:
            self.btn_toggle_capture.setText("Stop Capture")
            self.btn_toggle_capture.setStyleSheet("background-color: #f44336; color: white;")
        else:
            self.btn_toggle_capture.setText("Start Capture")
            self.btn_toggle_capture.setStyleSheet("")

    def stop(self) -> None:
        """Cleanup logic. Robust and idempotent."""
        if self._is_capturing:
            try:
                self.backend.stop_capture()
            except Exception:
                return

            self._is_capturing = False
            self.capture_toggled.emit(False)
            self._update_ui_state()
