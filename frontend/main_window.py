import sys
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QFileDialog, QLineEdit, 
                             QSizePolicy, QGroupBox)
from PySide6.QtCore import Qt, QTimer

from frontend.gl_widget import CameraDisplayWidget
from frontend.gl_upload_thread import GLTextureUploadThread
from frontend.analyzer_widget import AnalyzerWidget

class MainWindow(QMainWindow):
    """
    Main Application Window. Manages UI layout and interactions.
    """
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
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

        # 1. Exposure Control
        controls_layout.addWidget(QLabel("Exposure (ms)"))
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(1, 100)
        self.exposure_slider.setValue(10)
        controls_layout.addWidget(self.exposure_slider)
        self.exposure_label = QLabel(f"{self.exposure_slider.value()} ms")
        controls_layout.addWidget(self.exposure_label)

        # 2. FPS Control
        controls_layout.addWidget(QLabel("FPS"))
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setRange(1, 100)
        self.fps_slider.setValue(30) # Default
        controls_layout.addWidget(self.fps_slider)
        self.fps_label = QLabel(f"{self.fps_slider.value()} FPS")
        controls_layout.addWidget(self.fps_label)

        # 3. Recording Path Selection
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Output:"))
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        path_layout.addWidget(self.path_edit)
        self.browse_button = QPushButton("...")
        self.browse_button.setFixedWidth(30)
        path_layout.addWidget(self.browse_button)
        controls_layout.addLayout(path_layout)
        
        # 4. Analyzer Controls
        self.analyzer_widget = AnalyzerWidget(self.backend)
        controls_layout.addWidget(self.analyzer_widget)
        # Connect bboxes to draw signal
        self.analyzer_widget.bboxes_to_draw.connect(self.display_widget.update_overlay_lines)

        controls_layout.addStretch(1)

        # 5. Status Display
        status_group = QGroupBox("Statistics")
        status_vbox = QVBoxLayout(status_group)
        
        self.buffer_label = QLabel(f"Buffer load: 0 / {backend.frame_server.buffer.buffer_capacity} frames (0.0 %)")
        self.encoded_frames_label = QLabel("Encoded: 0 frames")
        self.encoding_fps_label = QLabel("Encoding Speed: 0.0 FPS")
        
        status_vbox.addWidget(self.buffer_label)
        status_vbox.addWidget(self.encoded_frames_label)
        status_vbox.addWidget(self.encoding_fps_label)
        
        controls_layout.addWidget(status_group)

        # Status Update Timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(500)

        # 5. Main Action Buttons
        self.capture_button = QPushButton("Start Capture")
        self.capture_button.setCheckable(True)
        controls_layout.addWidget(self.capture_button)

        self.record_button = QPushButton("Start Recording")
        self.record_button.setCheckable(True)
        self.record_button.setEnabled(False)
        controls_layout.addWidget(self.record_button)
        
        main_layout.addWidget(controls_widget, 1)

        # --- Connect Signals ---
        self.exposure_slider.valueChanged.connect(self._on_exposure_changed)
        self.fps_slider.valueChanged.connect(self._on_fps_changed)
        self.browse_button.clicked.connect(self.select_output_path)
        self.path_edit.textChanged.connect(self._update_record_button_state)
        self.capture_button.toggled.connect(self.toggle_capture)
        self.record_button.toggled.connect(self.toggle_recording)

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

        # encoder status
        encoder = getattr(self.backend, 'encoder', None)
        status = getattr(encoder, 'status', {}) if encoder else {}

        # encoder frame count
        frame_count = status.get('frame_count', 0)
        fps = status.get('fps', 0.0)

        self.encoded_frames_label.setText(f"Encoded: {frame_count} frames")
        self.encoding_fps_label.setText(f"Encoding Speed: {fps:.1f} FPS")
        
        # TODO: can add color warning based on buffer load

    def _on_frame_swapped(self):
        """Triggered on VSync. Wakes up render thread and requests next frame draw."""
        if self.capture_button.isChecked():
            # Do not send vsync signal to save CPU.
            self.render_thread.vsync_event.set()
            self.display_widget.update()

    def _on_exposure_changed(self, value):
        self.exposure_label.setText(f"{value} ms")
        self.backend.set_exposure_time(float(value))

    def _on_fps_changed(self, value):
        self.fps_label.setText(f"{value} FPS")
        self.backend.set_fps(float(value))

    def select_output_path(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "MP4 Files (*.mp4)")
        if path:
            self.path_edit.setText(path)

    def _update_record_button_state(self):
        can_record = self.capture_button.isChecked() and bool(self.path_edit.text())
        self.record_button.setEnabled(can_record)

    def toggle_capture(self, checked):
        if checked:
            self.capture_button.setText("Stop Capture")
            self.backend.start_capture()
            self._update_record_button_state()
        else:
            self.capture_button.setText("Start Capture")
            self.backend.stop_capture()
            if self.record_button.isChecked():
                self.record_button.setChecked(False)
            self.record_button.setEnabled(False)

    def toggle_recording(self, checked):
        if checked:
            output_path = self.path_edit.text()
            if not output_path:
                self.record_button.setChecked(False)
                return
            self.record_button.setText("Stop Recording")
            self.backend.start_recording(output_path)
            self.fps_slider.setEnabled(False)
        else:
            self.record_button.setText("Start Recording")
            self.backend.stop_recording()
            self.fps_slider.setEnabled(True)

    def closeEvent(self, event):
        """Handle cleanup on exit."""
        if self.record_button.isChecked():
            self.backend.stop_recording()
            
        # Notify analyzer widget to cleanup its own resources (worker thread, plot window)
        if hasattr(self, 'analyzer_widget'):
            self.analyzer_widget.stop()
        
        if getattr(self, 'render_thread', None):
            # self.render_thread.frame_ready.disconnect()
            self.render_thread.stop()
            
        event.accept()
