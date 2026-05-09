import os
import numpy as np
from PySide6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from .analyzer_listening_thread import AnalyzerListeningThread
from .analyzer_plot_window import AnalyzerPlotWindow

class AnalyzerWidget(QGroupBox):
    """
    Generic Widget to configure and control an Analyzer.
    Includes configuration for Model Path, Save Path, and Start/Stop toggle.
    Also manage listening thread and plot window.
    """
    # Translate and forward the bbox to lines to draw in the gl widget
    bboxes_to_draw = Signal(object) # object: 

    def __init__(self, backend, parent=None):
        super().__init__("Analyzer Controls", parent)
        self.backend = backend
        self._init_ui()
        
        # Status polling timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(500)

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Model Path Selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_edit = QLineEdit()
        self.model_edit.setReadOnly(True)
        model_layout.addWidget(self.model_edit)
        self.btn_browse_model = QPushButton("...")
        self.btn_browse_model.setFixedWidth(30)
        self.btn_browse_model.clicked.connect(self._select_model_path)
        model_layout.addWidget(self.btn_browse_model)
        layout.addLayout(model_layout)

        # 2. Save Path Selection
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("Save CSV:"))
        self.save_edit = QLineEdit()
        self.save_edit.setReadOnly(True)
        save_layout.addWidget(self.save_edit)
        self.btn_browse_save = QPushButton("...")
        self.btn_browse_save.setFixedWidth(30)
        self.btn_browse_save.clicked.connect(self._select_save_path)
        save_layout.addWidget(self.btn_browse_save)
        layout.addLayout(save_layout)

        # 3. Start/Stop Button & Show Plot
        action_layout = QHBoxLayout()
        self.btn_toggle_analyzer = QPushButton("Start Analyzer")
        self.btn_toggle_analyzer.setCheckable(True)
        self.btn_toggle_analyzer.clicked.connect(self._toggle_analyzer)
        action_layout.addWidget(self.btn_toggle_analyzer)
        
        self.btn_show_plot = QPushButton("Show Plot")
        action_layout.addWidget(self.btn_show_plot)
        layout.addLayout(action_layout)

        # 4. Status Display
        status_layout = QVBoxLayout()
        self.analyzed_label = QLabel("Analyzed: 0 frames")
        self.speed_label = QLabel("Analysis Speed: 0.0 FPS")
        status_layout.addWidget(self.analyzed_label)
        status_layout.addWidget(self.speed_label)
        layout.addLayout(status_layout)

        # 5. Plot Window & Worker
        self.plot_window = AnalyzerPlotWindow(self)
        self.btn_show_plot.clicked.connect(self.plot_window.show)

        # 6. Start Worker Thread and Connect Signals
        self.analyzer_worker = AnalyzerListeningThread(self.backend)
        self.analyzer_worker.result_ready.connect(self.plot_window.update_plot, Qt.QueuedConnection)
        self.analyzer_worker.result_ready.connect(self._on_analyzer_result_ready, Qt.QueuedConnection)

    def stop(self):
        """Cleanup resources: stop the worker thread and close the plot window."""
        self.status_timer.stop()
        if hasattr(self, 'analyzer_worker'):
            self.analyzer_worker.stop()
        if hasattr(self, 'plot_window'):
            # Close/Hide plot window
            self.plot_window.close()
    
    @Slot(int, 'qint64', object, object, object)
    def _on_analyzer_result_ready(self, frame_id, timestamp, disps, grays, bboxes: dict[int, tuple]):
        if bboxes:
            color = [0.0, 1.0, 0.0, 1.0] # Green bounding box
            # Extract list of bboxes from dict
            bboxes = list(bboxes.values())
            # Preallocate memory, 8 points per bbox, 6f4 per point.
            lines_arr = np.empty((8 * len(bboxes), 6), dtype='f4')

            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                lines_arr[i*8:(i+1)*8, ...] = np.array([
                    [x1, y1, *color], [x2, y1, *color], # Top
                    [x2, y1, *color], [x2, y2, *color], # Right
                    [x2, y2, *color], [x1, y2, *color], # Bottom
                    [x1, y2, *color], [x1, y1, *color]  # Left
                ], dtype='f4')

            self.bboxes_to_draw.emit(lines_arr)
        else:
            self.bboxes_to_draw.emit(None)

    # TODO: 将所有穿透到后端的调用, 在backend里面做转发, 可能可以以analyzer_作为前缀.
    def _select_model_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "PyTorch Models (*.pt)")
        if path:
            self.model_edit.setText(path)
            # Dynamic update via property setter (may raise error if analyzer is running)
            if self.backend.analyzer:
                try:
                    self.backend.analyzer.model_path = path
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to dynamically update model:\n{e}")

    def _select_save_path(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv)")
        if path:
            self.save_edit.setText(path)
            # Dynamic update via property setter
            if self.backend.analyzer:
                try:
                    self.backend.analyzer.save_path = path
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to dynamically update save path:\n{e}")

    def _toggle_analyzer(self, checked):
        if checked:
            model_path = self.model_edit.text()
            save_path = self.save_edit.text()
            
            if not model_path:
                QMessageBox.warning(self, "Validation Error", "Model path is required to start analyzer.")
                self.btn_toggle_analyzer.setChecked(False)
                return
                
            try:
                # 1. Start backend analyzer
                self.backend.start_analyzer(
                    model_path=model_path,
                    save_path=save_path if save_path else None
                )
                # 2. Start result polling worker
                self.analyzer_worker.start()
                self.btn_toggle_analyzer.setText("Stop Analyzer")
                self.btn_toggle_analyzer.setStyleSheet("background-color: #f44336; color: white;") # Red for stop
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start Analyzer:\n{e}")
                self.btn_toggle_analyzer.setChecked(False)
        else:
            # 1. Stop result polling worker
            self.analyzer_worker.stop()
            # 3. Stop backend analyzer
            self.backend.stop_analyzer()
            self.btn_toggle_analyzer.setText("Start Analyzer")
            self.btn_toggle_analyzer.setStyleSheet("")

    def _update_status(self):
        """Poll analyzer status from backend."""
        analyzer = getattr(self.backend, 'analyzer', None)
        if analyzer:
            status = analyzer.status
            frame_count = status.get('frame_count', 0)
            fps = status.get('fps', 0.0)
            self.analyzed_label.setText(f"Analyzed: {frame_count} frames")
            self.speed_label.setText(f"Analysis Speed: {fps:.1f} FPS")
        elif not self.btn_toggle_analyzer.isChecked():
            # Reset labels if not running
            self.analyzed_label.setText("Analyzed: 0 frames")
            self.speed_label.setText("Analysis Speed: 0.0 FPS")
