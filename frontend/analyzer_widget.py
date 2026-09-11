import os
import numpy as np
from typing import Dict, Any, Optional
from PySide6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QWidget,
                               QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from .analyzer_listening_thread import AnalyzerListeningThread
from .analyzer_plot_window import AnalyzerPlotWindow

class BaseAnalyzerWidget(QGroupBox):
    """
    Generic Base Widget to configure and control an Analyzer.
    Includes configuration for Save Path, and Start/Stop toggle. 
    Also manage listening thread and plot window.
    Subclasses should implement `_init_subclass_ui` to provide custom UI components (like Model Selection).
    """
    # Translate and forward the bbox to lines to draw in the gl widget
    bboxes_to_draw = Signal(object) # object: numpy array or None

    def __init__(self, backend, plot_config: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__("Analyzer Controls", parent)
        self.backend = backend
        self.plot_config = plot_config
        self._init_ui()
        
        # Status polling timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(500)

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Custom UI from Subclass (such as Model Selection)
        custom_ui = self._init_subclass_ui()
        if custom_ui is not None:
            if isinstance(custom_ui, QWidget):
                layout.addWidget(custom_ui)
            elif isinstance(custom_ui, (QVBoxLayout, QHBoxLayout)):
                layout.addLayout(custom_ui)

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
        self.plot_window = AnalyzerPlotWindow(self.plot_config, self)
        self.btn_show_plot.clicked.connect(self.plot_window.show)

        # 6. Start Worker Thread and Connect Signals
        self.analyzer_worker = AnalyzerListeningThread(self.backend)
        # Connect to widget's result handler (for UI updates, plot updates, and GL drawing)
        self.analyzer_worker.result_ready.connect(self._on_analyzer_result_ready, Qt.QueuedConnection)

    def stop(self):
        """Cleanup resources: stop the worker thread and close the plot window."""
        self.status_timer.stop()
        if hasattr(self, 'analyzer_worker'):
            self.analyzer_worker.stop()
        if hasattr(self, 'plot_window'):
            # Close/Hide plot window
            self.plot_window.close()

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

    def _get_backend_init_kwargs(self) -> Optional[Dict[str, Any]]:
        """
        Provide kwargs to start the analyzer.
        Return None indicates the start should be aborted.
        """
        custom_kwargs = self._get_custom_init_kwargs()
        if custom_kwargs is None:
            return None
        
        save_path = self.save_edit.text()
        return {
            "save_path": save_path if save_path else None,
            **custom_kwargs
        }

    def _toggle_analyzer(self, checked):
        if checked:
            kwargs = self._get_backend_init_kwargs()
            if kwargs is None: # Start Aborting.
                self.btn_toggle_analyzer.setChecked(False)
                return
                
            try:
                # 1. Start backend analyzer
                self.backend.start_analyzer(**kwargs)
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

    def _init_subclass_ui(self) -> Optional[QWidget | QVBoxLayout | QHBoxLayout]:
        """Override to inject custom UI at the top. Should return QWidget or QLayout."""
        return None

    @Slot(int, object)
    def _on_analyzer_result_ready(self, frame_id: int, res_dict: Dict[Any, Any]):
        """
        Base implementation handles the result ready signal.
        Forwards the result to the plot window.
        
        Subclasses can override this method to handle the event.
        e.g., Implement GL overlay drawing logic, or custom result 
              preprocessing before passing to PlotWindow, or modify plot config...
        Remember to call super()._on_analyzer_result_ready(...) to ensure
        the plot window receives the data.
        """
        self.plot_window._on_result_ready(frame_id, res_dict)

    def _get_custom_init_kwargs(self) -> Optional[Dict[str, Any]]:
        """
        Hook function where subclasses can add more kwargs (like model_path) 
        for backend component startup.
        Return `None` indicates the start should be aborted.
        Return `{}` for no custom init kwargs.
        """
        return {}
    

class YOLOAnalyzerWidget(BaseAnalyzerWidget):
    """
    Specific Analyzer Widget for YOLO models.
    Provides Model Path selection and BBox drawing over GL.
    """
    def __init__(self, backend, parent=None):
        plot_config = {
            "plot_kwargs": {"title_template": "YOLO Metrics - Tile {tile_id}", "showGrid": {"x": True, "y": True}},
            "curves": [
                {"key": "displacement", "name": "Displacement", "curve_kwargs": {"pen": "y"}},
                {"key": "grayscale", "name": "Grayscale", "curve_kwargs": {"pen": "c"}}
            ]
        }
        super().__init__(backend, plot_config=plot_config, parent=parent)

    def _init_subclass_ui(self):
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_edit = QLineEdit()
        self.model_edit.setReadOnly(True)
        model_layout.addWidget(self.model_edit)
        self.btn_browse_model = QPushButton("...")
        self.btn_browse_model.setFixedWidth(30)
        self.btn_browse_model.clicked.connect(self._select_model_path)
        model_layout.addWidget(self.btn_browse_model)
        return model_layout

    def _select_model_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "PyTorch Models (*.pt)")
        if path:
            self.model_edit.setText(path)
            if self.backend.analyzer:
                try:
                    self.backend.analyzer.model_path = path
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to dynamically update model:\n{e}")

    def _get_custom_init_kwargs(self) -> Optional[Dict[str, Any]]:
        model_path = self.model_edit.text()
        if not model_path:
            QMessageBox.warning(self, "Validation Error", "Model path is required to start YOLO analyzer.")
            return None
        return {"model_path": model_path}

    @Slot(int, object)
    def _on_analyzer_result_ready(self, frame_id: int, res_dict: Dict[Any, Any]):
        """Extract BBox and emit signals for GL Widget Overlay."""
        # Forward data to PlotWindow
        super()._on_analyzer_result_ready(frame_id, res_dict)

        # Draw GL BBox
        res_dict = res_dict.copy()
        res_dict.pop("timestamp", None)
        if not res_dict: # no valid data
            self.bboxes_to_draw.emit(None)
            return

        bboxes = []
        for tid, info in res_dict.items():
            if not isinstance(tid, int) or not isinstance(info, dict):
                continue
            bbox = info.get('bbox')
            if bbox:
                bboxes.append(bbox)

        if bboxes:
            color = [0.0, 1.0, 0.0, 1.0] # Green bounding box
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

class DiffAnalyzerWidget(BaseAnalyzerWidget):
    """
    Specific Analyzer Widget for Diff Analyzer.
    Plots MAD and Px_Changed metrics.
    """
    def __init__(self, backend, parent=None):
        plot_config = {
            "plot_kwargs": {"title_template": "Difference Metrics - Tile {tile_id}", "showGrid": {"x": True, "y": True}},
            "curves": [
                {"key": "mad", "name": "MAD (Mean Abs Diff)", "curve_kwargs": {"pen": "y"}},
                {"key": "px_changed", "name": "Pixels Changed", "curve_kwargs": {"pen": "c"}}
            ]
        }
        super().__init__(backend, plot_config=plot_config, parent=parent)

    def _init_subclass_ui(self):
        # DiffAnalyzer does not need extra UI
        return None
