import pyqtgraph as pg
from typing import Dict, List, Any, Optional
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QTabWidget, QPushButton
from PySide6.QtCore import Slot, Qt, QTimer
from .utils.dict_filter import rupdate_dict

class AnalyzerPlotWindow(QMainWindow):
    """
    Independent Window for plotting Analyzer metrics using PyQtGraph.
    Supports dynamic configurations and multiple tiles via tabs.
    """
    def __init__(self, plot_config: Optional[Dict[str, Any]] = None, parent=None):
        """
        Args:
            plot_config: Configuration dictionary for plots. 
                e.g.,
                {
                    "plot_kwargs": {"title_template": "Analyzer Metrics - Tile {tile_id}", "showGrid": {"x": True, "y": True}},
                    "curves": [
                        {"key": "displacement", "name": "Displacement", "curve_kwargs": {"pen": "y"}},
                        {"key": "grayscale", "name": "Grayscale", "curve_kwargs": {"pen": "c"}}
                    ]
                }
        """
        super().__init__(parent)
        self.setWindowTitle("Analyzer Results Plot")
        self.setGeometry(200, 200, 800, 600)
        
        self._plot_config = {
            "plot_kwargs": {},
            "curves": []
        }
        self.plot_config = plot_config
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Multi-tab support for different tiles
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        
        # Add a "Clear Results" button to the corner of the tab bar
        self.btn_clear = QPushButton("Clear Results")
        self.btn_clear.clicked.connect(self.reset_plot)
        self.tab_widget.setCornerWidget(self.btn_clear, Qt.TopRightCorner)
        
        layout.addWidget(self.tab_widget)
        
        # Storage: tile_id -> data/ui objects
        self.data_buffers: Dict[int, Dict[str, List[float]]] = {} # tile_id -> {"ts": [], "key1": [], ...}
        self.curves: Dict[int, Dict[str, pg.PlotDataItem]] = {}   # tile_id -> {"key1": curve, ...}

        # tab index -> tile id mapping
        self.tab_id_map = [] 
        
        self.start_time = None
        self.max_points = 3600
        
        self._needs_update = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)

    @property
    def plot_config(self) -> Dict[str, Any]:
        return self._plot_config
        
    @plot_config.setter
    def plot_config(self, value: Optional[Dict[str, Any]]):
        if value:
            # Use rupdate_dict to merge new config into the base structure
            # to ensure keys like "plot_kwargs" and "curves" always exist
            rupdate_dict(self._plot_config, value)

    def showEvent(self, event):
        super().showEvent(event)
        self.timer.start(33) # ~30 FPS

    def hideEvent(self, event):
        super().hideEvent(event)
        self.timer.stop()

    def _add_tile_tab(self, tile_id: int):
        """Dynamically create a new tab for a tile based on plot config."""
        plot_kwargs = self.plot_config["plot_kwargs"]
        title_template = plot_kwargs.get("title_template", "Analyzer Metrics - Tile {tile_id}")
        title = title_template.format(tile_id=tile_id)
        
        plot_widget = pg.PlotWidget(title=title)
        plot_widget.addLegend()
        
        # Apply PyQtGraph PlotWidget kwargs if present
        if "showGrid" in plot_kwargs:
            plot_widget.showGrid(**plot_kwargs["showGrid"])
        else:
            plot_widget.showGrid(x=True, y=True)
            
        if "labels" in plot_kwargs:
            for axis, text in plot_kwargs["labels"].items():
                plot_widget.setLabel(axis, text)
        
        # Apply per curve configs
        self.curves[tile_id] = {}
        self.data_buffers[tile_id] = {"ts": []}
        
        for curve_cfg in self.plot_config["curves"]:
            key = curve_cfg["key"]
            name = curve_cfg.get("name", key)
            curve_kwargs = curve_cfg.get("curve_kwargs", {})
            curve = plot_widget.plot(name=name, **curve_kwargs)
            self.curves[tile_id][key] = curve
            self.data_buffers[tile_id][key] = []
        
        self.tab_widget.addTab(plot_widget, f"Tile {tile_id}")
        self.tab_id_map.append(tile_id)

    @Slot(int, object)
    def _on_result_ready(self, frame_id: int, res_dict: Dict[Any, Any]):
        """Extract data from res_dict and append to buffers."""
        if not res_dict:
            return

        timestamp = res_dict.pop("timestamp", 0)

        if self.start_time is None:
            self.start_time = timestamp
            
        relative_time = (timestamp - self.start_time) / 1e9 # convert ns to seconds
        
        for tid, metrics in res_dict.items():
            if not isinstance(tid, int) or not isinstance(metrics, dict):
                # Skip not recognizable items.
                continue
                
            if tid not in self.data_buffers:
                self._add_tile_tab(tid)
            
            buf = self.data_buffers[tid]
            buf["ts"].append(relative_time)
            
            for curve_cfg in self.plot_config["curves"]:
                key = curve_cfg["key"]
                val = metrics.get(key, float('nan')) # if not available for this timepoint, use nan.
                buf[key].append(val)
            
            # Prune old data
            if len(buf["ts"]) > self.max_points:
                buf["ts"] = buf["ts"][-self.max_points:]
                for curve_cfg in self.plot_config["curves"]:
                    key = curve_cfg["key"]
                    buf[key] = buf[key][-self.max_points:]

            self._needs_update = True

    def _on_tab_changed(self, index: int):
        """Sync plot data immediately when switching tabs."""
        self.update_plot(force=True)

    def update_plot(self, force=False):
        """Pure drawing logic: Update the plot curves for the active tile."""
        if not (self._needs_update or force) or not self.isVisible():
            return
        
        index = self.tab_widget.currentIndex()
        if 0 <= index < len(self.tab_id_map):
            tid = self.tab_id_map[index]
            if tid in self.data_buffers:
                buf = self.data_buffers[tid]
                curves = self.curves[tid]
                
                for key, curve in curves.items():
                    if key in buf:
                        curve.setData(buf["ts"], buf[key])
                        
            self._needs_update = False

    def reset_plot(self):
        """Reset plot data for all tiles."""
        for tid in self.data_buffers:
            buf = self.data_buffers[tid]
            buf["ts"].clear()
            for key in buf:
                if key != "ts":
                    buf[key].clear()
            
            # Clear UI
            curves = self.curves[tid]
            for key, curve in curves.items():
                curve.setData([], [])
            
        self.start_time = None

    def closeEvent(self, event):
        """Override close event to hide instead of destroy, preserving state."""
        self.hide()
        event.ignore()
