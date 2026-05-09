import pyqtgraph as pg
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QTabWidget, QPushButton
from PySide6.QtCore import Slot, Qt, QTimer

class AnalyzerPlotWindow(QMainWindow):
    """
    Independent Window for plotting Analyzer metrics using PyQtGraph.
    Supports multiple tiles via tabs.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analyzer Results Plot")
        self.setGeometry(200, 200, 800, 600)
        
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
        self.data_buffers = {} # tile_id -> {"ts": [], "disp": [], "gray": []}
        self.curves = {}       # tile_id -> {"disp": curve, "gray": curve}

        # tab index -> tile id mapping
        self.tab_id_map = [] 
        
        self.start_time = None
        self.max_points = 3600
        
        self._needs_update = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_plots)

    def showEvent(self, event):
        super().showEvent(event)
        self.timer.start(33) # ~30 FPS

    def hideEvent(self, event):
        super().hideEvent(event)
        self.timer.stop()

    def _add_tile_tab(self, tile_id: int):
        """Dynamically create a new tab for a tile."""
        plot_widget = pg.PlotWidget(title=f"Analyzer Metrics - Tile {tile_id}")
        plot_widget.addLegend()
        plot_widget.showGrid(x=True, y=True)
        
        curve_disp = plot_widget.plot(pen='y', name="Displacement")
        curve_gray = plot_widget.plot(pen='c', name="Grayscale")
        
        self.curves[tile_id] = {"disp": curve_disp, "gray": curve_gray}
        self.data_buffers[tile_id] = {"ts": [], "disp": [], "gray": []}
        
        self.tab_widget.addTab(plot_widget, f"Tile {tile_id}")
        self.tab_id_map.append(tile_id)

    @Slot(int, 'qint64', object, object, object)
    def update_plot(self, frame_id, timestamp, disps: dict[int, float], grays: dict[int, float], bboxes: dict[int, tuple]):
        """Update the plot curves for all tiles."""
        if self.start_time is None:
            self.start_time = timestamp
            
        relative_time = (timestamp - self.start_time) / 1e9 # convert ns to seconds
        
        tids = set(disps.keys()) | set(grays.keys())
        for tid in tids:
            disp = disps.get(tid, float('nan'))
            gray = grays.get(tid, float('nan'))
            
            if tid not in self.data_buffers:
                self._add_tile_tab(tid)
            
            buf = self.data_buffers[tid]
            buf["ts"].append(relative_time)
            buf["disp"].append(disp)
            buf["gray"].append(gray)
            
            # Prune old data
            if len(buf["ts"]) > self.max_points:
                buf["ts"] = buf["ts"][-self.max_points:]
                buf["disp"] = buf["disp"][-self.max_points:]
                buf["gray"] = buf["gray"][-self.max_points:]

            self._needs_update = True

    def _on_tab_changed(self, index: int):
        """Sync plot data immediately when switching tabs."""
        if 0 <= index < len(self.tab_id_map):
            tid = self.tab_id_map[index]
            if tid in self.data_buffers:
                buf = self.data_buffers[tid]
                self.curves[tid]["disp"].setData(buf["ts"], buf["disp"])
                self.curves[tid]["gray"].setData(buf["ts"], buf["gray"])

    def _refresh_plots(self):
        """Timer callback: Update the plot curves for the active tile."""
        if not self._needs_update or not self.isVisible():
            return
        
        index = self.tab_widget.currentIndex()
        if 0 <= index < len(self.tab_id_map):
            tid = self.tab_id_map[index]
            if tid in self.data_buffers:
                buf = self.data_buffers[tid]
                self.curves[tid]["disp"].setData(buf["ts"], buf["disp"])
                self.curves[tid]["gray"].setData(buf["ts"], buf["gray"])
            self._needs_update = False

    def reset_plot(self):
        """Reset plot data for all tiles."""
        for tid in self.data_buffers:
            buf = self.data_buffers[tid]
            buf["ts"].clear()
            buf["disp"].clear()
            buf["gray"].clear()
            
            # Clear UI
            self.curves[tid]["disp"].setData([], [])
            self.curves[tid]["gray"].setData([], [])
            
        self.start_time = None

    def closeEvent(self, event):
        """Override close event to hide instead of destroy, preserving state."""
        self.hide()
        event.ignore()
