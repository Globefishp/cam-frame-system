import pyqtgraph as pg
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QTabWidget, QPushButton
from PySide6.QtCore import Slot, Qt

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
        
        self.start_time = None
        self.max_points = 3600

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

    @Slot(int, 'qint64', object, object, object)
    def update_plot(self, frame_id, timestamp, disps, grays, bboxes):
        """Update the plot curves for all tiles."""
        if self.start_time is None:
            self.start_time = timestamp
            
        relative_time = (timestamp - self.start_time) / 1e9 # convert ns to seconds
        
        for tid, (disp, gray) in enumerate(zip(disps, grays)):
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
            
            # Update UI only for the currently visible tab and if window is visible
            if self.isVisible() and self.tab_widget.currentIndex() == tid:
                self.curves[tid]["disp"].setData(buf["ts"], buf["disp"])
                self.curves[tid]["gray"].setData(buf["ts"], buf["gray"])

    def _on_tab_changed(self, index: int):
        """Sync plot data immediately when switching tabs."""
        if index in self.data_buffers:
            buf = self.data_buffers[index]
            self.curves[index]["disp"].setData(buf["ts"], buf["disp"])
            self.curves[index]["gray"].setData(buf["ts"], buf["gray"])

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
