import time
from PySide6.QtCore import QThread, Signal
from backend.system_backend import HeadlessBackend

class AnalyzerListeningThread(QThread):
    """
    Background worker that polls Analyzer results using blocking get_result().
    Emits data to be consumed by the UI (Plotting and GL overlay).
    """
    # Signals to emit data: frame_id, results
    result_ready = Signal(int, object)
    
    def __init__(self, backend: HeadlessBackend, parent=None):
        super().__init__(parent)
        self.backend = backend
        self._running = False
        self._current_frame_id = 0

    def run(self):
        self._running = True
        self._current_frame_id = 0
        
        while self._running:
            if not self.backend.analyzer:
                # Auto-reset frame_id if analyzer is restarted
                self._current_frame_id = 0
                time.sleep(0.1)
                continue
                
            # Blocks until current_frame_id is available or times out
            res: dict = self.backend.get_analyzer_result(self._current_frame_id, timeout=0.2)
            if res is not None:
                # res is a mixed dict: {"timestamp": ts, 0: {info}, 1: {info}, ...}
                self.result_ready.emit(self._current_frame_id, res.copy())
                self._current_frame_id += 1

        # Force clear UI overlay
        self.result_ready.emit(0, {})

    def stop(self):
        self._running = False
        self.wait()
