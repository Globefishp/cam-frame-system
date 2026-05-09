import time
from PySide6.QtCore import QThread, Signal
from backend.system_backend import HeadlessBackend

class AnalyzerListeningThread(QThread):
    """
    Background worker that polls Analyzer results using blocking get_result().
    Emits data to be consumed by the UI (Plotting and GL overlay).
    """
    # Signals to emit data: frame_id, timestamp, disps_list, grays_list, bboxes_list
    result_ready = Signal(int, 'qint64', object, object, object)
    
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
            res = self.backend.get_analyzer_result(self._current_frame_id, timeout=0.2)
            if res is not None:
                # res is a mixed dict: {"timestamp": ts, 0: {info}, 1: {info}, ...}
                timestamp = res.get("timestamp", 0)
                
                all_disps = {}
                all_grays = {}
                all_bboxes = {}
                
                # Extract all tile info into dicts to preserve tile ID mapping
                for tid, info in res.items():
                    if not isinstance(tid, int):
                        continue
                    all_disps[tid] = info.get('displacement', float('nan'))
                    all_grays[tid] = info.get('grayscale', float('nan'))
                    if info.get('bbox'):
                        all_bboxes[tid] = info['bbox']
                
                self.result_ready.emit(self._current_frame_id, timestamp, all_disps, all_grays, all_bboxes)
                self._current_frame_id += 1

        # Force clear UI overlay
        self.result_ready.emit(0, 0, {}, {}, {})

    def stop(self):
        self._running = False
        self.wait()
