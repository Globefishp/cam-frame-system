import os
import time
import threading
from typing import Optional, Any
from PySide6.QtCore import Signal, QThread

class RecordingThread(QThread):
    """
    Background thread to handle recording lifecycle, timing and file rotation.
    """
    rotation_state_changed = Signal(bool)
    error = Signal(str)

    def __init__(self, backend: Any, base_path: str, 
                interval_sec: Optional[float] = None, 
                file_timestamp: bool = False) -> None:
        """
        Initialize the rotation/recording thread.

        :param backend: The backend system object.
        :param base_path: The template path for video files.
        :param interval_sec: Auto rotation interval in seconds. None for infinite.
        :param file_timestamp: Whether to add a timestamp to the very first file.
        """
        super().__init__()
        self.backend: Any = backend
        self.base_path: str = base_path
        self.interval_sec: Optional[float] = interval_sec
        self.file_timestamp: bool = file_timestamp
        self._stop_event: threading.Event = threading.Event()
        self._manual_trigger: threading.Event = threading.Event()

    def stop(self) -> None:
        """
        Request the thread to stop and stop the active recording.
        Sets the stop event and wakes up the wait loop.
        """
        self._stop_event.set()
        self._manual_trigger.set() # wake up from wait()

    def trigger_rotation(self) -> None:
        """Trigger a manual file rotation by waking up the wait loop."""
        self._manual_trigger.set()

    def run(self) -> None:
        """Main thread loop managing the recording lifecycle."""
        first_loop = True
        try:
            while not self._stop_event.is_set():
                # 1. Determine next file path
                if not self.file_timestamp and first_loop:
                    # Use original path if first file and timestamp is not requested
                    new_path = self.base_path
                else:
                    # Use timestamped path for rotation or if requested for first file
                    base, ext = os.path.splitext(self.base_path)
                    new_path = f"{base}_{int(time.time())}{ext}"
                
                # 2. Start or Rotate via backend
                self.rotation_state_changed.emit(True)
                try:
                    if first_loop:
                        self.backend.start_recording(new_path)
                        first_loop = False
                    else:
                        self.backend.rotate_recording(new_path)
                except Exception as e:
                    cur_time = time.strftime("%H:%M:%S", time.localtime())
                    self.error.emit(f"[{cur_time}] Triggering recording failed: {e}")
                    break # Exit loop and cleanup on critical failure
                finally:
                    self.rotation_state_changed.emit(False)
                
                # 3. Wait for the next rotation event
                # Event.wait() with timeout=None blocks until set
                triggered = self._manual_trigger.wait(timeout=self.interval_sec)
                self._manual_trigger.clear() # Reset for next wait
                
                if self._stop_event.is_set():
                    break
                
                # If triggered by timeout or manual button, loop continues to next rotation
        finally:
            # Ensure the backend actually stops the recording on thread exit
            self.backend.stop_recording()
            self.rotation_state_changed.emit(False)
