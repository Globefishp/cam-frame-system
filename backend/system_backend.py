from cameras import CamException
import multiprocessing as mp
import time
import numpy as np
from typing import Optional, Tuple, List, Type, Any

from ringbuffers.shared_ring_buffer_v4 import ProcessSafeSharedRingBuffer
from frameserver.v3 import FrameServer, FrameTicket, TicketExpireException
from utils.mp_obj_proxy import MpObjProxy
from cameras.abstractcamera import AbstractCamera, CamException
from encoders.videoencoder_v3 import BaseVideoEncoder
from analyzers.analyzer import BaseAnalyzer

from loguru import logger as file_logger
from loguru._logger import Logger # for type hint only

class CameraProcess(mp.Process):
    """
    Subprocess running the camera capture loop. 
    It receives the proxy and blocks for the ring buffer configuration.
    """
    def __init__(self, camera_proxy: MpObjProxy, ring_buffer: ProcessSafeSharedRingBuffer, 
                 inject_logger: Optional[Logger]):
        super().__init__(daemon=True, name="CameraProcess")
        self.camera_proxy = camera_proxy
        self.ring_buffer = ring_buffer

        if inject_logger is not None:
            if isinstance(inject_logger, Logger):
                self._logger = inject_logger.bind(friendly_name="CameraCaptureProcess")
            else:
                raise TypeError("inject_logger must be a loguru.Logger instance.")
        else:
            self._logger = None
        
        self.grabbed_frames: int = 0

        self.start_event = mp.Event()
        self.stop_event = mp.Event()
        self.exit_event = mp.Event()

    def run(self):
        logger = self._logger
        # Instantiate the target object and start RPC inside subprocess
        bare_camera, rpc_lock = self.camera_proxy()
        self.camera_proxy.start_service_thread()
        # Use bare camera to avoid RPC lock allowing concurrent grab and property get/set (avoid blocking UI).
        # This is usually ok, will not cause race condition (but not documented in AbstractCamera)
        camera = bare_camera 
        
        try:
            camera.open()
        except CamException as e:
            if logger: logger.opt(exception=e).error("Fail to open camera, exiting.")
            camera.close()
            return 
        try:
            while not self.exit_event.is_set():
                # Wait for start signal
                self.start_event.wait(timeout=0.1)
                # check if should exiting
                if self.exit_event.is_set(): break
                if not self.start_event.is_set(): continue
                self.start_event.clear()

                try:
                    camera.start_capture()
                except CamException as e:
                    if logger: logger.opt(exception=e).error("Fail to start capturing, exiting.")
                    break

                # Main capture loop
                if logger: logger.info("Camera starts capturing.")
                while not self.stop_event.is_set():
                    frame = camera.grab_extended_info()
                    if frame is not None:
                        # Note: ring_buffer.put expects shape (batch, H, W, C)
                        # Use a short timeout so we can periodically check stop_event
                        if not self.ring_buffer.put(np.expand_dims(frame, axis=0), timeout=0.1):
                            if logger: logger.warning(f"Timeout putting grabbed frame "
                                f"(#{self.grabbed_frames}), skipping.")
                    else:
                        time.sleep(0.0001)
                    self.grabbed_frames += 1
                camera.stop_capture()
                if logger: logger.info("Camera stops capturing.")
        finally:
            if camera.is_capturing():
                camera.stop_capture()
            if camera.is_opened():
                camera.close()
            if logger: 
                logger.info("Camera is now closed.")
                logger.complete()


class HeadlessBackend: # TODO: Rename as Backend????
    """
    Dependency Injection container and controller for the camera system.
    Manages resources and exposes simple get/release APIs for the GUI consumer.
    """
    def __init__(self, camera_class: Type[AbstractCamera], camera_kwargs: dict, 
                 encoder_class: Type[BaseVideoEncoder], encoder_kwargs: dict, 
                 analyzer_class: Optional[Type[BaseAnalyzer]] = None, analyzer_kwargs: Optional[dict] = None,
                 buffer_capacity: int=120, inject_logger: Optional[Logger] = None):
        if inject_logger is not None:
            if isinstance(inject_logger, Logger):
                self._logger = inject_logger.bind(friendly_name="Backend")
            else:
                raise TypeError("inject_logger must be a loguru.Logger instance.")
        else:
            self._logger = None
        
        self.buffer_capacity = buffer_capacity
        self.camera_class    = camera_class
        self.camera_kwargs   = camera_kwargs # Camera specific kwargs
        self.encoder_class   = encoder_class
        self.encoder_kwargs  = encoder_kwargs # Encoder specific kwargs
        
        self.analyzer_class  = analyzer_class
        self.analyzer_kwargs = analyzer_kwargs or {}
        
        self.ring_buffer: Optional[ProcessSafeSharedRingBuffer] = None
        self.frame_server: Optional[FrameServer] = None
        self.camera_dimension: Tuple[int, int, int] = None
        self.camera_proxy: Optional[MpObjProxy] = None
        self.camera_process: Optional[CameraProcess] = None
        self.encoder: Optional[BaseVideoEncoder] = None
        self.analyzer: Optional[BaseAnalyzer] = None
        
        self.gui_cid: Optional[int] = None
        
        self._init_components()

    def _init_components(self):
        # Try open camera in current process and get camera properties
        logger = self._logger
        if logger: logger.info("Initialize backend components...")

        camera = self.camera_class(**self.camera_kwargs)
        camera.open()
        fw, fh, w, h, c, t = camera.full_width, camera.full_height, camera.width, camera.height, camera.channels, camera.dtype
        camera.close()
        self.buffer_dimension = (fh, fw, c)
        self.camera_dimension = (h, w, c)
        logger.info(f"Got camera dimension  {w} x {h}, {c} channels, {t}")

        # Create the unified RingBuffer and FrameServer
        self.ring_buffer = ProcessSafeSharedRingBuffer(
            create=True,
            buffer_capacity=self.buffer_capacity,
            frame_shape=self.buffer_dimension,
            dtype=t
        )
        self.frame_server = FrameServer(create=True, ring_buffer=self.ring_buffer)
        self.ring_buffer.trigger_release = self.frame_server._gc
        
        # Create capture process with proxy
        self.camera_proxy = MpObjProxy(self.camera_class, **self.camera_kwargs)
        self.camera_process = CameraProcess(self.camera_proxy, self.ring_buffer, self._logger)
        self.camera_process.start()
        self.camera_proxy.wait_handshake()

        if logger: logger.success("Backend components initialized.")
        # (VideoEncoder instantiation is deferred to start_recording)

    def shutdown(self, join_timeout=3.0):
        """Cleanly stops all background workers and releases shared memory."""
        logger = self._logger
        self.stop_capture()
        if self.camera_process:
            self.camera_process.exit_event.set()
            if logger: logger.debug("Signalled exiting camera process.")
            self.camera_process.join(timeout=join_timeout)
            if self.camera_process.is_alive():
                if logger: logger.warning(f"Timeout ({join_timeout}s) waiting camera to stop, force terminating...")
                self.camera_process.terminate()

        self.stop_recording()
        self.stop_analyzer()

        if self.frame_server:
            self.frame_server.close()
            self.frame_server.unlink()
            
        if self.ring_buffer:
            self.ring_buffer.close()
            self.ring_buffer.unlink()

    # ================= GUI Specific Interfaces =================

    def start_capture(self):
        """Starts the capture loop."""
        logger = self._logger
        # Unblock the camera process by providing the ring buffer
        self.camera_process.stop_event.clear()
        self.camera_process.start_event.set()
        if logger: logger.debug("Signalled starting camera capture.")

    def stop_capture(self):
        """Stops the capture loop."""
        logger = self._logger
        self.camera_process.stop_event.set()
        if logger: logger.debug("Signalled stopping camera capture.")
        # FrameServer GC to clear the buffer.
        release_count = self.frame_server._gc()
        read_ptr, write_ptr = self.frame_server.buffer.read_ptr_, self.frame_server.buffer.write_ptr_
        if logger: logger.debug(f"Released {release_count} frames from buffer, new pointer: {read_ptr}, {write_ptr}")

    def start_recording(self, path: str):
        if self.encoder:
            self.stop_recording()
            
        # Dynamically create the encoder instance with the required output path
        try:
            fps = self.camera_proxy.actual_fps
        except AttributeError:
            fps = self.camera_proxy.target_fps
        
        extinfo_extractor = self.camera_proxy.get_extended_info_extractor()
        extinfo_extractor.timebase = self.camera_proxy.hw_timecode_timebase
        extinfo_extractor.timecode_key = "hw_timecode"

        encoder_config = {
            **self.encoder_kwargs,
            'frame_server': self.frame_server,
            'output_path': path,
            'batch_size': 1,
            'target_fps': fps,
            'stat_interval': max(0.5, 10/fps),
            'inject_logger': self._logger,
            'frame_size': self.camera_dimension,
            'extinfo_extractor': extinfo_extractor,
        }
        self.encoder = self.encoder_class(**encoder_config)
        self.encoder.start()

    def stop_recording(self):
        if self.encoder:
            self.camera_process.stop_event.set() # suspend new frame producing
            self.encoder.stop()
            self.camera_process.stop_event.clear()
            self.camera_process.start_event.set()
            self.encoder = None

    def rotate_recording(self, new_path: str):
        """Seamlessly rotate the recording to a new file."""
        if not self.encoder:
            if self._logger: self._logger.error("No active recording to rotate.")
            return
        
        self.encoder.stop(resumable=True)
        self.encoder.output_path = new_path
        self.encoder.start()

    def start_analyzer(self, model_path: Optional[str] = None, save_path: Optional[str] = None):
        """Starts the analyzer background process."""
        if self.analyzer:
            self.stop_analyzer()
            
        if not self.analyzer_class:
            if self._logger: self._logger.error("No analyzer_class injected.")
            return
        
        extinfo_extractor = self.camera_proxy.get_extended_info_extractor()
        extinfo_extractor.timebase = self.camera_proxy.hw_timecode_timebase
        extinfo_extractor.timecode_key = "hw_timecode"

        config = {
            **self.analyzer_kwargs,
            'frame_server': self.frame_server,
            'extinfo_extractor': extinfo_extractor,
        }
        if model_path: config['model_path'] = model_path
        if save_path: config['save_path'] = save_path

        self.analyzer = self.analyzer_class(**config)
        self.analyzer.start()
        if self._logger: self._logger.info("Analyzer started.")

    def stop_analyzer(self):
        """Stops the analyzer."""
        if self.analyzer:
            self.analyzer.stop()
            self.analyzer = None
            if self._logger: self._logger.info("Analyzer stopped.")

    def get_analyzer_results(self) -> dict:
        """Returns a snapshot of the latest accumulated results from the analyzer."""
        if self.analyzer:
            return self.analyzer.get_results()
        return {}
    def get_analyzer_result(self, key: Any, timeout: Optional[float]=None) -> Any:
        """
        Query the key in analyzer's result dict with timeout. 
        Return None if query timeout or no active analyzer.
        """
        if self.analyzer:
            return self.analyzer.get_result(key, timeout)
        return None

    # For preview thread
    def get(self, size: int=1, timeout: Optional[float]=None) -> Tuple[Optional[FrameTicket], Optional[List[np.ndarray]]]:
        """Provides Zero-copy async access to the latest frames."""
        if self.frame_server is None:
            return None, None
        ticket = self.frame_server.get_async(size)
        if ticket is None:
            return None, None
        
        data = self.frame_server.get_from_ticket(ticket, timeout)
        
        return ticket, data

    def set_exposure_time(self, ms: float):
        if self.camera_proxy:
            self.camera_proxy.exposure_time_ms = ms

    def set_fps(self, fps: float):
        if self.camera_proxy:
            self.camera_proxy.target_fps = fps