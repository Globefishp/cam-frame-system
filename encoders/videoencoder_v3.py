# encoders/videoencoder_v3.py
# History:
#  v2   (260407): Using ring buffer v2a, supports functional cross-process 
#                 pickle (not standard pickle, but works for mp) ->
#                 Using loguru to do logging;
#                 Cleanup codes, docs and comments.
#                 Improve stream control logic when stopped. 
#  v3   (260510): Adapt v2 to inject a FrameServer class rather than a ring 
#                 buffer. We intended to use FrameServer to handle 1 producer
#                 multiple consumers scenario. 

# Fails early is good.

from __future__ import annotations # for Synchronized[int]
import inspect
from abc import ABC, abstractmethod
import multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.connection import Connection
import multiprocessing.synchronize as mp_sync
import numpy as np
from numpy.typing import NDArray
import threading as t
from typing import Tuple, Any, Optional, List, Dict, TypeVar, Callable, cast
from functools import wraps
from .videoencoder_types import EncoderException
from frameserver import FrameServer # Import the FrameServer class
import time # Import time for speed calculation and warning throttling

from loguru import logger as file_logger
from loguru._logger import Logger # For Type Hinting Only

F = TypeVar('F', bound=Callable[..., Any])

class BaseVideoEncoder(ABC):
    """
    Abstract base class for a video encoder using multiprocessing.


    Handles IPC resource creation, process management, frame processing and logging
    from a shared buffer.
    Requires subclasses to implement the encoder initialization and frame encoding logic.
    Use `self._logger` in subclass implememted methods if you want.

    Cannot do standard picklization due to internal IPC resources.
    Probably process-safe, but not tested.
    """

    @staticmethod
    def require_stop(func: F) -> F:
        """Decorator to ensure the encoder is stopped before calling the method."""
        @wraps(func)
        def wrapper(self: "BaseVideoEncoder", *args, **kwargs):
            if self._worker_enable.is_set() or self._worker_process is not None:
                cls_name = self.__class__.__name__
                func_name = func.__name__
                raise RuntimeError(f"Operation failed, VideoEncoder is currently running. src_func={cls_name}.{func_name}")
            return func(self, *args, **kwargs)
        return cast(F, wrapper)

    def __init__(self,
                 frame_server: FrameServer,
                 output_path: str, # although not used in IPC, required by most encoder.
                 batch_size: int = 5,
                 target_fps: Optional[float] = None,
                 stat_interval: float = 1.0,
                 extinfo_extractor: Optional[Callable[[NDArray,], tuple[NDArray, List[Dict[str, Any]]]]] = None,
                 inject_logger: Optional[Logger] = None,
                 **kwargs 
                 # Extra kwargs should be handled and store per subclass implementation.
        ):
        """
        Initialize the base encoder with a pre-created shared buffer instance.

        Args:
            frame_server: (FrameServer), the frame server instance to get frame from.
                The VideoEncoder instance will be registered as a strict consumer.
            output_path: (str), path to the output video file.
            batch_size: (int), number of frames to get from the buffer at once. 
                Defaults to 5.
            target_fps: (Optional[float]), target encoding speed in fps. Used in speed
                warning for base class. Subclass may also use it. Defaults to None.
            stat_interval: (float), interval between status updates and warning checks
                in seconds. Defaults to 1.0.
            extinfo_extractor: (Optional[Callable[[NDArray,], tuple[NDArray, List[Dict[str, Any]]]]]), 
                the extended info extractor instance for extracting timecodes from 
                frames. if None, timecodes will not be extracted. Default to None. 
            inject_logger: (Optional[Logger]), the loguru logger instance for logging.
                enqueue=True is required. if None, a default logger is acquired 
                from *current process* (i.e. could differ from the logger instance 
                from `main` if current processs is not `main`).
        """
        if isinstance(frame_server, FrameServer):
            self._frame_server : FrameServer = frame_server
            self._fs_cid: Optional[int] = None # read-only in worker
        else:
            raise TypeError("frame_server must be a FrameServer instance.")
        # TODO: change below code to v3.
        self._output_path: str = output_path
        self._batch_size: int = batch_size
        self._target_fps: Optional[float] = target_fps
        self._stat_interval: float = stat_interval
        self._logger: Logger # For subclass use
        if inject_logger is not None:
            if isinstance(inject_logger, Logger):
                self._logger = inject_logger
            else:
                raise TypeError("inject_logger must be a loguru.Logger instance.")
        else:
            self._logger = file_logger 
        # Base class bind its own friendly_name for logging.
        self.__logger: Logger = self._logger.bind(friendly_name="VideoEncoder")
        self._extinfo_extractor = extinfo_extractor

        # Multiprocessing resources
        # v2: Graded signal for running and exiting in multiprocessing env.
        self._frame_count: Synchronized[int] = mp.Value('I', 0) # Only write in _worker.
        self._worker_enable : mp_sync.Event = mp.Event() # True during normal working loop, False to trigger normal stop.
        self._not_eager_stop : mp_sync.Event = mp.Event() # False to trigger a eager stop.
        self._not_eager_stop.set() # Default True
        self._worker_ready : mp_sync.Event = mp.Event() # Signals when the worker is ready, write in run, read only outside.
        self._worker_process: Optional[mp.Process] = None 

        self._status_lock: t.Lock
        self._status_thread: Optional[t.Thread] = None
        self._status_rx: Optional[Connection] = None
        self._status_tx: Optional[Connection] = None
        
        self._status: dict = {}
        
        self._init_status_ipc()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove resources that cannot be pickled (Lock, Thread, Rx Connection)
        state.pop('_status_lock', None)
        state.pop('_status_thread', None)
        state.pop('_status_rx', None)
        return state

    def _init_status_ipc(self):
        """Initializes or resets the status IPC resources."""
        self._status = {}
        self._status_lock = t.Lock()
        self._status_thread = None # discard history thread ref, let it leak.
        
        self._status_rx, self._status_tx = mp.Pipe(duplex=False)

    def _status_upd_worker(self):
        """Daemon thread loop for receiving status updates from child process."""
        logger = self.__logger
        while self._worker_enable.is_set():
            try:
                if self._status_rx.poll(0.1):
                    msg_type, data = self._status_rx.recv()
                    with self._status_lock:
                        if msg_type == 'update':
                            self._status.update(data)
                        elif msg_type == 'del':
                            for k in data:
                                self._status.pop(k, None)
                        elif msg_type == 'overwrite':
                            self._status.clear()
                            self._status.update(data)
            except (EOFError, BrokenPipeError):
                logger.debug("Status worker exited due to EOF or BrokenPipe.")
                break
            except Exception as e:
                logger.opt(exception=e).error("Error in daemon thread for status dict.")

    def _start_status_ipc(self):
        """Starts the background status fetching thread."""
        self._status_thread = t.Thread(target=self._status_upd_worker, daemon=True, name="EncoderStatusUpd")
        self._status_thread.start()

    def _stop_status_ipc(self):
        """Stops the status thread and closes pipes to reset state."""
        # Close pipes to interrupt any pending reads or writes
        if self._status_rx:
            self._status_rx.close()
            self._status_rx = None
        if self._status_tx:
            self._status_tx.close()
            self._status_tx = None

        if self._status_thread and self._status_thread.is_alive():
            self._status_thread.join(timeout=1.0)
            if self._status_thread.is_alive():
                self.__logger.warning("Status updater thread failed to join.")
                # keep reference for future handling if needed (currently none)
            else:
                self._status_thread = None

    @property
    def status(self) -> dict:
        """Current encoder status, read-only. 
        Will be syncronized between parent and child processes."""
        with self._status_lock:
            return self._status.copy()

    def _status_update(self, data: dict):
        """Subclass specific method to update status."""
        self._status.update(data) # Update local resource.
        self._status_tx.send(('update', data)) # Update remote resource.

    def _status_del(self, keys: list):
        """Subclass specific method to delete key in status dict."""
        for k in keys: self._status.pop(k, None)
        self._status_tx.send(('del', keys))

    def _status_overwrite(self, data: dict):
        """Subclass specific method to overwrite status dict."""
        self._status.clear()
        self._status.update(data)
        self._status_tx.send(('overwrite', data))

    @property
    def is_ready(self) -> bool:
        """Checks if the encoder worker process has signaled it's ready."""
        return self._worker_ready.is_set() and self._worker_enable.is_set()

    @property
    def frame_encoded(self) -> int:
        """Return encoded frame count. Process-safe."""
        return self._frame_count.value

    @property
    def output_path(self) -> str:
        """The output path of the video encoder."""
        return self._output_path

    @output_path.setter
    @require_stop
    def output_path(self, value: str):
        """Set the output path. Only allowed when the encoder is stopped."""
        self._output_path = value

    @abstractmethod
    def _initialize_encoder(self):
        """
        Initialize the specific video encoder within the worker process.
        This method MUST be implemented by subclasses and is called once
        when the worker process starts. It should handle encoder setup.

        Raises:
            EncoderException: when critical error occurs and continue encoding is 
                not possible.
        """
        raise NotImplementedError

    @abstractmethod
    def _encode_frames(self, frames: List[np.ndarray], ext_info: Optional[List[dict]] = None, **kwargs) -> bool:
        """
        Encode a batch of frames using the specific video encoder.
        This method MUST be implemented by subclasses.
        Args:
            frames (List[np.ndarray]): a list of input frames to encode.
                    the list contains **1 OR 2** np.ndarray objects,
                    each with shape (frame, height, width, channel).
                    The total number of frames should ideally match batch_size,
                    but **MAY BE LESS** for the last batch.
            ext_info (Optional[List[dict]]): List of extended information dicts
                extracted by ExtInfoExtractor. None if extractor is not provided.
        Returns:
            bool: True if encoding was successful, False otherwise.

        Raises:
            EncoderException: when critical error occurs and continue encoding is 
                not possible.
        
        Notes:
            The shape of frame is decided by the FrameServer. If `extinfo_extractor`
            is provided, it will also affect the shape of frame. Be compatible 
            with both case.
        """
        raise NotImplementedError

    @abstractmethod
    def _uninitialize_encoder(self):
        """
        Uninitialize the specific video encoder within the worker process.
        This method MUST be implemented by subclasses and is called once
        when the worker process is stopping. It should handle encoder cleanup.

        Raises:
            EncoderException: when critical error occurs and continue encoding is 
                not possible.
        """
        raise NotImplementedError

    def _worker(self, cid: int):
        """
        The main function executed by the background worker process.
            - Initializes the encoder,
            - Get frames from the frame server via cid,
                - Encodes the frames by calling subclass method `_encode_frames`,
                - Update statistics, 
            - When the is signal to stop, it ensures the encoder the remaining 
              frames is encoded and the subclass encoder is uninitialized.
        """
        # --- Init Cross-process Resources ---
        frame_server = self._frame_server

        pid = mp.current_process().pid
        logger = self.__logger

        self._worker_ready.clear() # Ensure not set initially
        try:
            # 1. Initialize the specific encoder
            logger.info(f"Encoder worker initializing...")
            self._initialize_encoder()
            logger.success(f"Encoder worker initialized successfully.")
        except Exception as e:
            logger.opt(exception=e).error(f"Encoder worker failed during initialization.")
            frame_server.close() # Close frameserver connection on init failure
            return # Exit worker process on initialization failure

        # --- Speed Calculation Initialization ---
        if self._target_fps:
            logger.info(f"Expected encoding speed: {self._target_fps} fps. Speed warning enabled.")
        else:
            logger.info(f"Have no `expected_fps`, Encoding speed warning disabled.")
        frame_count_since_last_check = 0
        time_last_check = time.monotonic()
        # --- End Speed Calculation Initialization ---

        # 2. Main processing loop
        try:
            batch_size = self._batch_size
            last_ticket = None
            # Signal readiness for the next frame BEFORE waiting
            self._worker_ready.set()
            while self._not_eager_stop.is_set():
                # v2: Graded stop signal, still process remaining frames until eager_stop.
                try:
                    # --- Get frames from the frame server (blocks if no enough data) ---
                    # Get a batch of frames from the frame server
                    ticket = frame_server.get_sync(cid, batch_size, timeout=0.1)
                    # Release after next ticket is got.
                    if last_ticket is not None: frame_server.release_sync(last_ticket)

                    if ticket is None:
                        # Timeout occurred, check if still running
                        if self._worker_enable.is_set():
                            continue # Continue waiting new frames to collect a batch_size
                        else: # Shutdown signalled, no more frames will come in.
                            if batch_size == 1: 
                                break # No more frame to get, Exit loop
                            # Reduce effective batch_size and retry for stream end.
                            # This design avoid to touch buffer in frameserver
                            batch_size = max(1, batch_size // 2) # directly set to 1 is ok for most case.
                            continue

                    if ticket is not None:
                        frames_list = frame_server.get_from_ticket(ticket, timeout=0.1)
                        
                        # --- Encode the batch of frames ---
                        if frames_list:
                            # Extract extended info and update frames_list with pure image data
                            if self._extinfo_extractor is not None:
                                ext_info_list = []
                                for i, frames in enumerate(frames_list):
                                    frames_list[i], ext_info_block = self._extinfo_extractor(frames)
                                    ext_info_list.extend(ext_info_block)
                            else:
                                ext_info_list = None

                            # Calculate number of frames processed in this batch
                            num_frames_processed = sum(arr.shape[0] for arr in frames_list)
                            if num_frames_processed == 0: # Should not happen
                                continue # Skip if no frames were actually processed
    
                            # Blocking method, Pass the list of frame views
                            ret = self._encode_frames(frames_list, ext_info=ext_info_list)
                            if ret:
                                self._frame_count.value += num_frames_processed
                            else:
                                logger.warning(f"Failed to encode #{num_frames_processed} frames.")
    
                            # Update Speed Calculation 
                            frame_count_since_last_check += num_frames_processed
                            current_time = time.monotonic()
                            time_elapsed = current_time - time_last_check
    
                            # Update status
                            if time_elapsed >= self._stat_interval:
                                encoding_speed = frame_count_since_last_check / time_elapsed
                                self._status_update({"frame_count": self._frame_count.value, "fps": encoding_speed})
                                if self._target_fps is not None:
                                    # Check if speed is noticeably low and issue warning (throttled)
                                    if encoding_speed < self._target_fps * 0.9:
                                        logger.warning(f"VideoEncoder encoding speed ({encoding_speed:.2f} FPS) "
                                                f"is noticeably lower than target FPS ({self._target_fps:.2f}). "
                                                "Frames might be dropped after buffer filled.")
                                if frame_server.buffer.occupied_count_ / frame_server.buffer.buffer_capacity > 0.5:
                                    logger.warning(f"Current buffer load: {frame_server.buffer.occupied_count_} frames, "
                                            f"{frame_server.buffer.occupied_count_ / frame_server.buffer.buffer_capacity * 100:.2f}%")
    
                                # Reset for next measurement interval
                                frame_count_since_last_check = 0
                                time_last_check = current_time
                    # Release is deferred to the next cycle to avoid async consumer starvation.
                    last_ticket = ticket
                except EncoderException as e:
                    logger.opt(exception=e).error(f"[{e.name} ({e.pid})] {e.message}")
                    self._not_eager_stop.clear() # exit immediately
                except Exception as e: # Framework error, should not happen.
                    logger.opt(exception=e).error(f"Unexpected error in encoder working loop.")
                    self._not_eager_stop.clear()

            self._worker_ready.clear()

        finally:
            # Ensure uninitialization and unregister consumer are attempted
            logger.debug(f"Encoder worker exited working loop, cleaning up...")
            try:
                self._uninitialize_encoder()
            except Exception as e:
                logger.opt(exception=e).error(f"Encoder worker failed during uninitialization.")

            if ticket is not None: frame_server.release_sync(ticket)
            if last_ticket is not None: frame_server.release_sync(last_ticket)
            frame_server.close() # Close connection on exit
            logger.success(f"Encoder worker cleanup completed.")
            logger.complete() # Flush logging buffer


    def start(self):
        """
        Launches the background worker process and starts the video encoder.

        Does nothing if the encoder is already running.
        It is not safe to call `start` and `stop` concurrently.
        """
        pid = mp.current_process().pid
        logger = self.__logger

        if self._worker_enable.is_set():
            logger.error("Encoder already running, cannot start twice.")
            return

        logger.info("Starting VideoEncoder...")
        if self._fs_cid is None:
            try:
                self._fs_cid = self._frame_server.register_consumer(historical_data=True)
            except RuntimeError as e:
                logger.error(f"Failed to register consumer in frame server: {e}")
                return
        try:
            # Set running state and start worker process
            self._not_eager_stop.set()
            self._worker_enable.set()
            self._frame_count.value = 0 # Reset counter.
            
            # Start status IPC before worker
            self._init_status_ipc()
            self._start_status_ipc()
            
            logger.debug("Starting worker process...")

            # v3: pass in cid, allow multiple sessions rotate seamlessly.
            wp = mp.Process(target=self._worker, args=(self._fs_cid,))
            self._worker_process = wp
            wp.start()

            # Wait for worker to signal readiness
            logger.debug(f"Worker process started (PID {wp.pid}), waiting to be ready...")

            # Add timeout to worker ready wait
            if not self._worker_ready.wait(timeout=10.0): # e.g., 10 seconds timeout
                 logger.error(f"Timeout waiting for encoder worker ({wp.pid}) to become ready.")
                 self.stop() # Attempt cleanup if worker doesn't start
                 raise TimeoutError("Encoder worker failed to initialize within timeout.")
            logger.success(f"VideoEncoder started successfully with worker PID: {wp.pid}")

        except Exception as e:
            logger.opt(exception=e).error(f"Error in starting VideoEncoder.")
            # If startup fails, clean up potentially created resources
            self.stop() # Use stop to ensure cleanup
            raise # Re-raise the exception after cleanup attempt

    def stop(self, resumable: bool=False, exit_timeout=None, terminate_timeout=20.0):
        """
        Stops the video encoder.
            Signals the worker process to stop, waits for it to join,
            and cleans up the created IPC resources (shared ring buffer).
        Overwrite it if you want to control timeout. Usually terminate_timeout is
            set to a large value.

        Does nothing if the encoder is already stopped.
        It is not safe to call `start` and `stop` concurrently.

        Args:
            resumable (bool): If True, will not release the consumer from frame 
                server, so the encoder can be resumed later seamlessly. 
                Useful for file rotation. To fully release resources, call `stop()`
                with `resumable=False` again. Default False.
            exit_timeout (float): Timeout for the worker process to exit. Default None
                represents auto determined by buffer size and `target_fps`. If not
                available, use 10s.
            terminate_timeout (float): Timeout for the worker process to be terminated.

        Raises:
            TimeoutError: If the worker process does not terminate within the timeout.
        """
        pid = mp.current_process().pid
        logger = self.__logger
        wp = self._worker_process
        buffer = self._frame_server.buffer
        if self._target_fps is not None:
            exit_timeout = buffer.buffer_capacity / self._target_fps * 1.5 
        else:
            exit_timeout = 10
        
        if wp is None and self._fs_cid is None:
            logger.info("VideoEncoder already stopped.")
            return

        if wp is not None:
            logger.info(f"Stopping VideoEncoder worker (PID {wp.pid})...")
            # Signal worker to stop
            self._worker_enable.clear()
            if resumable: 
                # stop immediately without waiting remaining frames being encoded.
                self._not_eager_stop.clear() 

            # Wait for worker process to finish
            if wp.is_alive():
                logger.info(f"Waiting for VideoEncoder worker {wp.pid} to join. "
                            f"Buffer load: {buffer.occupied_count_} frames.")
                # Wait mainly for all buffered frame get encoded, but sometimes waiting _uninit_encoder,
                # depends on the impl. of encoder (whether _encoder_frame is blocking or not)
                wp.join(timeout=exit_timeout) 
                if wp.is_alive():
                    logger.warning(f"Worker process {wp.pid} did not exit within "
                                f"timeout ({exit_timeout}s). Signal eager stop, "
                                f"unprocessed frames may be lost...")
                    self._not_eager_stop.clear() # Eager stop. Turn to _uninit_encoder as soon as possible.
                # Wait mainly for _uninit_encoder to finish, a large timeout.
                wp.join(timeout=terminate_timeout)
                if wp.is_alive():
                    logger.warning(f"Worker process {wp.pid} did not exit within "
                                f"timeout ({terminate_timeout}s). System-level force terminating, "
                                f"data may corrupted...")
                    # TODO: Signal outside to choose behaviour?
                    wp.terminate() # Force terminate if join times out
                wp.join(timeout=1.0) # Wait for the process to actually terminate
                if wp.is_alive():
                    raise TimeoutError("Worker process did not terminate within timeout.")
                logger.info(f"Worker process {wp.pid} joined.")

            self._worker_process = None
            self._stop_status_ipc()
        
        if self._fs_cid is not None and not resumable:
            self._frame_server.unregister_consumer(self._fs_cid)
            self._fs_cid = None

        logger.success(f"VideoEncoder stopped{'.' if self._fs_cid is None else ' without releasing buffer.'}")
