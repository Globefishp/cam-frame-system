# encoders/videoencoder.py
# History:
#  v2   (260407): Using ring buffer v2a, supports functional cross-process 
#                 pickle (not standard pickle, but works for mp) ->
#                 Using loguru to do logging;
#                 Cleanup codes, docs and comments.
#                 Improve stream control logic when stopped. 

# Fails early is good.

from __future__ import annotations # for Synchronized[int]
import inspect
from abc import ABC, abstractmethod
import multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
import multiprocessing.synchronize as mp_sync
import numpy as np
from typing import Tuple, Any, Optional, List
from .videoencoder_types import EncoderException
from ringbuffers.shared_ring_buffer_v2a import ProcessSafeSharedRingBuffer # Import the ring buffer class
import time # Import time for speed calculation and warning throttling

from loguru import logger as file_logger
from loguru._logger import Logger # For Type Hinting Only

class BaseVideoEncoder(ABC):
    """
    Abstract base class for a video encoder using multiprocessing.
    Can only be started once per instance lifetime. Recreate if restart encoding is needed.

    Handles IPC resource creation, process management, frame processing and logging
    from a shared buffer.
    Requires subclasses to implement the encoder initialization and frame encoding logic.
    Use `self._logger` in subclass implememted methods if you want.

    Cannot do standard picklization due to internal IPC resources.
    Probably process-safe, but not tested.
    """

    def __init__(self,
                 shared_buffer: ProcessSafeSharedRingBuffer,
                 output_path: str, # although not used in IPC, required by most encoder.
                 batch_size: int = 5,
                 expected_fps: Optional[float] = None,
                 inject_logger: Optional[Logger] = None, 
                 # Extra kwargs should be handled and store per subclass implementation.
        ):
        """
        Initialize the base encoder with a pre-created shared buffer instance.

        Args:
            shared_buffer: (ProcessSafeSharedRingBuffer), the shared buffer 
                instance created externally.
            output_path: (str), path to the output video file.
            batch_size: (int), number of frames to get from the buffer at once. 
                Defaults to 5.
            expected_fps: (Optional[float]), expected encoding speed in fps. Used in speed
                warning, NOT related with the encoding procedure. Defaults to None.
            inject_logger: (Optional[Logger]), the loguru logger instance for logging.
                enqueue=True is required. if None, a default logger is acquired 
                from *current process* (i.e. could differ from the logger instance 
                from `main` if current processs is not `main`).
        """
        if isinstance(shared_buffer, ProcessSafeSharedRingBuffer):
            self._ring_buffer : ProcessSafeSharedRingBuffer = shared_buffer
        else:
            raise TypeError("shared_buffer must be a ProcessSafeSharedRingBuffer instance.")
        self._output_path: str = output_path
        self._batch_size: int = batch_size
        self._expected_fps: Optional[float] = expected_fps
        self._logger: Logger
        if inject_logger is not None:
            if isinstance(inject_logger, Logger):
                self._logger = inject_logger
            else:
                raise TypeError("inject_logger must be a loguru.Logger instance.")
        else:
            self._logger = file_logger

        # Multiprocessing resources
        # v2: Graded signal for running and exiting in multiprocessing env.
        self._frame_count: Synchronized[int] = mp.Value('I', 0) # Only write in _worker.
        self._worker_enable : mp_sync.Event = mp.Event() # True during normal working loop, False to trigger normal stop.
        self._not_eager_stop : mp_sync.Event = mp.Event() # False to trigger a eager stop.
        self._not_eager_stop.set() # Default True
        self._worker_ready : mp_sync.Event = mp.Event() # Signals when the worker is ready, write in run, read only outside.
        self._worker_process: Optional[mp.Process] = None 

    @property
    def is_ready(self) -> bool:
        """Checks if the encoder worker process has signaled it's ready."""
        return self._worker_ready.is_set() and self._worker_enable.is_set()

    @property
    def frame_encoded(self) -> int:
        """Return encoded frame count. Process-safe."""
        return self._frame_count.value

    @abstractmethod
    def _initialize_encoder(self):
        """
        Initialize the specific video encoder within the worker process.
        This method MUST be implemented by subclasses and is called once
        when the worker process starts. It should handle encoder setup.

        Raises:
            EncoderException: when critical error occurs and continue encoding is 
                not possible.
        
        Logging: Should have `pid` and `name` fields.

        """
        raise NotImplementedError

    @abstractmethod
    def _encode_frames(self, frames: List[np.ndarray]):
        """
        Encode a batch of frames using the specific video encoder.
        This method MUST be implemented by subclasses.
        Args:
            frames: (List[np.ndarray]), a list of input frames to encode.
                    the list contains **1 OR 2** np.ndarray objects,
                    each with shape (frame, height, width, channel).
                    The total number of frames should ideally match batch_size,
                    but **MAY BE LESS** for the last batch.
        Returns:
            bool: True if encoding was successful, False otherwise.

        Raises:
            EncoderException: when critical error occurs and continue encoding is 
                not possible.
        
        Logging: Should have `pid` and `name` fields.
        
        Notes:
            np.concat(List[np.ndarray], axis=0) will results in a full batch
            of frames with shape (batch_size(ideally), height, width, channel), 
            but will also introduce memory copy. It's recommended to handle the
            list of frames carefully.
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
        
        Logging: Should have `pid` and `name` fields.
        """
        raise NotImplementedError

    def _worker(self):
        """
        The main function executed by the background worker process.
            - Initializes the encoder,
            - Processes frames from the shared ring buffer,
                - Handles buffer underflow gracefully,
                - Encodes the frames,
            - When the process is stopped, it ensures the encoder is uninitialized.
        """
        # --- Init Cross-process Resources ---
        # In v2a, the ring buffer supports functional picklization, no need for attachment anymore.
        ring_buffer = self._ring_buffer

        pid, friendly_name = mp.current_process().pid, "VideoEncoder"
        logger = self._logger.bind(friendly_name=friendly_name)

        self._worker_ready.clear() # Ensure not set initially
        try:
            # 1. Initialize the specific encoder
            logger.info(f"Encoder worker initializing...")
            self._initialize_encoder()
            logger.success(f"Encoder worker initialized successfully.")
        except Exception as e:
            logger.error(f"FATAL: Encoder worker failed during initialization: {e}")
            ring_buffer.close() # Close ring buffer connection on init failure
            return # Exit worker process on initialization failure

        # --- Speed Calculation Initialization ---
        if self._expected_fps:
            logger.info(f"Expected encoding speed: {self._expected_fps} fps. Speed warning enabled.")
        else:
            logger.info(f"Have no `expected_fps`, Encoding speed warning disabled.")
        frame_count_since_last_check = 0
        time_last_check = time.monotonic()
        check_interval = 5.0 # Seconds between warnings
        # --- End Speed Calculation Initialization ---

        # 2. Main processing loop
        try:
            # Signal readiness for the next frame BEFORE waiting
            self._worker_ready.set()
            while self._not_eager_stop.is_set():
                # v2: Graded stop signal, still process remaining frames until eager_stop.
                try:
                    # --- Get frames from the ring buffer (blocks if no enough data) ---
                    # Get a batch of frames from the ring buffer
                    frames_list = ring_buffer.get(self._batch_size, timeout=0.1) # Use batch_size

                    # Handle buffer underflow (timeout) here.
                    # Core logic: prevent a buffer which has less than batch_size frames.
                    #             except for 1. the final batch; 2. batch_size is 1.
                    # Usually the batch_size is not set to 1, in order to ensure the analysis thread
                    # can always get the latest frame.  
                    if frames_list is None:
                        # Timeout occurred, check if still running
                        if self._worker_enable.is_set():
                            continue # Continue waiting new frames to collect a batch_size
                        else: # Shutdown signalled, no more frames will come in.
                            if ring_buffer.unread_count > 0:
                                # v2: Get remaining data one by one, rather than all unread_count
                                frames_list = ring_buffer.get(1, timeout=0.1)
                                if frames_list is None: # Timeout again, raise error
                                    raise TimeoutError("Ring buffer underflows unexpectedly.")
                                else: # Successfully get remaining frame.
                                    pass # To Encode
                            else: # Buffer is empty
                                break # Exit loop

                    # --- Encode the batch of frames ---
                    if frames_list:
                        # Calculate number of frames processed in this batch
                        num_frames_processed = sum(arr.shape[0] for arr in frames_list)
                        if num_frames_processed == 0: # Should not happen
                            continue # Skip if no frames were actually processed

                        # Blocking method, Pass the list of frame views
                        self._encode_frames(frames_list)
                        self._frame_count.value += num_frames_processed

                        # Update Speed Calculation 
                        frame_count_since_last_check += num_frames_processed
                        current_time = time.monotonic()
                        time_elapsed = current_time - time_last_check

                        # Check speed periodically if target_fps specified
                        if self._expected_fps is not None:
                            if time_elapsed >= check_interval:
                                encoding_speed = frame_count_since_last_check / time_elapsed
                                # print(f"Debug: Avg encoding speed over {time_elapsed:.2f}s: {encoding_speed:.2f} FPS") # Optional debug

                                # Check if speed is noticeably low and issue warning (throttled)
                                if encoding_speed < self._expected_fps * 0.9:
                                    logger.warning(f"VideoEncoder encoding speed ({encoding_speed:.2f} FPS) "
                                          f"is noticeably lower than target FPS ({self._expected_fps:.2f}). "
                                          "Frames might be dropped after buffer filled.")
                                    logger.warning(f"Current buffer load: {ring_buffer.unread_count} frames, "
                                          f"{ring_buffer.unread_count / ring_buffer.buffer_capacity * 100:.2f}%")

                            # Reset for next measurement interval
                            frame_count_since_last_check = 0
                            time_last_check = current_time
                except EncoderException as e:
                    logger.error(f"[{e.name} ({e.pid})] {e.message}")
                    self._not_eager_stop.clear() # exit immediately
                except Exception as e: # Framework error, should not happen.
                    logger.error(f"Unexpected error in encoder working loop: {e}")
                    self._not_eager_stop.clear()
            self._worker_ready.clear()
            logger.info(f"Encoder worker exited working loop.")

        finally:
            # Ensure uninitialization and ring buffer closure are attempted
            logger.info(f"Encoder worker uninitializing encoder and closing ring buffer.")
            try:
                self._uninitialize_encoder()
            except Exception as e:
                logger.error(f"During encoder uninitialization in worker: {e}")

            ring_buffer.close() # Close ring buffer connection on exit
            logger.success(f"Encoder worker cleanup completed.")
            logger.complete() # Flush logging buffer


    def start(self):
        """
        Overwrite mp.Process.start(), Can only be called once per instance lifetime.
        Launches the background worker process and starts the video encoder.

        Does nothing if the encoder is already running.
        """
        pid, friendly_name = mp.current_process().pid, "VideoEncoder"
        logger = self._logger.bind(friendly_name=friendly_name)

        if self._worker_enable.is_set():
            logger.error("Encoder already running, cannot start twice.")
            return

        logger.info("Starting VideoEncoder...")
        try:
            # Ensure shared buffer instance was provided during initialization
            if self._ring_buffer is None:
                raise ValueError("Shared buffer instance must be provided during initialization.")
            if not isinstance(self._ring_buffer, ProcessSafeSharedRingBuffer):
                 raise TypeError("Provided shared_buffer is not a ProcessSafeSharedRingBuffer instance.")

            # Set running state and start worker process
            self._worker_enable.set()
            self._frame_count.value = 0 # Reset counter.
            logger.info("Starting worker process...")

            # v2: No arguments need to be passed.
            wp = mp.Process(target=self._worker)
            self._worker_process = wp
            wp.start()

            # Wait for worker to signal readiness
            logger.info(f"Worker process started (PID {wp.pid}), waiting to be ready...")

            # Add timeout to worker ready wait
            if not self._worker_ready.wait(timeout=10.0): # e.g., 10 seconds timeout
                 logger.error(f"Timeout waiting for encoder worker ({wp.pid}) to become ready.")
                 self.stop() # Attempt cleanup if worker doesn't start
                 raise TimeoutError("Encoder worker failed to initialize within timeout.")
            logger.success(f"VideoEncoder started successfully with worker PID: {wp.pid}")

        except Exception as e:
            logger.error(f"Error starting VideoEncoder: {e}")
            # If startup fails, clean up potentially created resources
            self.stop() # Use stop to ensure cleanup
            raise # Re-raise the exception after cleanup attempt

    def stop(self, exit_timeout=5.0, terminate_timeout=20.0):
        """
        Stops the video encoder. Can only be called once per instance lifetime.
            Signals the worker process to stop, waits for it to join,
            and cleans up the created IPC resources (shared ring buffer).
        Overwrite it if you want to control timeout. Usually terminate_timeout is
            set to a large value.

        Does nothing if the encoder is already stopped.

        Args:
            exit_timeout (float): Timeout for the worker process to exit.
            terminate_timeout (float): Timeout for the worker process to be terminated.

        Raises:
            TimeoutError: If the worker process does not terminate within the timeout.
        """
        pid, friendly_name = mp.current_process().pid, "VideoEncoder"
        logger = self._logger.bind(friendly_name=friendly_name)
        wp = self._worker_process
        
        if wp is None:
            logger.info("Encoder already stopped, cannot stop twice.")
            return

        logger.info(f"Stopping VideoEncoder (PID {wp.pid})...")
        # Signal worker to stop
        self._worker_enable.clear()

        # Wait for worker process to finish
        if wp.is_alive():
            logger.info(f"Waiting for VideoEncoder worker ({wp.pid}) to join...")
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
            time.sleep(1.0) # Wait for the process to actually terminate
            if wp.is_alive():
                raise TimeoutError("Worker process did not terminate within timeout.")
            logger.info(f"Worker process ({wp.pid}) joined.")

        logger.success("VideoEncoder stopped.")
