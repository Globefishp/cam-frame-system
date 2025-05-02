import abc
import multiprocessing as mp
import multiprocessing.shared_memory as mp_shm
import multiprocessing.synchronize as mp_sync
import numpy as np
from typing import Tuple, Any, Optional, List
from shared_ring_buffer import ProcessSafeSharedRingBuffer # Import the ring buffer class

class BaseVideoEncoder(abc.ABC):
    """
    Abstract base class for a video encoder using multiprocessing.
    Handles IPC resource creation, process management, and frame processing from a shared buffer.
    Requires subclasses to implement the encoder initialization and frame encoding logic.
    """
    def __init__(self,
                 shared_buffer: ProcessSafeSharedRingBuffer, # Accept shared buffer instance
                 output_path: str,
                 batch_size: int = 5,
                 **kwargs): # Allow passing extra args to specific encoders
        """
        Initialize the base encoder with a pre-created shared buffer instance.
        The worker process will attach to this buffer in start().
        Args:
            shared_buffer: (ProcessSafeSharedRingBuffer), the shared buffer instance created externally.
            output_path: (str), path to the output video file.
            batch_size: (int), number of frames to get from the buffer at once. Defaults to 5.
            **kwargs: Additional keyword arguments for specific encoder implementations.
        """
        self._output_path = output_path
        self._batch_size = batch_size # Store batch size
        self._ring_buffer = shared_buffer # Store the injected buffer instance for worker to use

        # Multiprocessing resources
        self._running = mp.Event()
        self._worker_process: Optional[mp.Process] = None
        self._worker_ready = mp.Event() # Signals when the worker is ready for the next frame
        # Store kwargs for potential use in subclasses _initialize_encoder
        self._encoder_kwargs = kwargs

    @property
    def is_ready(self) -> bool:
        """Checks if the encoder worker process has signaled it's ready."""
        return self._worker_ready.is_set()

    # Read properties from the injected buffer if needed (optional)
    # Note: Accessing internal attributes like _frame_shape is generally discouraged.
    # It's better if ProcessSafeSharedRingBuffer provides public properties.
    # @property
    # def frame_size(self) -> Tuple[int, int, int]:
    #     if self._ring_buffer and hasattr(self._ring_buffer, '_frame_shape'): # Check attribute existence
    #          return self._ring_buffer._frame_shape
    #     raise ValueError("Shared buffer not initialized or frame shape unavailable.")

    # @property
    # def buffer_capacity(self) -> int:
    #     if self._ring_buffer and hasattr(self._ring_buffer, '_buffer_capacity'): # Check attribute existence
    #          return self._ring_buffer._buffer_capacity
    #     raise ValueError("Shared buffer not initialized or capacity unavailable.")


    @abc.abstractmethod
    def _initialize_encoder(self):
        """
        Initialize the specific video encoder within the worker process.
        This method MUST be implemented by subclasses and is called once
        when the worker process starts. It should handle encoder setup.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _encode_frames(self, frames: List[np.ndarray]):
        """
        Encode a batch of frames using the specific video encoder.
        This method MUST be implemented by subclasses.
        Args:
            frames: (List[np.ndarray]), a list of input frames to encode.
                    the list contains **1 OR 2** np.ndarray objects,
                    each with shape (frame, height, width, channel).
                    The total number of frames should ideally match batch_size,
                    but **MAY BE LESS** if buffer underflow causes a get-timeout.
        
        Notes:
            np.concat(List[np.ndarray], axis=0) will results in a full batch
            of frames with shape (batch_size(ideally), height, width, channel), 
            but will also introduce memory copy. It's recommended to handle the
            list of frames carefully.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _uninitialize_encoder(self):
        """
        Uninitialize the specific video encoder within the worker process.
        This method MUST be implemented by subclasses and is called once
        when the worker process is stopping. It should handle encoder cleanup.
        """
        raise NotImplementedError

    def _worker(self, source_ring_buffer: ProcessSafeSharedRingBuffer): # Accept ring buffer as argument
        """
        The main function executed by the background worker process.
            - Initializes the encoder,
            - Processes frames from the shared ring buffer,
                - Handles buffer underflow gracefully,
                - Encodes the frames,
            - When the process is stopped, it ensures the encoder is uninitialized.
        """
        # Attach to the shared ring buffer in the worker process
        ring_buffer = ProcessSafeSharedRingBuffer(create=False, source_buffer=source_ring_buffer)

        self._worker_ready.clear() # Ensure not set initially
        try:
            # 1. Initialize the specific encoder (once per process)
            print(f"Encoder worker process ({mp.current_process().pid}) initializing...")
            self._initialize_encoder()
            print(f"Encoder worker process ({mp.current_process().pid}) initialized successfully.")
        except Exception as e:
            # Catch any exception during initialization
            print(f"FATAL: Encoder worker process ({mp.current_process().pid}) failed during initialization: {e}")
            ring_buffer.close() # Close ring buffer connection on init failure
            return # Exit worker function on initialization failure

            # 2. Main processing loop
        try:
            while self._running.is_set():
                try:
                    # Signal readiness for the next frame BEFORE waiting
                    self._worker_ready.set()

                    # Get frames from the ring buffer (blocks if no enough data)
                    # Get a batch of frames from the ring buffer
                    frames_list = ring_buffer.get(self._batch_size, timeout=1.0) # Use batch_size

                    # Handle buffer underflow (timeout) here.
                    # Core logic: prevent a buffer which has less than batch_size frames.
                    #             except for 1. the final batch; 2. batch_size is 1.
                    # Usually the batch_size is not set to 1, in order to ensure the analysis thread
                    # can always get the latest frame.  
                    if frames_list is None:
                        # Timeout occurred, check if still running
                        if self._running.is_set():
                            continue # Continue waiting new frames
                        else: # Stop is signaled, no more frames will come in.
                            if ring_buffer.unread_count > 0:
                                # Get all remaining data
                                frames_list = ring_buffer.get(ring_buffer.unread_count, timeout=1.0)
                                if frames_list is None: # Timeout again, raise error
                                    raise TimeoutError("Ring buffer underflows unexpectedly.")
                                else: # Successfully get all remaining frames
                                    # Encode
                                    pass
                            else: # Buffer is empty
                                break # Exit loop

                    # Encode the batch of frames
                    if frames_list: # Only call if frames were received (after handling underflow)
                        # Blocking method, Pass the list of frame views
                        self._encode_frames(frames_list)

                except Exception as e:
                    # Catch errors during frame batch processing, print and continue loop
                    print(f"Error during frame encoding in worker ({mp.current_process().pid}): {e}")
                    # Depending on requirements, might need more sophisticated error handling

            print(f"VideoEncoder worker ({mp.current_process().pid}) exiting loop.")

        finally:
            # Ensure uninitialization and ring buffer closure are attempted
            print(f"Encoder worker process ({mp.current_process().pid}): Attempting to uninitialize encoder and close ring buffer.")
            try:
                self._uninitialize_encoder()
            except Exception as e:
                print(f"Error during encoder uninitialization in worker ({mp.current_process().pid}): {e}")

            ring_buffer.close() # Close ring buffer connection on exit
            print(f"Encoder worker process ({mp.current_process().pid}): Cleanup attempted.")


    def start(self):
        """
        Starts the video encoder.
        Creates the shared ring buffer and launches the background worker process.
        Does nothing if the encoder is already running.
        """
        if self._running.is_set():
            print("Encoder already running.")
            return

        print("Starting VideoEncoder...")
        try:
            # Ensure shared buffer instance was provided during initialization
            if self._ring_buffer is None:
                raise ValueError("Shared buffer instance must be provided during initialization.")
            if not isinstance(self._ring_buffer, ProcessSafeSharedRingBuffer):
                 raise TypeError("Provided shared_buffer is not a ProcessSafeSharedRingBuffer instance.")

            # The main process no longer creates or attaches to the buffer here.
            # The worker process will attach using the provided instance.

            # Set running state and start worker process
            self._running.set()
            print("Starting worker process...")
            # Pass the created ring buffer instance to the worker
            self._worker_process = mp.Process(
                target=self._worker,
                name="VideoEncoderWorker",
                args=(self._ring_buffer,) # Pass the ring buffer instance
            )
            self._worker_process.start()
            print(f"Worker process started with PID: {self._worker_process.pid}")

            # Wait for worker to signal readiness
            print(f"Waiting for encoder worker ({self._worker_process.pid}) to be ready...")
            # Add timeout to worker ready wait
            if not self._worker_ready.wait(timeout=10.0): # e.g., 10 seconds timeout
                 print(f"FATAL: Timeout waiting for encoder worker ({self._worker_process.pid}) to become ready.")
                 self.stop() # Attempt cleanup if worker doesn't start
                 raise TimeoutError("Encoder worker failed to initialize within timeout.")
            print(f"VideoEncoder started successfully with worker PID: {self._worker_process.pid}")

        except Exception as e:
            print(f"Error starting VideoEncoder: {e}")
            # If startup fails, clean up potentially created resources
            self.stop() # Use stop to ensure cleanup
            raise # Re-raise the exception after cleanup attempt


    # submit_frame is removed as CameraSystem now writes directly to the buffer.
    # TODO: Previously, submit_frame will check _worker_ready, but now CameraSystem
    #       should manage this. Need to provide a method(property) to check worker status.


    def stop(self):
        """
        Stops the video encoder.
        Signals the worker process to terminate, waits for it to join,
        and cleans up the created IPC resources (shared ring buffer).
        Does nothing if the encoder is already stopped.
        """
        if not self._running.is_set() and self._worker_process is None:
            print("Encoder already stopped.")
            return

        print("Stopping VideoEncoder...")
        # Signal worker to stop
        self._running.clear()

        # Wait for worker process to finish
        if self._worker_process:
            print(f"Waiting for VideoEncoder worker ({self._worker_process.pid}) to join...")
            self._worker_process.join(timeout=5.0) # Wait with a timeout
            if self._worker_process.is_alive():
                print(f"Warning: Worker process {self._worker_process.pid} did not exit gracefully within timeout. Terminating.")
                self._worker_process.terminate() # Force terminate if join times out
            print(f"Worker process ({self._worker_process.pid}) joined.")
            self._worker_process = None # Clear process reference

        # Main process does not hold a connection to the buffer that needs closing here.
        # The worker process closes its own connection upon exit (handled in _worker's finally block).
        # Unlinking is handled by CameraSystem.
        # if self._ring_buffer:
        #     print("Closing VideoEncoder's connection to Shared Ring Buffer...")
        #     try:
        #         # Main process doesn't need to close the buffer connection itself
        #         # self._ring_buffer.close()
        #         print("VideoEncoder main process does not need to close buffer connection.")
        #     except Exception as e:
        #         print(f"Error related to buffer in VideoEncoder stop: {e}")
        #     finally:
        #          # Keep the reference until the object is destroyed, worker might still need it briefly?
        #          # Or set to None if certain worker is done. Let's clear it.
        #          self._ring_buffer = None # Reset ring buffer reference in main process

        # Reset worker_ready event
        self._worker_ready.clear()

        print("VideoEncoder stopped.")

from x264_encoder import X264Encoder # Import X264Encoder from the new file
