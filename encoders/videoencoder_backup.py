import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Any
import numpy as np
import imageio # Using imageio for encoding

class VideoEncoder:
    """
    Encodes video frames in a background process using multiprocessing, using imageio.
    """
    def __init__(self,
                 output_path: str,
                 frame_size: tuple[int, int, int], # (height, width, channels)
                 fps: int,
                 codec: str = 'libx264',
                 codec_options: Optional[dict] = None):
        """
        Initialize the VideoEncoder.

        Args:
            output_path: (str), path to the output video file. imageio primarily writes to files.
            frame_size: (tuple[int, int, int]), expected frame size (height, width, channels).
            fps: (int), frames per second.
            codec: (str), video codec to use (default: 'libx264').
            codec_options: (Optional[dict]), dictionary of codec options for imageio writer.
        """
        self.output_path = output_path
        self.frame_size = frame_size
        self.fps = fps
        self.codec = codec
        # imageio codec options are passed directly to the writer
        self.codec_options = codec_options if codec_options is not None else {}

        # Multiprocessing resources
        # Using SharedMemory and Condition for efficient frame transfer
        self.shm: Optional[SharedMemory] = None
        self.shm_cond: Optional[mp_sync.Condition] = None

        self.worker_process: Optional[mp.Process] = None
        self.working = mp.Event() # Event to signal worker to run/stop

        self.writer: Optional[imageio.get_writer] = None # imageio writer instance

    def _initialize_worker(self):
        """
        Initialize the imageio writer in the worker process.
        """
        print(f"Encoder worker process ({mp.current_process().pid}) initializing...")
        try:
            # Create imageio writer
            # imageio uses ffmpeg as a backend for many video formats
            # The 'codec' parameter maps to ffmpeg's -vcodec
            # The 'fps' parameter sets the frame rate
            # Codec options are passed as keyword arguments
            writer_params = {
                'fps': self.fps,
                'codec': self.codec,
                **self.codec_options # Unpack additional codec options
            }
            print(f"Creating imageio writer with params: {writer_params}")
            self.writer = imageio.get_writer(self.output_path, **writer_params)

            print(f"Encoder worker process ({mp.current_process().pid}) initialized successfully.")
        except Exception as e:
            print(f"FATAL: Encoder worker process ({mp.current_process().pid}) failed to initialize: {e}")
            self.writer = None # Ensure writer is None on failure
            raise # Re-raise the exception to signal initialization failure

    def _encode_frame_data(self, frame_data: np.ndarray):
        """
        Encodes a single frame using the imageio writer.
        This function is called by the worker process.
        """
        if self.writer is None:
            print(f"Error: imageio writer not initialized in worker process ({mp.current_process().pid}). Cannot encode frame.")
            return

        try:
            # Append the frame to the video file
            # imageio expects numpy array in (height, width, channels) format
            # and handles the conversion to the appropriate pixel format for the codec
            # Ensure the frame data is in the correct format (e.g., uint8)
            if frame_data.dtype != np.uint8:
                 print(f"Warning: Frame data type is not uint8, converting. Got {frame_data.dtype}")
                 frame_data = frame_data.astype(np.uint8)

            if frame_data.shape[:2] != self.frame_size[:2]:
                 print(f"Warning: Frame size mismatch. Expected {self.frame_size[:2]}, got {frame_data.shape[:2]}")
                 # Optionally resize or skip
                 return # Skipping for now

            # imageio expects channels last (height, width, channels)
            # If input is (channels, height, width), you might need to transpose
            # Assuming input is already (height, width, channels)
            self.writer.append_data(frame_data)

        except Exception as e:
            print(f"Error during frame encoding in worker ({mp.current_process().pid}): {e}")
            # Continue loop in worker to process next frame


    def worker(self):
        """
        The main function executed by the background encoder process.
        Responsible for data scheduling (getting frames from queue) and calling encode function.
        """
        initialized = False
        try:
            self._initialize_worker()
            initialized = True
        except Exception:
            # Initialization failed, worker cannot proceed
            return

        if not initialized or self.writer is None:
             print(f"Worker process ({mp.current_process().pid}) initialization failed, exiting.")
             return

        print(f"Encoder worker ({mp.current_process().pid}) entering encoding loop.")
        while self.working.is_set():
            try:
                frame_to_encode = None
                # Use shared memory and condition variable
                if self.shm is None or self.shm_cond is None:
                     print("Error: Shared memory or condition variable not initialized in worker.")
                     # This is a fatal error for the worker, break the loop
                     break

                with self.shm_cond:  # Acquire condition lock
                    # Wait for submit_frame notification
                    # Use a timeout to periodically check the working event
                    notified = self.shm_cond.wait(timeout=0.1)
                    if not notified or not self.working.is_set():
                        continue # Timeout or received stop signal

                    # Read frame from shared memory only after notification
                    # Create a copy to avoid issues if shared memory is written to during encoding
                    expected_shape = (self.frame_size[0], self.frame_size[1], self.frame_size[2])
                    expected_dtype = np.uint8
                    frame_to_encode = np.ndarray(
                        expected_shape,
                        dtype=expected_dtype,
                        buffer=self.shm.buf
                    ).copy() # IMPORTANT: Create a copy!

                if frame_to_encode is not None:
                    # Call the separate function to encode the frame
                    self._encode_frame_data(frame_to_encode)

            except Exception as e:
                print(f"Error during frame processing in worker ({mp.current_process().pid}): {e}")
                # Continue loop to process next frame

        # Close the writer
        print(f"Encoder worker ({mp.current_process().pid}) closing writer.")
        try:
            if self.writer:
                self.writer.close()
                print(f"Encoder worker ({mp.current_process().pid}) writer closed.")
        except Exception as e:
            print(f"Error closing writer in worker ({mp.current_process().pid}): {e}")
        finally:
            self.writer = None # Clear writer reference

        print(f"Encoder worker ({mp.current_process().pid}) exiting.")


    def submit_frame(self, frame: np.ndarray):
        """
        Submits a frame for encoding.

        Args:
            frame: (np.ndarray), the frame to submit.
        """
        if not self.working.is_set() or self.worker_process is None or not self.worker_process.is_alive():
            print("Warning: Encoder is not running, cannot submit frame.")
            print("Please call start() before submit_frame().")
            return

        # Use shared memory and condition variable
        if self.shm is None or self.shm_cond is None:
             print("Error: Shared memory or condition variable not initialized.")
             return

        try:
            with self.shm_cond: # Acquire condition lock
                # Ensure frame size and dtype match the shared memory buffer
                expected_shape = (self.frame_size[0], self.frame_size[1], self.frame_size[2])
                expected_dtype = np.uint8 # imageio expects uint8

                if frame.shape != expected_shape or frame.dtype != expected_dtype:
                     print(f"Error: Frame shape/dtype mismatch. Expected {expected_shape} {expected_dtype}, got {frame.shape} {frame.dtype}")
                     return

                # Create a numpy array view of the shared memory buffer
                dest = np.ndarray(expected_shape,
                                  dtype=expected_dtype,
                                  buffer=self.shm.buf)

                # Copy the frame data to shared memory
                np.copyto(dest, frame)

                # Notify the worker process
                self.shm_cond.notify()

        except Exception as e:
            print(f"Error submitting frame to shared memory: {e}")


    def start(self):
        """
        Starts the VideoEncoder background process.
        """
        if self.working.is_set() and self.worker_process is not None and self.worker_process.is_alive():
            print("Encoder already running.")
            return

        print("Starting VideoEncoder...")
        try:
            # Create shared memory and condition variable
            frame_bytes = np.prod(self.frame_size) * np.dtype(np.uint8).itemsize
            # Allocate space for at least one frame
            shared_mem_size = int(frame_bytes)
            print(f"Creating SharedMemory (size: {shared_mem_size} bytes)...")
            self.shm = SharedMemory(create=True, size=shared_mem_size)
            print("Creating Condition...")
            self.shm_cond = mp.Condition()

            self.working.set()
            self.worker_process = mp.Process(target=self.worker, name="VideoEncoderWorker")
            self.worker_process.start()
            print("VideoEncoder started successfully.")

        except Exception as e:
            print(f"Error starting VideoEncoder: {e}")
            self.working.clear()
            if self.worker_process:
                self.worker_process.terminate()
                self.worker_process = None


    def stop(self):
        """
        Stops the VideoEncoder background process.
        """
        if not self.working.is_set() and (self.worker_process is None or not self.worker_process.is_alive()):
            print("Encoder already stopped.")
            return

        print("Stopping VideoEncoder...")
        # Signal the worker to stop
        self.working.clear()
        # Notify the worker to stop if it's waiting on the condition variable
        with self.shm_cond:
            self.shm_cond.notify_all()

        # Wait for the worker process to finish
        if self.worker_process:
            print("Waiting for encoder worker to join...")
            self.worker_process.join(timeout=5.0) # Add timeout
            if self.worker_process.is_alive():
                print(f"Warning: Encoder worker process {self.worker_process.pid} did not exit gracefully. Terminating.")
                self.worker_process.terminate() # Force terminate
            self.worker_process = None # Clear process reference

        # Clean up shared memory resources
        if self.shm:
            print("Cleaning up SharedMemory...")
            try:
                self.shm.close()
                # Only unlink if this is the last process to use it.
                # In this simple example, the main process is the creator, so it unlinks.
                # In a more complex scenario with multiple processes attaching,
                # you might need a different strategy (e.g., a manager or reference counting).
                self.shm.unlink()
                print("SharedMemory cleaned up.")
            except FileNotFoundError:
                 print("SharedMemory already unlinked.")
            except Exception as e:
                print(f"Error cleaning up SharedMemory: {e}")
            finally:
                 self.shm = None # Reset shm reference

        # Reset condition variable reference
        self.shm_cond = None

        print("VideoEncoder stopped.")

# Example Usage (for testing)
if __name__ == '__main__':
    # This part will only run when videoencoder.py is executed directly
    print("Running VideoEncoder example with imageio...")

    # Dummy frame data (e.g., 100 frames of 640x480 RGB)
    frame_height, frame_width, frame_channels = 480, 640, 3
    dummy_frame_size = (frame_height, frame_width, frame_channels)
    dummy_fps = 30
    output_file = "output_imageio.mp4" # Changed output file name

    # Example codec options for libx264 (optional)
    # See ffmpeg documentation for available options
    # codec_opts = {'crf': '23', 'preset': 'medium'}
    codec_opts = {} # Using default options for simplicity

    encoder = VideoEncoder(output_file, dummy_frame_size, dummy_fps, codec_options=codec_opts)

    try:
        encoder.start()

        # Submit dummy frames
        print("Submitting dummy frames...")
        import time # Import time for sleep
        for i in range(100):
            # Create a dummy frame (e.g., random noise)
            dummy_frame = np.random.randint(0, 256, size=dummy_frame_size, dtype=np.uint8)
            # Optional: Add some visual pattern for testing
            dummy_frame[:, :, 0] = i * 2 % 256 # Vary blue channel
            dummy_frame[:, i*5 % frame_width, 1] = 255 # Vertical line in green

            encoder.submit_frame(dummy_frame)
            # time.sleep(1/dummy_fps) # Simulate real-time capture

        print("Finished submitting frames.")

    finally:
        # Stop the encoder and wait for it to finish
        encoder.stop()
        print("VideoEncoder example finished.")
        print(f"Output video saved to {output_file}")
