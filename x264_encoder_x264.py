import abc
import multiprocessing as mp
import multiprocessing.shared_memory as mp_shm
import multiprocessing.synchronize as mp_sync
import numpy as np
from typing import Tuple, Any, Optional, List
from shared_ring_buffer import ProcessSafeSharedRingBuffer
from videoencoder import BaseVideoEncoder # Import BaseVideoEncoder

# Add imports for subprocess, sys, threading, and os here
import subprocess
import sys
import threading
import os # Import os for path manipulation
# Removed select import as it's not reliable for pipes on Windows
import time # Import time for the example

# Import functions and constants for timecode extraction
from huateng_camera_tc import extract_tc_from_frames, TIMECODE_DTYPE, APPENDED_ROWS_FOR_TIMECODE

# Confirming file state after user feedback
class X264Encoder(BaseVideoEncoder):
    """
    Concrete implementation of BaseVideoEncoder for x264 encoding using x264.exe via pipe.
    """
    def __init__(self,
                 shared_buffer: ProcessSafeSharedRingBuffer,
                 output_path: str,
                 batch_size: int = 5,
                 **kwargs):
        super().__init__(shared_buffer, output_path, batch_size=batch_size, **kwargs)
        # The following attributes should now be accessed from self._encoder_kwargs
        self._fps = kwargs.get('fps', 30)
        self._threads = kwargs.get('threads', 0) # 0: all available threads
        self._preset = kwargs.get('preset', 'fast') # 'ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow', 'placebo'
        self._rc_lookahead = kwargs.get('rc_lookahead', None) 

        self._frame_size = kwargs.get('frame_size') # Assuming frame_size is required and passed in kwargs
        if self._frame_size is None:
             raise ValueError("frame_size must be provided in kwargs")

        self._crf = kwargs.get('crf')
        self._bitrate = kwargs.get('bitrate')
        if self._crf is None and self._bitrate is None:
             print("Warning: Neither crf nor bitrate provided. Using default CRF 23.")
             self._crf = 23 # 设置默认 CRF

        # Attribute to hold the x264 process
        self._x264_process: Optional[subprocess.Popen] = None
        # Attributes to hold reader threads
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        # Attributes to hold x264 finished signal 
        self._x264_finished: Optional[threading.Event] = None
        
        # Timecode related attributes
        self._timecode_extraction_attempt_possible: bool = False # Determined in _initialize_encoder
        self._timecode_timebase = kwargs.get('timebase', None) 
        self._last_timecode_value: Optional[int] = None # For future use: checking monotonicity
        self._timecode_log_path: Optional[str] = None # File path for timecode logging.
        self._timecode_file: Optional[Any] = None # File object for writing timecodes
        self._frame_count_for_timecode: int = 0 # Frame counter for timecode file

        # These threading obj is only accessible in the _worker process.
        # and should be initialized in the `_initialize_encoder` method.

        self._encoder_intermediates: Optional[str] = None # Intermediate file path for x264 encoder.


    def _read_stdout(self):
        """Reads stdout from the x264 process in a separate thread."""
        if self._x264_process and self._x264_process.stdout:
            print(f"X264Encoder worker ({mp.current_process().pid}): Starting stdout reader thread.")
            try:
                # Read line by line while the process is running and encoder is active.
                # Note: Blocking readline is used here.
                # Daemon threads do not respond the exit signal (self._running)
                while self._x264_process.poll() is None:
                    line = self._x264_process.stdout.readline().decode().strip()
                    if line:
                        # Process or log stdout line (e.g., for debugging)
                        # print(f"x264 stdout: {line}")
                        pass # Discard stdout for now
                    # Add a small sleep to prevent tight loop if no output
                    time.sleep(0.01)
            
                # After the process has ended, read any remaining output
                if self._x264_process and self._x264_process.stdout:
                    print(f"X264Encoder worker ({mp.current_process().pid}): x264 process ended, try reading remaining stdout.")
                    for line_bytes in iter(self._x264_process.stdout.readline, b''): # b'' means EOF
                        line = line_bytes.decode().strip()
                        if line:
                            # Process or log stdout line (e.g., for debugging)
                            # print(f"x264 stdout: {line}")
                            pass # Discard stdout for now
                    print(f"X264Encoder worker ({mp.current_process().pid}): Finished reading remaining stdout.")
            
            except Exception as e:
                print(f"Error reading x264 stdout: {e}")
            finally:
                if self._x264_process and self._x264_process.stdout:
                    self._x264_process.stdout.close()
                print(f"X264Encoder worker ({mp.current_process().pid}): Stdout reader thread exiting.")


    def _read_stderr(self):
        """Reads stderr from the x264 process in a separate thread."""
        if self._x264_process and self._x264_process.stderr:
            print(f"X264Encoder worker ({mp.current_process().pid}): Starting stderr reader thread.")
            try:
                # Read line by line while the process is running and encoder is active.
                # Note: Blocking readline is used here.
                # Daemon threads do not respond the exit signal (self._running)
                while self._x264_process.poll() is None:
                    line = self._x264_process.stderr.readline().decode().strip()
                    if line:
                        # Process or log stderr line (e.g., for progress or errors)
                        print(f"x264 stderr: {line}") # Keep printing stderr for now
                        # Check for a message indicating x264 has finished writing the file
                        # Example: "x264 [info]: encoded 150 frames, 29.97 fps, 1000.00 kb/s"
                        if "encoded " in line and " frames, " in line and " kb/s" in line:
                             print(f"X264Encoder worker ({mp.current_process().pid}): Detected x264 completion message.")
                             self._x264_finished.set() # Signal that x264 has finished

                    # Add a small sleep to prevent tight loop if no output
                    time.sleep(0.01)

                # After the process has ended, read any remaining output
                if self._x264_process and self._x264_process.stderr:
                    print(f"X264Encoder worker ({mp.current_process().pid}): x264 process ended, try reading remaining stderr.")
                    for line_bytes in iter(self._x264_process.stderr.readline, b''): # b'' means EOF
                        line = line_bytes.decode().strip()
                        if line:
                            # Process or log stderr line (e.g., for progress or errors)
                            print(f"x264 stderr: {line}") # Keep printing stderr for now
                            # Check for a message indicating x264 has finished writing the file
                            # Example: "x264 [info]: encoded 150 frames, 29.97 fps, 1000.00 kb/s"
                            if "encoded " in line and " frames, " in line and " kb/s" in line:
                                if not self._x264_finished.is_set(): # Avoid re-setting if already set
                                    print(f"X264Encoder worker ({mp.current_process().pid}): Detected x264 completion message in remaining stderr.")
                                    self._x264_finished.set() # Signal that x264 has finished
                    print(f"X264Encoder worker ({mp.current_process().pid}): Finished reading remaining stderr.")

            except Exception as e:
                print(f"Error reading x264 stderr: {e}")
            finally:
                if self._x264_process and self._x264_process.stderr:
                    self._x264_process.stderr.close()
                print(f"X264Encoder worker ({mp.current_process().pid}): Stderr reader thread exiting.")


    def _initialize_encoder(self):
        """
        Initializes the x264 encoder by starting an x264.exe process.
        This runs in the worker process.
        """
        print(f"X264Encoder worker ({mp.current_process().pid}): Initializing x264 encoder (using x264.exe)...")

        # Initialize process-specific attributes (called in `_worker process`)
        self._x264_finished = threading.Event() # Event to signal x264 completion
        self._frame_count_for_timecode = 0 # Reset frame count for each new encoding session
        self._timecode_extraction_attempt_possible = False # Default, will be set based on frame heights

        # --- Frame Height Check & Timecode Detection ---
        # This check is now done here, using self._ring_buffer which is available
        # as BaseVideoEncoder._worker passes it to its own ring_buffer instance.
        # However, self._ring_buffer in the child process is a *new* instance
        # that connects to the *same* shared memory. So, its attributes like
        # frame_shape are correctly reflecting the shared buffer's configuration.
        if self._ring_buffer is None or self._ring_buffer.frame_shape is None:
            raise RuntimeError("X264Encoder: Shared ring buffer or its frame_shape is not available in _initialize_encoder.")
        
        buffer_frame_height = self._ring_buffer.frame_shape[0]
        original_height, original_width, channels = self._frame_size # Original image dimensions

        if buffer_frame_height > original_height:
            self._timecode_extraction_attempt_possible = True
            actual_appended_rows = buffer_frame_height - original_height
            print(f"X264Encoder worker ({mp.current_process().pid}): Buffer frames H ({buffer_frame_height}) > Original H ({original_height}). Timecode extraction will be attempted.")
            if actual_appended_rows != APPENDED_ROWS_FOR_TIMECODE:
                print(f"X264Encoder worker ({mp.current_process().pid}): WARNING - Actual appended rows in buffer ({actual_appended_rows}) "
                      f"does not match expected ({APPENDED_ROWS_FOR_TIMECODE}). Timecode extraction might be unreliable.")
        elif buffer_frame_height == original_height:
            self._timecode_extraction_attempt_possible = False
            print(f"X264Encoder worker ({mp.current_process().pid}): Buffer frames H ({buffer_frame_height}) == Original H ({original_height}). No timecode data expected.")
        else: # buffer_frame_height < original_height
            error_msg = (f"X264Encoder worker ({mp.current_process().pid}): CRITICAL ERROR - Buffer frame height ({buffer_frame_height}) "
                         f"is less than configured original frame height ({original_height}). This indicates a configuration mismatch.")
            print(error_msg)
            raise ValueError(error_msg)
        # --- End Timecode Detection ---

        # Prepare timecode log file path
        base_path, ext = os.path.splitext(self._output_path)
        self._timecode_log_path = base_path + "_timecode.txt"
        self._encoder_intermediates = base_path + "_noTC.mp4"
        

        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the potential path to x264.exe in the same directory
        x264_path_in_dir = os.path.join(script_dir, 'x264.exe')

        # Check if x264.exe exists in the script directory, otherwise rely on PATH
        if os.path.exists(x264_path_in_dir):
            x264_executable = x264_path_in_dir
            print(f"X264Encoder worker ({mp.current_process().pid}): Found x264.exe in script directory: {x264_executable}")
        else:
            x264_executable = 'x264' # Rely on system PATH (Windows will append .exe if needed)
            print(f"X264Encoder worker ({mp.current_process().pid}): x264.exe not found in script directory, relying on system PATH.")

        # Build x264 command line
        height, width, channels = self._frame_size

        # Construct x264 command with parameters
        x264_cmd = [
            x264_executable,
            '--input-res', f'{width}x{height}',
            '--fps', str(self._fps),
            '--input-csp', 'bgr',       # Input color space
            '--demuxer', 'raw',         # Expect raw video frames
            '-',                        # Input from stdin
            '--preset', self._preset,
            '--threads', str(self._threads),
            # '--quiet', # Optional: to reduce console output from x264 if needed
        ]

        # Favour CRF first, then consider using bitrate.
        if self._crf is not None:
            x264_cmd.extend(['--crf', str(self._crf)])
        elif self._bitrate is not None:
            # x264.exe expects bitrate in kbps
            bitrate_kbps = int(self._bitrate) // 1000
            x264_cmd.extend(['--bitrate', str(bitrate_kbps)])
        # If neither is provided, use default CRF 23 (specified in __init__)
        
        if self._rc_lookahead is not None:
            x264_cmd.extend(['--rc-lookahead', str(self._rc_lookahead)])

        x264_cmd.extend(['--output', self._encoder_intermediates])
        # x264.exe typically overwrites by default, no explicit -y needed.

        print(f"X264Encoder worker ({mp.current_process().pid}): Starting x264.exe with command: {' '.join(x264_cmd)}")

        try:
            # Start the x264.exe process with pipes for stdin, stdout, and stderr
            self._x264_process = subprocess.Popen(
                x264_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, # x264 might not output much to stdout when outputting to file
                stderr=subprocess.PIPE  # x264 outputs progress/info to stderr
            )
            print(f"X264Encoder worker ({mp.current_process().pid}): x264.exe process started with PID {self._x264_process.pid}")

            # Start separate threads to continuously read stdout and stderr.
            self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
            self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
            self._stdout_thread.start()
            self._stderr_thread.start()

            # Open timecode log file and write header
            try:
                self._timecode_file = open(self._timecode_log_path, 'w')
                self._timecode_file.write("# timecode format v2\n") # Write header
                print(f"X264Encoder worker ({mp.current_process().pid}): Timecode log file opened at {self._timecode_log_path} and header written.")
            except IOError as e:
                print(f"X264Encoder worker ({mp.current_process().pid}): CRITICAL ERROR - Failed to open timecode log file {self._timecode_log_path}: {e}")
                self._timecode_file = None # Ensure it's None if open fails
                # Encoding will proceed, but timecodes won't be logged.

        except FileNotFoundError:
            # Handle the case where the x264 executable is not found
            print(f"FATAL: x264.exe command not found. Please ensure x264.exe is in your system's PATH, or place 'x264.exe' in the same directory as the script.")
            self._x264_process = None
            raise # Re-raise the exception to indicate initialization failure

        except Exception as e:
            # Handle other potential errors during process startup
            print(f"FATAL: Error starting x264.exe process: {e}")
            self._x264_process = None
            raise # Re-raise the exception to indicate initialization failure


    def _encode_frames(self, frames_list: List[np.ndarray]):
        """
        Encodes a list of frame chunks using the initialized x264 encoder.
        Each chunk in the list is an np.ndarray of frames.
        This method processes each chunk: extracts timecodes (if applicable),
        writes timecodes to a log, and writes image data frame-by-frame to the
        x264 process's stdin.

        Args:
            frames_list (List[np.ndarray]): A list of np.ndarray objects.
                Each np.ndarray has a shape like (num_frames_in_chunk, H_buffer, W, C),
                where H_buffer is the height of frames from the shared buffer (potentially
                including appended timecode data).
        """
        if not frames_list: # Check if the list itself is empty
            print(f"X264Encoder worker ({mp.current_process().pid}): Received empty frames list.")
            return

        if not self._x264_process or not self._x264_process.stdin or self._x264_process.stdin.closed:
            print(f"X264Encoder worker ({mp.current_process().pid}): Error - x264 process or stdin not available/closed.")
            return

        original_height, original_width, channels = self._frame_size

        for frame_chunk_arr in frames_list: # Iterate over each ndarray in the list
            if frame_chunk_arr is None or frame_chunk_arr.size == 0:
                print(f"X264Encoder worker ({mp.current_process().pid}): Received empty or None frame_chunk_arr in list.")
                continue # Skip this chunk

            image_data_chunk_to_encode = None
            extracted_timecodes_for_chunk = None

            if self._timecode_extraction_attempt_possible:
                try:
                    image_data_chunk_to_encode, extracted_timecodes_for_chunk = extract_tc_from_frames(
                        combined_frames=frame_chunk_arr,
                        original_height=original_height,
                        original_width=original_width,
                        channels=channels,
                        timecode_dtype=TIMECODE_DTYPE,
                        expected_appended_rows=APPENDED_ROWS_FOR_TIMECODE
                    )
                except ValueError as ve:
                    print(f"X264Encoder worker ({mp.current_process().pid}): ValueError during TC extraction for chunk: {ve}.")
                    image_data_chunk_to_encode = frame_chunk_arr[:, :original_height, :, :] # Fallback image data
                    extracted_timecodes_for_chunk = None
                except Exception as e_extract:
                    print(f"X264Encoder worker ({mp.current_process().pid}): Unexpected error during TC extraction for chunk: {e_extract}.")
                    image_data_chunk_to_encode = frame_chunk_arr[:, :original_height, :, :] # Fallback image data
                    extracted_timecodes_for_chunk = None
            else: # No TC extraction attempt possible (e.g., buffer_H == original_H)
                  # In this case, frame_chunk_arr should already be (N, original_H, W, C)
                image_data_chunk_to_encode = frame_chunk_arr
                extracted_timecodes_for_chunk = None
            
            if image_data_chunk_to_encode is None: # Should not happen if logic above is correct
                 print(f"X264Encoder worker ({mp.current_process().pid}): CRITICAL - image_data_chunk_to_encode is None. Skipping chunk.")
                 continue

            # --- Write Timecodes to Log File for this chunk ---
            if self._timecode_file:
                num_frames_in_this_chunk = image_data_chunk_to_encode.shape[0]
                for i in range(num_frames_in_this_chunk):
                    current_frame_abs_idx = self._frame_count_for_timecode # For logging message
                    if extracted_timecodes_for_chunk is not None and i < len(extracted_timecodes_for_chunk):
                        current_tc = extracted_timecodes_for_chunk[i]
                        # Rescale tc in timebase to ms(float) for timecode format v2
                        current_tc_ms = current_tc * 1000.0 / self._timecode_timebase
                        self._timecode_file.write(f"{current_tc_ms}\n")
                        
                    # Optional: Monotonicity check
                        if self._last_timecode_value is not None and current_tc < self._last_timecode_value:
                            print(f"X264Encoder worker ({mp.current_process().pid}): Warning - Timecode non-monotonic. Frame {current_frame_abs_idx}. Prev: {self._last_timecode_value}, Curr: {current_tc}")
                        self._last_timecode_value = current_tc
                    else:
                        self._timecode_file.write(f"Missing timecode for frame: {current_frame_abs_idx}\n")
                    self._frame_count_for_timecode += 1 # Increment for each frame processed
            
            # --- Write Image Data for this chunk to x264 stdin (frame by frame) ---
            try:
                for frame_idx_in_chunk in range(image_data_chunk_to_encode.shape[0]):
                    single_frame_view = image_data_chunk_to_encode[frame_idx_in_chunk]
                    # Each single_frame_view from a slice of a C-contiguous array along the first axis
                    # is typically C-contiguous. See `extract_tc_from_frames`.
                    # No np.ascontiguousarray needed here.
                    # Check in case the array is not C-contiguous (should not happen)
                    if not single_frame_view.flags["C_CONTIGUOUS"]:
                        raise ValueError(f"X264Encoder worker ({mp.current_process().pid}): Non-contiguous array encountered.")
                    self._x264_process.stdin.write(single_frame_view.tobytes())
                # Deliberately not flushing after every chunk from the list to batch writes a bit more.
                # Will flush once after all chunks in frames_list are processed.
            except BrokenPipeError:
                print(f"X264Encoder worker ({mp.current_process().pid}): BrokenPipeError while writing frame data for a chunk. Stopping encoder.")
                self._running.clear() # Signal worker to stop
                return # Exit _encode_frames, further chunks in frames_list won't be processed
            except Exception as e:
                print(f"X264Encoder worker ({mp.current_process().pid}): Error writing frame data for a chunk: {e}. Stopping encoder.")
                self._running.clear() # Signal worker to stop
                return # Exit _encode_frames

        # Flush once after all chunks in frames_list have been processed and written
        if self._x264_process and self._x264_process.stdin and not self._x264_process.stdin.closed:
            try:
                self._x264_process.stdin.flush()
            except OSError as e: # Catch errors if stdin is already closed (e.g. BrokenPipe)
                print(f"X264Encoder worker ({mp.current_process().pid}): Error flushing stdin (possibly already closed): {e}")

    def _uninitialize_encoder(self):
        """
        Uninitializes the x264 encoder by stopping the x264.exe process and reader threads.
        This runs in the worker process.
        """
        print(f"X264Encoder worker ({mp.current_process().pid}): Uninitializing x264 encoder (x264.exe)...")

        # Signal x264.exe to finish by closing stdin
        if self._x264_process and self._x264_process.stdin:
            print(f"X264Encoder worker ({mp.current_process().pid}): Closing x264.exe stdin to signal end of stream.")
            try:
                self._x264_process.stdin.close()
            except Exception as e:
                print(f"Error closing x264.exe stdin: {e}")

        # Close the timecode log file if it was opened
        if self._timecode_file:
            print(f"X264Encoder worker ({mp.current_process().pid}): Closing timecode log file.")
            try:
                self._timecode_file.close()
            except Exception as e:
                print(f"X264Encoder worker ({mp.current_process().pid}): Error closing timecode log file: {e}")
            finally:
                self._timecode_file = None

        # Wait for x264.exe to finish encoding gracefully, with a timeout
        print(f"X264Encoder worker ({mp.current_process().pid}): Waiting for x264.exe to finish encoding...")
        finished_gracefully = self._x264_finished.wait(timeout=2.0) # Wait for up to 2 seconds

        if finished_gracefully:
            print(f"X264Encoder worker ({mp.current_process().pid}): x264.exe finished encoding gracefully (detected completion message).")
        else:
            print(f"Warning: x264.exe did not signal completion via stderr message within timeout, or process ended before message.")

        # Wait for a while (optional, x264 should flush on stdin close)
        # time.sleep(0.5) # Give x264 some time to finish writing the file if needed

        # Terminate x264.exe process if it's still running
        if self._x264_process and self._x264_process.poll() is None:
            print(f"X264Encoder worker ({mp.current_process().pid}): Terminating x264.exe process {self._x264_process.pid}...")
            try:
                self._x264_process.terminate()
                self._x264_process.wait(timeout=5.0) # Wait a bit for termination
                print(f"X264Encoder worker ({mp.current_process().pid}): x264.exe process terminated.")
            except subprocess.TimeoutExpired:
                print(f"Warning: x264.exe process {self._x264_process.pid} did not terminate gracefully. Killing.")
                self._x264_process.kill()
            except Exception as e:
                print(f"Error terminating x264.exe process: {e}")
            finally:
                self._x264_process = None # Clear process reference
        elif self._x264_process:
             # Process already exited, just clear reference
             print(f"X264Encoder worker ({mp.current_process().pid}): x264.exe process {self._x264_process.pid} already exited.")
             self._x264_process = None


        # Signal reader threads to stop and join them
        # The threads' loops check self._running, which is cleared by BaseVideoEncoder.stop()
        if self._stdout_thread and self._stdout_thread.is_alive():
            print(f"X264Encoder worker ({mp.current_process().pid}): Waiting for stdout reader thread to join...")
            self._stdout_thread.join(timeout=5.0)
            if self._stdout_thread.is_alive():
                 print("Warning: Stdout reader thread did not exit gracefully.")
            self._stdout_thread = None

        if self._stderr_thread and self._stderr_thread.is_alive():
            print(f"X264Encoder worker ({mp.current_process().pid}): Waiting for stderr reader thread to join...")
            self._stderr_thread.join(timeout=5.0)
            if self._stderr_thread.is_alive():
                 print("Warning: Stderr reader thread did not exit gracefully.")
            self._stderr_thread = None

        self._mux_timecode(tc_file=self._timecode_log_path, mp4_file=self._encoder_intermediates, out_file=self._output_path)

        print(f"X264Encoder worker ({mp.current_process().pid}): X264 encoder uninitialized.")
    
    def _mux_timecode(self, tc_file, mp4_file, out_file) -> bool:
        """
        Muxes the timecode file with the encoded video file using mp4fpsmod.
        """
        print(f"X264Encoder worker ({mp.current_process().pid}): Starting timecode muxing with mp4fpsmod...")
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mp4fpsmod_path_in_dir = os.path.join(script_dir, 'mp4fpsmod.exe')

        if os.path.exists(mp4fpsmod_path_in_dir):
            mp4fpsmod_executable = mp4fpsmod_path_in_dir
            print(f"X264Encoder worker ({mp.current_process().pid}): Found mp4fpsmod.exe in script directory: {mp4fpsmod_executable}")
        else:
            mp4fpsmod_executable = 'mp4fpsmod' # Rely on system PATH (Windows will append .exe if needed)
            print(f"X264Encoder worker ({mp.current_process().pid}): mp4fpsmod.exe not found in script directory, relying on system PATH.")
        try:
            # Run mp4fpsmod to mux the timecode file with the encoded video file
            mp4fpsmod_cmd = [
                mp4fpsmod_executable,
                '-c', # fix non-zero of timecode head
                '-t', tc_file,  # Timecode file path
                '-o', out_file,  # Output file path
                mp4_file,  # Input video file path
            ]
            print(f"X264Encoder worker ({mp.current_process().pid}): Running mp4fpsmod with command: {' '.join(mp4fpsmod_cmd)}")
            subprocess.run(mp4fpsmod_cmd, check=True)
            print(f"X264Encoder worker ({mp.current_process().pid}): Timecode muxing completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"X264Encoder worker ({mp.current_process().pid}): Error running mp4fpsmod: {e}")
            print(f"X264Encoder worker ({mp.current_process().pid}): mp4fpsmod stderr: {e.stderr.decode()}")
            # raise  # Re-raise the exception to indicate muxing failure
            return False
        except FileNotFoundError:
            # Handle the case where the mp4fpsmod executable is not found
            print(f"FATAL: mp4fpsmod command not found. Please ensure mp4fpsmod.exe is in your system's PATH, or place 'mp4fpsmod.exe' in the same directory as the script.")
            # raise  # Re-raise the exception to indicate initialization failure
            return False
        except Exception as e:
            # Handle other potential errors during muxing
            print(f"FATAL: Error running mp4fpsmod: {e}")
            # raise  # Re-raise the exception to indicate muxing failure
            return False
        return True


if __name__ == "__main__":
    output_file = "test_x264_encoder.mp4"
    frame_height = 480
    frame_width = 640
    frame_channels = 3 # Assuming BGR or RGB
    fps = 30
    duration_seconds = 5
    total_frames = fps * duration_seconds
    batch_size = 10 # Example batch size
    buffer_capacity = batch_size * 2 # Example buffer capacity

    # 1. 创建 ProcessSafeSharedRingBuffer 实例
    print("Creating shared ring buffer...")
    shared_buffer = None
    try:
        shared_buffer = ProcessSafeSharedRingBuffer(
            create=True,
            buffer_capacity=buffer_capacity,
            frame_shape=(frame_height, frame_width, frame_channels),
            dtype=np.uint8
        )
        print(f"Shared ring buffer created: Metadata SHM: {shared_buffer.metadata_name}, Data SHM: {shared_buffer.data_name}")

        # 2. 创建 X264Encoder 实例，并传入共享缓冲区
        print("Creating X264Encoder instance...")

        # 构建一个字典来存放非必须的编码参数
        encoding_kwargs = {
            'frame_size': (frame_height, frame_width, frame_channels),
            'fps': fps,
            'camera_fps': fps, # Pass camera_fps for speed calculation in BaseVideoEncoder
            'crf': 23, # 添加 crf 参数来控制编码质量
            'preset': 'medium' # 添加 preset 参数
            # 可以在这里添加更多编码参数，例如 'bitrate', 'threads' 等
        }

        encoder = X264Encoder(
            shared_buffer=shared_buffer, # Pass the shared buffer instance
            output_path=output_file,
            batch_size=batch_size, # batch_size is a required parameter
            **encoding_kwargs # 使用 ** 解包字典，将键值对作为关键字参数传入
        )
        print("X264Encoder instance created.")

        # 3. 启动编码器工作进程
        print("Starting encoder...")
        encoder.start()
        print("Encoder started.")

        # 4. 生成帧并放入共享缓冲区
        print(f"Generating and putting {total_frames} frames into the shared buffer...")
        for i in range(total_frames):
            # 生成一个简单的黑色帧 (height, width, channels)
            frame = np.zeros((frame_height, frame_width, frame_channels), dtype=np.uint8)

            # 可选：在帧上绘制一些简单的内容以验证编码
            # 例如，绘制一个随时间移动的白色方块
            square_size = 50
            x = (i * 5) % (frame_width - square_size)
            y = (i * 3) % (frame_height - square_size)
            frame[y:y+square_size, x:x+square_size, :] = 255 # White square

            # 将帧放入共享缓冲区
            # put 方法期望一个 shape 为 (frame_num, h, w, c) 的 numpy 数组
            if not shared_buffer.put(np.expand_dims(frame, axis=0), timeout=1.0):
                 print(f"Warning: Timeout putting frame {i+1} into buffer.")
                 # Depending on test requirements, might break here or continue

            # 模拟实时帧率
            time.sleep(1.0 / fps)

        print(f"Finished putting frames into the buffer. Total frames: {i+1}")

    except Exception as e:
        print(f"An error occurred during the test: {e}")

    finally:
        # 5. 停止编码器
        print("Stopping encoder...")
        if 'encoder' in locals() and encoder:
            encoder.stop()
            # The encoder should process all remaining frames after stop and then exit.
            print(f"Encoder stopped. Output saved to {output_file}")
        else:
            print("Encoder instance was not created due to an error.")
        
        print(f"There are {shared_buffer.unread_count} frames left in the buffer.")

        # 6. 关闭并解除链接共享缓冲区
        print("Cleaning up shared memory...")
        if shared_buffer:
            try:
                 shared_buffer.close()
            except Exception as e:
                 print(f"Error closing buffer in main: {e}")
            try:
                 shared_buffer.unlink()
            except Exception as e:
                 print(f"Error unlinking buffer in main: {e}")
        else:
            print("Shared buffer instance was not created due to an error.")

    print("Test finished.")

# 获取总帧数：ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 test_x264_encoder.mp4

if __name__ == "__main__":
    import cProfile
    import pstats
    import io
    import os
    import numpy as np # Ensure numpy is imported here
    # time is already imported at the top level

    # --- Configuration for Profiling ---
    output_file = "profiled_test_x264_encoder.mp4"
    frame_height = 480
    frame_width = 640
    frame_channels = 3
    fps = 30
    
    # Encoder initialization parameters
    encoder_init_batch_size = 1 

    # Frame generation for profiling _encode_frames
    num_frame_batches_in_list = 20 # How many np.ndarray batches in the list for _encode_frames
    frames_per_ndarray_batch = 50 # How many actual frames in each np.ndarray
    total_frames_to_encode = num_frame_batches_in_list * frames_per_ndarray_batch

    # --- Mock Shared Buffer ---
    class MockSharedRingBuffer:
        """A mock object for ProcessSafeSharedRingBuffer for isolated testing."""
        def __init__(self):
            pass
        @property
        def frame_shape(self):
            return (frame_height, frame_width, frame_channels)

    mock_shared_buffer = MockSharedRingBuffer()

    print("Creating X264Encoder instance for profiling...")
    encoding_kwargs = {
        'frame_size': (frame_height, frame_width, frame_channels),
        'fps': fps,
        'camera_fps': fps, # For BaseVideoEncoder, not critical for _encode_frames profiling
        'crf': 23,
        'preset': 'medium', 
        'threads': 0 
    }

    # Ensure the X264Encoder class is defined above this block
    encoder = X264Encoder(
        shared_buffer=mock_shared_buffer,
        output_path=output_file,
        batch_size=encoder_init_batch_size,
        **encoding_kwargs
    )
    print("X264Encoder instance created.")

    try:
        print("Initializing encoder...")
        encoder._initialize_encoder() # Sets up self._container and self._stream

        print("Encoder initialized.")

        print(f"Preparing {total_frames_to_encode} sample frames ({num_frame_batches_in_list} batches of {frames_per_ndarray_batch} frames)...")
        sample_input_frames = []
        for _ in range(num_frame_batches_in_list):
            frame_batch_data = np.random.randint(
                0, 255,
                size=(frames_per_ndarray_batch, frame_height, frame_width, frame_channels),
                dtype=np.uint8
            )
            sample_input_frames.append(frame_batch_data)
        print("Sample frames prepared.")

        print(f"Profiling _encode_frames with {total_frames_to_encode} frames...")
        profiler = cProfile.Profile()
        profiler.enable()

        encoder._encode_frames(sample_input_frames)

        profiler.disable()
        print("_encode_frames execution finished.")

        print("\n--- cProfile Stats (sorted by cumulative time) ---")
        # Sort options: 'calls', 'cumulative', 'filename', 'nfl', 'pcalls', 'line', 'name', 'stdname', 'time', 'tottime'
        stats_stream = io.StringIO()
        profile_stats = pstats.Stats(profiler, stream=stats_stream).sort_stats('cumulative')
        profile_stats.print_stats(20) 
        # profile_stats.print_callers()
        print(stats_stream.getvalue())
        stats_stream.close()

        # To save to a file: 
        # # profile_stats.dump_stats("encode_frames_profile.prof") 
        # # Then view with snakeviz: snakeviz encode_frames_profile.prof
        
        print("Uninitializing encoder (flushing and closing)...")
        encoder._uninitialize_encoder()
        print("Encoder uninitialized.")

    except Exception as e:
        print(f"FATAL ERROR during profiling: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(output_file):
            try:
                # os.remove(output_file)
                print(f"Cleaned up temporary output file: {output_file}")
            except Exception as e_del:
                print(f"Error deleting temporary file {output_file}: {e_del}")

    print("Profiling test finished.")
