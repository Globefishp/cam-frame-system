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
        # These threading obj is only accessible in the _worker process.
        # and should be initialized in the `_initialize_encoder` method.


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

        x264_cmd.extend(['--output', self._output_path])
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


    def _encode_frames(self, frames: List[np.ndarray]):
        """
        Encodes a batch of frames using the initialized x264 encoder via pipe.
        This runs in the worker process.
        Args:
            frames: (List[np.ndarray[f,h,w,c]]), a list of input frames to encode.
        """
        if not frames:
            return # Nothing to encode

        # Calculate the total number of frames in the batch (for logging/debugging if needed)
        # total_frames_in_batch = sum(arr.shape[0] for arr in frames)
        # if total_frames_in_batch == 0:
        #      print(f"Warning: Received empty frames list or arrays with 0 frames in _encode_frames.")
        #      return

        # Write each array in the list to x264 process stdin
        if self._x264_process and self._x264_process.stdin:
            try:
                for frame_array in frames:
                    # The pipe write is blocking, providing flow control.
                    # This call will block if the pipe buffer is full until x264 reads data.
                    self._x264_process.stdin.write(frame_array.tobytes())
                self._x264_process.stdin.flush()
                # print(f"X264Encoder worker ({mp.current_process().pid}): Wrote {total_frames_in_batch} frames to x264.")

            except BrokenPipeError:
                print(f"Error: x264.exe process pipe is broken. It might have terminated unexpectedly.")
                # The worker loop in the base class will handle exiting if _running is cleared
            except Exception as e:
                print(f"Error writing frame to x264.exe process: {e}")
                # The worker loop in the base class will handle exiting if _running is cleared

        else:
            print("Error: x264.exe process or stdin pipe not available in _encode_frames.")
            # The worker loop in the base class will handle exiting if _running is cleared


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

        # Wait for x264.exe to finish encoding gracefully, with a timeout
        print(f"X264Encoder worker ({mp.current_process().pid}): Waiting for x264.exe to finish encoding...")
        finished_gracefully = self._x264_finished.wait(timeout=3.0) # Wait for up to 3 seconds

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

        print(f"X264Encoder worker ({mp.current_process().pid}): X264 encoder uninitialized.")


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
