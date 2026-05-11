# Requirements:
#   In the same folder:
#       x264.exe (recommended: https://github.com/Patman86/x264-Mod-by-Patman)
#       mp4fpsmod.exe (from https://github.com/nu774/mp4fpsmod)
# History:
# v2   (260407): Refactored upon the update of AbstractCamera and BaseVideoEncoder 
#                API. Now we can purely relys on abstract classes. 
#                Cleanup by Haiyun Huang

import multiprocessing as mp
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Any, Optional, List, Dict, Callable, Protocol
from ringbuffers.shared_ring_buffer_v4 import ProcessSafeSharedRingBuffer
from frameserver.v3.frameserver_v3 import FrameServer
from encoders.videoencoder_v3 import BaseVideoEncoder
from encoders.videoencoder_types import EncoderException
from utils.digits_overlay import FastDigitsOverlay

import subprocess
import threading
import os
from pathlib import Path

import time # Import time for the example

from loguru import logger as file_logger
from loguru._logger import Logger # For Type Hinting Only

_ENCODER_CURR_DIR = Path(__file__).resolve().parent

class TimecodeExtractor(Protocol):
    def __call__(self, image: NDArray, **kwargs: Any) -> tuple[NDArray, List[Dict[str, Any]]]:
        pass
    @property
    def timebase(self) -> int:
        """1/timebase equals time per tick in second."""
        pass
    @property
    def timecode_key(self) -> str:
        """The key to extract timecode from the extended info dict."""
        pass
    # TODO: Wrapper for ExtInfoExtractor, done in Backend (top level that responsible for DI)

class X264Encoder(BaseVideoEncoder):
    """
    Concrete implementation of BaseVideoEncoder for x264 encoding using x264.exe via pipe.
    """
    def __init__(self,
                 frame_server: FrameServer,
                 output_path: str,
                 batch_size: int = 5,
                 target_fps: Optional[float] = None,
                 stat_interval: float = 1.0,
                 extinfo_extractor: Optional[TimecodeExtractor] = None,
                 inject_logger: Logger = None,
                 mux_timecode: bool = False,
                 **kwargs
        ):
        """
        Args: 
            frame_server: (FrameServer), the frame server instance created externally.
            output_path: (str), path to the output video file.
            batch_size: (int), number of frames to get from the buffer at once. 
                Defaults to 5.
            stat_interval: (float), the interval for reporting statistics. 
                Defaults to 1.0 second. 
            extinfo_extractor: (Optional[TimecodeExtractor]), the timecode extractor 
                instance for extracting timecodes from frames. if None, timecodes will
                not be extracted. Default to None. 
            inject_logger: (Optional[Logger]), the loguru logger instance for logging.
                enqueue=True is required. if None, a default logger acquired 
                from current process will be used and could differ from the 
                logger instance from `main` if current processs is not `main`.
            mux_timecode: (bool), whether to mux the timecodes into the output video. 
                Defaults to False. If True, uninit will blocking for long time. 
            kwargs: 
                frame_size: (Tuple[int, int, int]), the frame size of the input frames.
                crf: (int), the constant rate factor for x264 encoding.
                bitrate: (int), the bitrate for x264 encoding. Should have at least one of crf or bitrate.
                fps: (int), the fps metadata of the output video. Defaults same as `target_fps`.
                threads: (int), the number of threads to use for x264 encoding. Defaults to 0 (all available threads).
                preset: (str), the preset for x264 encoding. Defaults to 'fast'.
                rc_lookahead: (int), the rc_lookahead for x264 encoding. Defaults to None.
                timebase: (int), the timebase for x264 encoding. Defaults to None.
            Any additional kwargs will be passed to x264 cmd line as `--str(key) str(value)`.
                input-csp: (str), the color space of the input frames. Defaults to 'bgr'.
                input-depth: (str), the depth of the input frames. Defaults to '8'.
        """
        pid, friendly_name = mp.current_process().pid, "X264Encoder"
        fps = kwargs.get('fps', target_fps)
        super().__init__(frame_server, output_path, batch_size=batch_size, 
            target_fps=fps, stat_interval=stat_interval, extinfo_extractor=extinfo_extractor, 
            inject_logger=inject_logger)
        self._logger = self._logger.bind(friendly_name=friendly_name)

        # Parse kwargs
        self._frame_size = kwargs.pop('frame_size') # the size of pure image data part for x264
        if self._frame_size is None:
             raise ValueError("frame_size must be provided in kwargs")

        self._crf = kwargs.pop('crf', None)
        self._bitrate = kwargs.pop('bitrate', None)
        if self._crf is None and self._bitrate is None:
             logger.warning(f"Neither crf nor bitrate provided. Using default CRF 23.")
             self._crf = 23

        self._fps = kwargs.pop('fps', 30)
        self._threads = kwargs.pop('threads', 0) # 0: all available threads
        self._preset = kwargs.pop('preset', 'fast') # 'ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow', 'placebo'
        self._rc_lookahead = kwargs.pop('rc_lookahead', None) 
        # Other kwargs is passed to x264 directly
        self._encoder_kwargs = kwargs

        self._x264_path: Path = (_ENCODER_CURR_DIR / ".." / "tools" / "x264.exe").resolve()
        # These threading obj is only accessible in the _worker process.
        # and should be initialized in the `_initialize_encoder` method.
        self._x264_process: Optional[subprocess.Popen] = None
        # console reader threads
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        # x264 finished signal 
        self._x264_finished: Optional[threading.Event] = None

        self._encoder_intermediates: Optional[str] = None # Intermediate file path for x264 encoder.
        
        # Timecode related attributes
        self._timecode_extractor: TimecodeExtractor = extinfo_extractor
        self._do_mux_timecode: bool = mux_timecode
        self._last_timecode_value: Optional[int] = None # For future use: checking monotonicity
        self._timecode_log_path: Optional[str] = None # File path for timecode logging.
        self._timecode_file: Optional[Any] = None # File object for writing timecodes
        self._processed_frames: Optional[int] = None

    def _read_stdout(self):
        """Reads stdout from the x264 process in a separate thread."""
        logger = self._logger

        process: subprocess.Popen = self._x264_process

        if process and process.stdout:
            logger.info(f"Starting stdout reader thread.")
            try:
                # Read line by line while the process is running and encoder is active.
                # Note: Blocking readline is used here.
                # Daemon threads do not respond the exit signal (self._running)
                while process.poll() is None:
                    line = process.stdout.readline().decode().strip()
                    if line:
                        # Process or log stdout line (e.g., for debugging)
                        # print(f"x264 stdout: {line}")
                        pass # Discard stdout for now
                    # Add a small sleep to prevent tight loop if no output
                    time.sleep(0.01)
            
                # After the process has ended, read any remaining output
                if process and process.stdout:
                    logger.info(f"x264 process ended, try reading remaining stdout.")
                    for line_bytes in iter(process.stdout.readline, b''): # b'' means EOF
                        line = line_bytes.decode().strip()
                        if line:
                            # Process or log stdout line (e.g., for debugging)
                            # print(f"x264 stdout: {line}")
                            pass # Discard stdout for now
                    logger.info(f"Finished reading remaining stdout.")
            
            except Exception as e:
                logger.opt(exception=e).error(f"Error in reading x264 stdout.")
            finally:
                if process and process.stdout:
                    process.stdout.close()
                logger.info(f"Stdout reader thread exiting.")


    def _read_stderr(self):
        """Reads stderr from the x264 process in a separate thread."""
        logger = self._logger

        process: subprocess.Popen = self._x264_process
        
        if process and process.stderr:
            logger.info(f"Starting stderr reader thread.")
            try:
                # Read line by line while the process is running and encoder is active.
                # Note: Blocking readline is used here.
                # Daemon threads do not respond the exit signal (self._running)
                while process.poll() is None:
                    line = process.stderr.readline().decode().strip()
                    if line:
                        # Process or log stderr line (e.g., for progress or errors)
                        logger.debug(f"{line}") # Keep printing stderr for now
                        # Check for a message indicating x264 has finished writing the file
                        # Example: "x264 [info]: encoded 150 frames, 29.97 fps, 1000.00 kb/s"
                        if "encoded " in line and " frames, " in line and " kb/s" in line:
                             logger.info(f"Detected x264 completion message.")
                             self._x264_finished.set() # Signal that x264 has finished

                    # Add a small sleep to prevent tight loop if no output
                    time.sleep(0.01)

                # After the process has ended, read any remaining output
                if process and process.stderr:
                    logger.info(f"X264Encoder: x264 process ended, try reading remaining stderr.")
                    for line_bytes in iter(process.stderr.readline, b''): # b'' means EOF
                        line = line_bytes.decode().strip()
                        if line:
                            # Process or log stderr line (e.g., for progress or errors)
                            logger.debug(f"{line}") # Keep printing stderr for now
                            # Check for a message indicating x264 has finished writing the file
                            # Example: "x264 [info]: encoded 150 frames, 29.97 fps, 1000.00 kb/s"
                            if "encoded " in line and " frames, " in line and " kb/s" in line:
                                if not self._x264_finished.is_set(): # Avoid re-setting if already set
                                    logger.info(f"Detected x264 completion message in remaining stderr.")
                                    self._x264_finished.set() # Signal that x264 has finished
                    logger.info(f"Finished reading remaining stderr.")

            except Exception as e:
                logger.opt(exception=e).error(f"Error in reading x264 stderr.")
            finally:
                if process and process.stderr:
                    process.stderr.close()
                logger.info(f"Stderr reader thread exiting.")


    def _initialize_encoder(self):
        """
        Initializes the x264 encoder by starting an x264.exe process.
        This runs in the worker process.
        """
        logger = self._logger
        logger.info(f"Initializing x264 encoder (x264.exe)...")

        # Initialize process-specific attributes (called in `_worker process`)
        self._x264_finished = threading.Event() # Event to signal x264 completion

        # Timecode now fully depends on the extractor. 

        # Prepare timecode log file path
        base_path, ext = os.path.splitext(self._output_path)
        self._timecode_log_path = base_path + "_timecode.txt"
        self._encoder_intermediates = base_path + "_noTC.mp4"
        self._processed_frames = 0
        
        x264_executable = self._x264_path

        # Check if x264.exe exists in the script directory, otherwise rely on PATH
        if x264_executable.exists():
            logger.info(f"Found x264.exe in script directory: {x264_executable}")
        else:
            x264_executable = 'x264' # Rely on system PATH (Windows will append .exe if needed)
            logger.info(f"x264.exe not found in script directory, relying on system PATH.")

        # --- Build x264 command line ---
        height, width, channels = self._frame_size

        # default internal params
        x264_params = {
            'input-res': f'{width}x{height}',
            'fps': str(self._fps),
            'input-csp': 'bgr',       # Input color space, usually BGR.
            'preset': self._preset,
            'threads': str(self._threads),
            'output': self._encoder_intermediates
            # x264.exe typically overwrites by default, no explicit -y needed.
        }
        x264_params.update(self._encoder_kwargs) # Overwrite user cfg.

        # Favour CRF first, then consider using bitrate.
        if self._crf is not None:
            x264_params['crf'] = str(self._crf)
        elif self._bitrate is not None:
            # x264.exe expects bitrate in kbps
            bitrate_kbps = int(self._bitrate) // 1000
            x264_params['bitrate'] = str(bitrate_kbps)
        # If neither is provided, use default CRF 23 (specified in __init__)
        if self._rc_lookahead is not None:
            x264_params['rc-lookahead'] = str(self._rc_lookahead)

        # IO params
        x264_params['demuxer'] = 'raw'

        x264_cmd = [str(x264_executable),]
        # Convert dict to command line args
        for key, value in x264_params.items():
            x264_cmd.extend([f'--{key}', str(value)])
        x264_cmd.append('-') # stdin Input

        logger.info(f"Starting x264.exe with command: {' '.join(x264_cmd)}")

        try:
            # Start the x264.exe process with pipes for stdin, stdout, and stderr
            self._x264_process = subprocess.Popen(
                x264_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, # x264 might not output much to stdout when outputting to file
                stderr=subprocess.PIPE  # x264 outputs progress/info to stderr
            )
            logger.info(f"x264.exe process started with PID {self._x264_process.pid}")

            # Start separate threads to continuously read stdout and stderr.
            self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
            self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
            self._stdout_thread.start()
            self._stderr_thread.start()

            # Open timecode log file and write header
            if self._timecode_extractor:
                try:
                    self._timecode_file = open(self._timecode_log_path, 'w')
                    self._timecode_file.write("# timecode format v2\n") # Write header
                    logger.info(f"Timecode log file opened at {self._timecode_log_path} and header written.")
                except IOError as e:
                    logger.error(f"Failed to open timecode log file {self._timecode_log_path}: {e}")
                    self._timecode_file = None # Ensure it's None if open fails
                    # Encoding will proceed, but timecodes won't be logged.

        except FileNotFoundError:
            # Handle the case where the x264 executable is not found
            logger.error(f"x264.exe command not found. Please ensure x264.exe "
                         "is in your system's PATH, or place 'x264.exe' in the same directory as the script.")
            self._x264_process = None
            raise # Re-raise the exception to indicate initialization failure

        except Exception as e:
            # Handle other potential errors during process startup
            logger.opt(exception=e).error(f"Error in starting x264.exe process.")
            self._x264_process = None
            raise # Re-raise the exception to indicate initialization failure


    def _encode_frames(self, frames_list: List[np.ndarray], ext_info: Optional[List[dict]] = None, **kwargs) -> bool:
        """
        Encodes a list of frame chunks using the initialized x264 encoder.
        Each chunk in the list is an np.ndarray of frames.

        Args:
            frames_list (List[np.ndarray]): A list of np.ndarray objects.
                Each np.ndarray has a shape like (num_frames_in_chunk, H, W, C),
                or (num_frames_in_chunk, h, w, c) if ExtInfoExtractor is loaded,
                where HWC is the dimension of buffer data and hwc is the 
                dimension of ExtInfoExtractor output.

        Returns:
            bool: True if encoding was successful, False otherwise.

        Raises:
            EncoderException: when critical error occurs and continue encoding is not possible.
        """
        pid, friendly_name = mp.current_process().pid, "X264Encoder"
        logger = self._logger

        if not frames_list: # Check if the list itself is empty
            logger.warning(f"Received empty frames list. Skip encoding.")
            return False

        if not self._x264_process or not self._x264_process.stdin or self._x264_process.stdin.closed:
            raise EncoderException("x264 process or stdin not available/closed.")

        # dimensions for unpacking
        height, width, channels = self._frame_size
        expected_pixels = height * width * channels

        for frame_chunk_arr in frames_list: # Iterate over each ndarray in the list
            if frame_chunk_arr is None or frame_chunk_arr.size == 0:
                logger.warning(f"Received empty or None frame_chunk_arr in list. Skip encoding.")
                continue # Skip this chunk

            if expected_pixels > frame_chunk_arr[0].size:
                raise EncoderException(f"Frame data mismatch (smaller) than expected. "
                    f"slot pixels: {frame_chunk_arr[0].size}, expected: {expected_pixels}")
            
            # Unpack the data from the head of buffer slots.
            frame_chunk_arr = frame_chunk_arr.reshape(frame_chunk_arr.shape[0], -1)[:, :expected_pixels].reshape(-1, height, width, channels)
            
            if ext_info: # have ext_info = have _timecode_extractor
                timecodes_list = [item[self._timecode_extractor.timecode_key] for item in ext_info]
            else: timecodes_list = []
            
            # --- Write Image Data for this chunk to x264 stdin (frame by frame) ---
            try:
                for idx, frame_view in enumerate(frame_chunk_arr):
                    # TODO: Batch write can be accepted by x264.
                    self._x264_process.stdin.write(frame_view.tobytes()) 
                    self._processed_frames += 1

                    if self._timecode_file and timecodes_list: 
                        # actually, have _timecode_file = have _timecode_extractor = timecodes_list
                        current_tc = timecodes_list[idx]
                        # Rescale tc in timebase to ms(float) for timecode format v2
                        current_tc_ms = current_tc * 1000.0 / self._timecode_extractor.timebase
                        self._timecode_file.write(f"{current_tc_ms}\n")
                            
                        # Optional: Monotonicity check
                        if self._last_timecode_value is not None and current_tc < self._last_timecode_value:
                            logger.warning(f"Timecode non-monotonic. Frame {self._processed_frames}. "
                                            f"Prev: {self._last_timecode_value}, Curr: {current_tc}")
                        self._last_timecode_value = current_tc
            except BrokenPipeError as e:
                raise EncoderException(f"BrokenPipeError while writing frame data to x264. Frame index: {self._processed_frames}.", 
                    pid=pid, name=friendly_name) from e
            except OSError as e:
                # Probably due to timecode file.
                raise EncoderException(f"OSError while writing data. Frame index: {self._processed_frames}.", 
                    pid=pid, name=friendly_name) from e
            except Exception as e:
                raise EncoderException(f"Error writing frame data for a chunk: {e}. Frame index: {self._processed_frames}.", 
                    pid=pid, name=friendly_name) from e

        # Flush once after all chunks in frames_list have been processed.
        if self._x264_process and self._x264_process.stdin and not self._x264_process.stdin.closed:
            try:
                self._x264_process.stdin.flush()
            except OSError as e: # Catch errors if stdin is already closed (e.g. BrokenPipe)
                logger.error(f"Error flushing stdin (possibly already closed): {e}")

        return True
    
    def _uninitialize_encoder(self):
        """
        Uninitializes the x264 encoder by stopping the x264.exe process and reader threads.
        This runs in the worker process.
        """
        logger = self._logger

        logger.info(f"Uninitializing x264 encoder (x264.exe)...")

        # Signal x264.exe to finish by closing stdin
        if self._x264_process and self._x264_process.stdin:
            logger.trace(f"Closing x264.exe stdin to signal end of stream.")
            try:
                self._x264_process.stdin.close()
            except Exception as e:
                logger.opt(exception=e).error(f"Error in closing x264.exe stdin.")

        # Close the timecode log file if it was opened
        if self._timecode_file:
            logger.trace(f"Closing timecode log file.")
            try:
                self._timecode_file.close()
            except Exception as e:
                logger.opt(exception=e).error(f"Error in closing timecode log file.")
            finally:
                self._timecode_file = None

        # Wait for x264.exe to finish encoding gracefully, with a timeout
        logger.info(f"Waiting for x264.exe to finish encoding...")
        finished_gracefully = self._x264_finished.wait(timeout=10.0) # Wait for up to 10 seconds

        if finished_gracefully:
            logger.info(f"x264.exe finished encoding (detected completion message).")
        else:
            logger.warning(f"x264.exe did not signal completion within timeout, or process ended before message.")

        # Wait for a while (optional, x264 should flush on stdin close)
        # time.sleep(0.5) # Give x264 some time to finish writing the file if needed

        # Terminate x264.exe process if it's still running
        # TODO: Blocking to UI choice?
        if self._x264_process and self._x264_process.poll() is None:
            logger.info(f"Terminating x264.exe process {self._x264_process.pid}...")
            try:
                self._x264_process.terminate() # on Windows terminate = kill
                self._x264_process.wait(timeout=10.0) # Wait a bit for termination
                logger.info(f"x264.exe process terminated.")
            except subprocess.TimeoutExpired:
                logger.warning(f"x264.exe process {self._x264_process.pid} did not terminate gracefully. Killing.")
                self._x264_process.kill()
            except Exception as e:
                logger.opt(exception=e).error(f"Error in terminating x264.exe process.")
            finally:
                self._x264_process = None # Clear process reference
        elif self._x264_process:
             # Process already exited, just clear reference
             logger.info(f"x264.exe process {self._x264_process.pid} has exited.")

        # Signal reader threads to stop and join them
        # The threads' loops check self._running, which is cleared by BaseVideoEncoder.stop()
        if self._stdout_thread and self._stdout_thread.is_alive():
            logger.trace(f"Waiting for stdout reader thread to join...")
            self._stdout_thread.join(timeout=5.0)
            if self._stdout_thread.is_alive():
                 logger.warning(f"Stdout reader thread did not exit gracefully.")
            self._stdout_thread = None

        if self._stderr_thread and self._stderr_thread.is_alive():
            logger.trace(f"Waiting for stderr reader thread to join...")
            self._stderr_thread.join(timeout=5.0)
            if self._stderr_thread.is_alive():
                 logger.warning(f"Stderr reader thread did not exit gracefully.")
            self._stderr_thread = None

        self._x264_process = None # All members finished using the Popen obj, reset to None

        if self._timecode_extractor and self._do_mux_timecode:
            self._mux_timecode(tc_file=self._timecode_log_path, mp4_file=self._encoder_intermediates, out_file=self._output_path)

        logger.success(f"X264 encoder uninitialized.")
    
    def _mux_timecode(self, tc_file, mp4_file, out_file) -> bool:
        """
        Muxes the timecode file with the encoded video file using mp4fpsmod.
        """
        logger = self._logger

        logger.info(f"Starting timecode muxing with mp4fpsmod...")
        # Get the directory of the current module
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = os.path.dirname(module_dir)
        mp4fpsmod_path_in_tools = os.path.join(project_root_dir, 'tools', 'mp4fpsmod.exe')

        if os.path.exists(mp4fpsmod_path_in_tools):
            mp4fpsmod_executable = mp4fpsmod_path_in_tools
            logger.info(f"Found mp4fpsmod.exe in tools directory: {mp4fpsmod_executable}")
        else:
            mp4fpsmod_executable = 'mp4fpsmod' # Rely on system PATH (Windows will append .exe if needed)
            logger.info(f"mp4fpsmod.exe not found in tools directory, relying on system PATH.")
        try:
            # Run mp4fpsmod to mux the timecode file with the encoded video file
            mp4fpsmod_cmd = [
                mp4fpsmod_executable,
                '-c', # fix non-zero of timecode head
                '-t', tc_file,  # Timecode file path
                '-o', out_file,  # Output file path
                mp4_file,  # Input video file path
            ]
            logger.info(f"Running mp4fpsmod with command: {' '.join(mp4fpsmod_cmd)}")
            subprocess.run(mp4fpsmod_cmd, check=True)
            logger.info(f"Timecode muxing completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running mp4fpsmod: {e}")
            logger.error(f"mp4fpsmod stderr: {e.stderr.decode()}")
            # raise  # Re-raise the exception to indicate muxing failure
            return False
        except FileNotFoundError:
            # Handle the case where the mp4fpsmod executable is not found
            logger.error(f"mp4fpsmod command not found. Please ensure mp4fpsmod.exe is in your system's PATH, or place 'mp4fpsmod.exe' in the same directory as the script.")
            # raise  # Re-raise the exception to indicate initialization failure
            return False
        except Exception as e:
            # Handle other potential errors during muxing
            logger.opt(exception=e).error(f"Error in running mp4fpsmod.")
            # raise  # Re-raise the exception to indicate muxing failure
            return False
        return True

# In spawn mode, the patch function must be accessible by any process (not in if __name__ == '__main__')
def patch_message(record):
    extra = record["extra"]
    # 获取 extra 中的字段（如果存在）
    friendly_name = extra.get("friendly_name")
    
    # 根据字段是否存在，构建前缀
    prefix_parts = []
    if friendly_name:
        prefix_parts.append(f"{friendly_name}")
    
    if prefix_parts:
        prefix = f"[{' '.join(prefix_parts)}] "
        # 修改 record["message"]，这会影响最终打印的内容
        record["message"] = f"{prefix}{record['message']}"

if __name__ == "__main__":
    from loguru import logger
    import sys
            
    logger.remove()

    logger = logger.patch(patch_message)
    logger.add(sys.stderr, enqueue=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{process.id}</cyan>:"
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

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
    frame_server = None
    try:
        shared_buffer = ProcessSafeSharedRingBuffer(
            create=True,
            buffer_capacity=buffer_capacity,
            frame_shape=(frame_height, frame_width, frame_channels),
            dtype=np.uint8
        )
        print(f"Shared ring buffer created: Metadata SHM: {shared_buffer.metadata_name}, Data SHM: {shared_buffer.data_name}")

        print("Creating FrameServer instance...")
        frame_server = FrameServer(create=True, ring_buffer=shared_buffer)

        # 2. 创建 X264Encoder 实例，并传入 FrameServer
        print("Creating X264Encoder instance...")

        # 构建一个字典来存放非必须的编码参数
        encoding_kwargs = {
            'frame_size': (frame_height, frame_width, frame_channels),
            'fps': fps,
            'crf': 23, # 添加 crf 参数来控制编码质量
            'preset': 'medium' # 添加 preset 参数
            # 可以在这里添加更多编码参数，例如 'bitrate', 'threads' 等
        }

        encoder = X264Encoder(
            frame_server=frame_server, # Pass the FrameServer instance
            output_path=output_file,
            batch_size=batch_size, # batch_size is a required parameter
            inject_logger=logger,
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
        
        print(f"There are {shared_buffer.unread_count()} (Not accurate in current FrameServer version) frames left in the buffer.")

        # 6. 关闭并解除链接 FrameServer 和共享缓冲区
        print("Cleaning up shared memory...")
        if frame_server:
            try:
                frame_server.close()
                frame_server.unlink()
            except Exception as e:
                print(f"Error closing/unlinking frameserver in main: {e}")
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
    mock_shared_buffer = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=10, frame_shape=(10,10,3))
    mock_frame_server = FrameServer(create=True, ring_buffer=mock_shared_buffer)

    print("Creating X264Encoder instance for profiling...")
    encoding_kwargs = {
        'frame_size': (frame_height, frame_width, frame_channels),
        'fps': fps,
        'crf': 23,
        'preset': 'medium', 
        'threads': 0 
    }

    # Ensure the X264Encoder class is defined above this block
    encoder = X264Encoder(
        frame_server=mock_frame_server,
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
        mock_frame_server.close()
        mock_frame_server.unlink()
        mock_shared_buffer.close()
        mock_shared_buffer.unlink()
        if os.path.exists(output_file):
            try:
                # os.remove(output_file)
                print(f"Cleaned up temporary output file: {output_file}")
            except Exception as e_del:
                print(f"Error deleting temporary file {output_file}: {e_del}")

    print("Profiling test finished.")
