import numpy as np
from threading import Thread, Event, Lock, Condition

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

from collections import deque
from typing import List, Optional, Any, Type # Import Type for class hints

from huateng_camera_tc import Camera
# from ringbuffer import RingBuffer # No longer needed
from shared_ring_buffer import ProcessSafeSharedRingBuffer
# No longer need specific analyzer import here
# from nnanalyzer import MySpecificNNAnalyzer
from nnanalyzer import NNAnalyzer # 导入基类用于类型提示
from videoencoder import BaseVideoEncoder # , X264Encoder # Import video encoder classes
from x264_encoder_x264 import X264Encoder # Import video encoder classes

import time

import mvsdk

FRAME_TIME = 8.3333 # 5ms = 200fps, a temp control here
# 此时encode出来的视频实际fps是120.061，但是实际（拍摄时钟）计算得到的fps是120.47，未知原因。

class CameraSystem:
    """
    Manages camera frame capture, analysis, and video encoding using separate threads.

    This system orchestrates the flow of frames from a camera to an analyzer and a video encoder.
    It uses a ring buffer for temporary frame storage and threading for concurrent operations.
    The lifecycle (start/stop) of the provided analyzer and video encoder instances
    It creates and manages the internal components including the shared buffer
    for video encoding.
    """
    def __init__(self,
                 camera: Camera,
                 AnalyzerClass: Type[NNAnalyzer],
                 VideoEncoderClass: Type[BaseVideoEncoder],
                 analyzer_config: dict,
                 video_encoder_config: dict,
                 buffer_capacity: int = 600): # Added buffer_capacity config
        """
        Initializes the CameraSystem by creating internal components.
        Args:
            camera (Camera): The camera object to grab frames from.
            AnalyzerClass (Type[NNAnalyzer]): The class of the NN analyzer to use.
            VideoEncoderClass (Type[BaseVideoEncoder]): The class of the video encoder to use.
            analyzer_config (dict): Configuration dictionary for the analyzer.
            video_encoder_config (dict): Configuration dictionary for the video encoder.
                                         Must include 'output_path', 'batch_size', etc.
                                         It should NOT include 'shared_buffer'.
            buffer_capacity (int): Capacity of the shared ring buffer. Defaults to 600.
        """
        self.snapshot_condition = Condition(Lock())
        self.running = Event()
        self.camera = camera
        self.thread_pool = []
        self.shared_buffer: Optional[ProcessSafeSharedRingBuffer] = None # Initialize as None

        # --- Create Shared Buffer ---
        try:
            original_width = self.camera.width
            original_height = self.camera.height # This is the original image height
            # Get the output frame height from the camera, which includes appended rows for timecode
            output_frame_height = self.camera.output_frame_height 
            channels = self.camera.channels # Assuming camera provides channels, else default to 3
            
            # frame_shape for the shared buffer should use the full output height
            frame_shape_for_buffer = (output_frame_height, original_width, channels)
            dtype = np.uint8 # Assuming uint8
            # print(f"CameraSystem: Original image HxWxC: {original_height}x{original_width}x{channels}")
            # print(f"CameraSystem: Output frame H'xWxC (with timecode space): {output_frame_height}x{original_width}x{channels}")
            print(f"CameraSystem: Determined frame shape for shared buffer: {frame_shape_for_buffer}, dtype {dtype}")

        except AttributeError as e:
            print(f"Warning: Camera object missing attributes (width, height, output_frame_height, or channels): {e}. Using defaults.")
            # Define defaults if camera attributes are not available
            original_width, original_height, output_frame_height, channels = 1280, 1024, 1025, 3
            frame_shape_for_buffer = (output_frame_height, original_width, channels)
            dtype = np.uint8
            print(f"CameraSystem: Using default frame shape for shared buffer {frame_shape_for_buffer}, dtype {dtype}")


        print(f"CameraSystem: Creating Shared Ring Buffer (capacity: {buffer_capacity}, shape: {frame_shape_for_buffer}, dtype: {dtype})...")
        try:
            self.shared_buffer = ProcessSafeSharedRingBuffer(
                create=True,
                buffer_capacity=buffer_capacity,
                frame_shape=frame_shape_for_buffer, # Use the correctly determined shape
                dtype=dtype
            )
            print("CameraSystem: Shared Ring Buffer created.")
        except Exception as e:
            print(f"FATAL: Failed to create Shared Ring Buffer: {e}")
            # Handle buffer creation failure (e.g., raise exception or set a failed state)
            raise RuntimeError("Failed to initialize shared buffer") from e

        # --- Instantiate Components ---
        print("CameraSystem: Instantiating Analyzer...")
        self.analyzer = AnalyzerClass(**analyzer_config)
        print("CameraSystem: Analyzer instantiated.")

        print("CameraSystem: Instantiating VideoEncoder...")
        # Get camera's target FPS
        try:
            camera_fps = self.camera.target_fps
            try: 
                camera_actual_fps = self.camera.actual_fps
                print(f"CameraSystem: Camera target FPS is {camera_fps:.2f}, actual FPS is {camera_actual_fps:.2f}. Using actual FPS.")
                camera_fps = camera_actual_fps # Use actual FPS if available
            except AttributeError:
                print(f"CameraSystem: Camera target FPS is {camera_fps:.2f}, actual FPS is unknown. Using target FPS.")
        except AttributeError:
            print("Warning: Camera object does not have target_fps attribute. Using 0.")
            camera_fps = 0.0 # Default or handle error

        # Get timebase for timecode from the camera.
        try:
            timebase = self.camera.timecode_timebase # This should be a float or int representing timebase.
            print(f"CameraSystem: Camera timebase is {timebase}.")
        except AttributeError:
            print("Warning: Camera object does not have timebase attribute. Using 10000.")
            timebase = 10000 # Default or handle error
        
        # The 'frame_size' passed to VideoEncoderClass should be the ORIGINAL image size,
        # as the encoder itself will handle stripping any appended data like timecodes.
        # This was already handled in the __main__ example, but good to be explicit.
        # We assume video_encoder_config already contains the original 'frame_size'.
        # If not, it should be added here based on self.camera.height, self.camera.width.
        # For now, we trust it's correctly set by the caller or in __main__.

        # Inject the created shared buffer, camera FPS and timebase of timecode
        # into the video encoder config
        encoder_final_config = {
            **video_encoder_config, # This should contain 'frame_size' as original dimensions
            'shared_buffer': self.shared_buffer,
            'camera_fps': camera_fps,
            'timebase': timebase,
        }
        self.video_encoder = VideoEncoderClass(**encoder_final_config)
        print("CameraSystem: VideoEncoder instantiated.")


    def submit_frame(self, frame: np.ndarray, timeout: float = 1.0) -> bool:
        """
        Submits a single frame to the shared buffer with timeout handling.

        Args:
            frame: The frame (without batch dimension) to submit.
            timeout: Seconds to wait for space in the buffer.

        Returns:
            True if successful, False if timeout or error occurred.
        """
        if not self.running.is_set() or self.shared_buffer is None:
            print("Submitting frame failed: CameraSystem not running or buffer not initialized.") # Optional
            return False

        try:
            # Add batch dimension as required by shared_buffer.put
            frame_batch = np.expand_dims(frame, axis=0)
            success = self.shared_buffer.put(frame_batch, timeout=timeout)
            if not success:
                print(f"Submitting frame failed: Timeout ({timeout}s) submitting frame to shared buffer.")
                # Consider adding more robust error handling/logging here
            return success
        except Exception as e:
            print(f"Error submitting frame to shared buffer: {e}")
            return False

    def capture_thread(self):
        """Captures frames from the camera and submits them to the shared buffer."""
        while self.running.is_set():
            # huateng_camera_tc.Camera.grab() returns a single frame with timecode appended
            combined_frame = self.camera.grab() 
            # The grab method returns a *view* of ndarray
            # So that submit_frame should NOT block, ensuring no skipping frames.
            if combined_frame is not None:
                # Check if encoder is ready before submitting
                # Add a small initial delay or check only after first few frames if needed
                if self.video_encoder and not self.video_encoder.is_ready:
                    print("Capture thread: Warning - VideoEncoder worker is not ready. Frame will submit but might fill buffer.")
                    # Optionally, could skip submission or sleep briefly if encoder not ready
                
                if not self.submit_frame(combined_frame, timeout=FRAME_TIME / 1000): # Submit the combined frame
                    # Handle submission failure (e.g., log, skip frame, slow down?)
                    print("Capture thread: Failed to submit frame, buffer might be full or error occurred. Frame dropped.")
                    # Depending on requirements, might need to sleep or break here
                    # time.sleep(0.01) # Example: small sleep if buffer is full

    def snapshot_thread(self):
        """Waits for a signal, then peeks the latest frame from the shared buffer for analysis."""
        while self.running.is_set():
            with self.snapshot_condition:
                self.snapshot_condition.wait()
                if not self.running.is_set(): # Check running flag again after wait
                    break

                # Ensure shared_buffer is initialized before peeking
                if self.shared_buffer is None:
                    print("Snapshot thread: Error - Shared buffer is not initialized.")
                    continue

                # Peek the latest frame (returns a copy)
                analysis_frame = self.shared_buffer.peek_last_frame()

                if analysis_frame is None:
                    print("Snapshot thread: No frame available in shared buffer or buffer error.")
                    continue

                # Submit the frame copy to the analyzer
                # The analysis_frame contains the timecode. Analyzer needs to handle it or it needs to be stripped here.
                # Assuming analyzer_config['frame_size'] is original, we should strip timecode here.
                try:
                    original_height = self.camera.height # Original height without timecode
                    if analysis_frame.shape[0] > original_height:
                        image_for_analyzer = analysis_frame[:original_height, :, :]
                        # print(f"Snapshot thread: Submitting stripped frame {image_for_analyzer.shape} to analyzer.") # Debug
                        self.analyzer.submit_frame(image_for_analyzer)
                    else: # Frame is already original size or smaller (error?)
                        # print(f"Snapshot thread: Submitting frame {analysis_frame.shape} (as-is or error) to analyzer.") # Debug
                        self.analyzer.submit_frame(analysis_frame) # Submit as-is if no appended data detected
                except AttributeError:
                    print("Snapshot thread: Warning - Camera object missing 'height' attribute. Submitting frame as-is.")
                    self.analyzer.submit_frame(analysis_frame) # Fallback
                except Exception as e:
                    print(f"Snapshot thread: Error preparing frame for analyzer: {e}. Submitting as-is.")
                    self.analyzer.submit_frame(analysis_frame) # Fallback

    # encode_thread is removed. VideoEncoder worker reads directly from shared_buffer.

    def snapshot_and_analyze(self):
        """Triggers the snapshot thread and retrieves the analysis result."""
        with self.snapshot_condition:
            self.snapshot_condition.notify_all()  # 发送快照开始信号

        # 4. 调用 analyzer 的 get_result 接口
        try:
            # print("Waiting for analysis result...") # 调试信息
            # 可以设置超时，避免永久阻塞
            result = self.analyzer.get_result(timeout=5.0)
            print(f"Analysis Result: {result}")
        except mp.queues.Empty:
            print("Error: Did not receive analysis result within timeout.")
        except Exception as e:
            print(f"Error getting analysis result: {e}")


    def start(self):
        """Starts the internal threads of the CameraSystem."""
        if self.running.is_set():
            print("CameraSystem already running.")
            return
        print("Starting CameraSystem threads...")
        self.running.set()
        # Analyzer and VideoEncoder start/stop are managed externally by the caller
        # Ensure components are initialized before starting threads
        if self.analyzer is None or self.video_encoder is None:
             raise RuntimeError("Analyzer or VideoEncoder not initialized before starting CameraSystem threads.")

        self.thread_pool.append(Thread(target=self.capture_thread, name="CaptureThread"))
        self.thread_pool.append(Thread(target=self.snapshot_thread, name="SnapshotThread"))
        # self.thread_pool.append(Thread(target=self.encode_thread, name="EncodeThread")) # Removed
        for thread in self.thread_pool:
            thread.start()

    def stop(self):
        """
        Stops the internal threads (capture, snapshot) of the CameraSystem.
        This does NOT clean up resources like the shared buffer. Call close() for that.
        """
        if not self.running.is_set():
            print("CameraSystem internal threads already stopped.")
            return
        print("Stopping CameraSystem internal threads...")
        # Analyzer and VideoEncoder stop are managed externally
        self.running.clear()
        # Wake up snapshot_thread if it's waiting on the condition
        with self.snapshot_condition:
            self.snapshot_condition.notify_all()
        print("Waiting for CameraSystem internal threads to join...")

        while self.thread_pool:
            thread = self.thread_pool.pop()
            print(f"Joining {thread.name}...")
            thread.join() # the thread must join.
            print(f"{thread.name} joined.")
        # self.thread_pool should be empty now

        print("CameraSystem internal threads stopped.")

    def close(self):
        """
        Cleans up resources managed by CameraSystem, specifically the shared ring buffer.
        Should be called after stop() and after external components using the buffer are stopped.
        """
        # Clean up the shared buffer created by CameraSystem
        if self.shared_buffer:
            print("Closing and unlinking CameraSystem's Shared Ring Buffer...")
            try:
                self.shared_buffer.close() # Close connection first
                self.shared_buffer.unlink() # Then unlink
                print("CameraSystem's Shared Ring Buffer closed and unlinked.")
            except Exception as e:
                print(f"Error cleaning up CameraSystem's Shared Ring Buffer: {e}")
            finally:
                self.shared_buffer = None # Clear reference
        else:
            print("CameraSystem shared buffer already cleaned up or was not initialized.")

        print("CameraSystem closed.")


if __name__ == "__main__":
    # Import specific implementations for testing in __main__
    from nnanalyzer import MySpecificNNAnalyzer
    # numpy is likely needed by dependencies or for frame creation if camera mock is used
    import numpy as np

    '''
    简单的测试
    '''
    # 枚举相机并选择
    DevList = mvsdk.CameraEnumerateDevice()
    if not DevList:
        print("No cameras found. Exiting test.")
        exit()
    # Use huateng_camera_tc.Camera which appends timecode
    camera = Camera(DevList[0], exposure_time_ms=FRAME_TIME, tc=True) # Use exposure_time_ms as per huateng_camera_tc
    if not camera.open():
        print("Failed to open camera. Exiting test.")
        exit()

    # Define configurations for components
    analyzer_config = {}
    video_encoder_config = {}
    camera_system = None # Initialize camera_system to None

    try:
        # --- Determine Original Frame Shape (for config) ---
        try:
            original_width = camera.width
            original_height = camera.height # This is the original image height
            channels = camera.channels
            original_frame_shape = (original_height, original_width, channels)
            print(f"CameraSystem __main__: Original camera frame HxWxC: {original_height}x{original_width}x{channels}")
        except AttributeError:
            print("CameraSystem __main__: Warning: Camera object does not provide full shape info. Using defaults for config.")
            original_height, original_width, channels = 1024, 1280, 3
            original_frame_shape = (original_height, original_width, channels) # Default

        # --- Prepare Configurations ---
        # Analyzer expects original frame size
        analyzer_config = {
            'model_path': 'example_path', # TODO: Use actual config
            'frame_size': original_frame_shape 
        }
        # VideoEncoder also expects original frame size in its config
        video_encoder_config = {
            'output_path': 'output.mp4', # Example output path
            'batch_size': 30,
            'fps': camera.actual_fps, 
            'preset': 'fast', 
            'crf': 23, 
            'frame_size': original_frame_shape, # IMPORTANT: This is the ORIGINAL frame size
            'threads': 2, 
            'rc_lookahead': 60,
            # 'shared_buffer' and 'camera_fps' will be added by CameraSystem.__init__
        }

        # --- Create CameraSystem (which creates components and buffer) ---
        print("Creating CameraSystem instance...")
        camera_system = CameraSystem(
            camera=camera,
            AnalyzerClass=MySpecificNNAnalyzer,
            VideoEncoderClass=X264Encoder,
            analyzer_config=analyzer_config,
            video_encoder_config=video_encoder_config,
            buffer_capacity=600 # Example capacity
        )
        print("CameraSystem instance created with internal components.")

        # --- Start Components (Managed Externally) ---
        # Start order: Analyzer and Encoder first, then CameraSystem threads
        print("Starting Analyzer...")
        camera_system.analyzer.start()
        print("Starting VideoEncoder...")
        camera_system.video_encoder.start()
        print("Starting CameraSystem internal threads...")
        camera_system.start() # Starts capture and snapshot threads

        # --- System Running ---
        print("System running. Press Enter to trigger snapshot and analyze.")
        time.sleep(1) # 等待线程启动
        input("Press Enter for first snapshot...")
        camera_system.snapshot_and_analyze()
        input("Press Enter for second snapshot...")
        camera_system.snapshot_and_analyze()
        input("Press Enter to stop the system...\n")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # --- Stop Components (Managed Externally) ---
        # Stop order: Analyzer and Encoder first, then CameraSystem threads & buffer cleanup
        if camera_system:
            # 1. Stop CameraSystem internal threads (capture/snapshot)
            print("Stopping CameraSystem internal threads...")
            camera_system.stop() # Stops capture/snapshot threads

            # 2. Stop Analyzer and Encoder processes first
            if camera_system.analyzer:
                print("Stopping Analyzer...")
                camera_system.analyzer.stop()
            if camera_system.video_encoder:
                print("Stopping VideoEncoder...")
                camera_system.video_encoder.stop()

            # 3. Clean up CameraSystem resources (shared buffer)
            print("Closing CameraSystem resources (shared buffer)...")
            camera_system.close() # Closes and unlinks buffer

        # --- Close Camera ---
        print(f"Camera got {camera.frames_captured} frames in total.")
        print("Closing camera hardware...")
        camera.close()

        print("System shutdown complete.")
