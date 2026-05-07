# Created by Google Gemini 2.5 Flash with Haiyun Huang.

# This Class handles the communication with the SAM2RT Remote Service.
# requests.Session is not irreplacable. Since the ability of picklization depends on platform(spawn/fork), 
# we use two separate requests.Session in the main process and the worker process.

# The Analyzer should first `start()`, call `set_prompt()` to set the prompt, 
# then call `submit_frame` to track the object.

import requests
import base64
import json
import time
import re
import multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
import numpy as np
import cv2 # For image encoding/decoding
from typing import Optional, Any, Tuple, Dict

from nnanalyzer import NNAnalyzer
from cameras.huateng_camera_tc import extract_tc_from_frames

class SAM2RTRemoteAnalyzer(NNAnalyzer):
    def __init__(self,
                 remote_address: str,
                 frame_shape: tuple[int, int, int], 
                 effective_image_shape: Optional[tuple[int, int, int]] = None, # H, W, C of the image itself
                 timecode_timebase: Optional[int] = None, # Denominator of timebase. 
                 track_timeout: float = 0.2, # Timeout for /track_frame requests in seconds
                 prompt_timeout: float = 60.0, # Timeout for /prompt_frame requests in seconds
                 use_proxy: bool = True, # Whether the session should use a proxy
                 ):
        # remote_address: "IP:port"
        super().__init__(remote_address, frame_shape)
        self.remote_address = remote_address
        self.use_proxy = use_proxy
        self.timecode_timebase = timecode_timebase
        self.track_timeout = track_timeout
        self.prompt_timeout = prompt_timeout
        self._request_session: Optional[requests.Session] = None # Requests session for the worker process, initialized in _initialize_analyzer

        # Shared variables for tracking state across processes (initialized in start)
        self._is_tracking_active: Optional[Synchronized[bool]] = None
        self._current_obj_id: Optional[Synchronized[int]] = None

        # Determine effective image shape and if timecode is appended
        self._is_timecode_appended = False
        if effective_image_shape:
            self.effective_image_height = effective_image_shape[0]
            self.effective_image_width = effective_image_shape[1]
            self.channels = effective_image_shape[2]
            # Check if frame_shape height is greater than effective_image_height, implying appended data
            if frame_shape[0] > self.effective_image_height and \
               frame_shape[1] == self.effective_image_width and \
               frame_shape[2] == self.channels:
                self._is_timecode_appended = True
            elif frame_shape[0] == self.effective_image_height and \
                 frame_shape[1] == self.effective_image_width and \
                 frame_shape[2] == self.channels:
                self._is_timecode_appended = False # No appended data
            else:
                raise ValueError(
                    f"Provided frame_shape {frame_shape} is inconsistent with effective_image_shape {effective_image_shape}. "
                    f"Height of frame_shape ({frame_shape[0]}) should be not less than effective_image_height ({self.effective_image_height}) "
                    f"and other dimensions should be the same."
                )
        else:
            # If no effective_image_shape is provided, assume frame_shape is the effective image shape
            self.effective_image_height = frame_shape[0]
            self.effective_image_width = frame_shape[1]
            self.channels = frame_shape[2]
            self._is_timecode_appended = False # No timecode appended by default

    def start(self):
        """
        Starts the SAM2RTRemoteAnalyzer.
        Initializes shared variables before calling the base class start method.
        """
        # Initialize shared variables in the main process before super().start()
        # This ensures they are properly shared across processes (main and worker)
        self._is_tracking_active = mp.Value('b', False)
        self._current_obj_id = mp.Value('i', 0)
        super().start()

    def _initialize_analyzer(self):
        """
        Initializes the analyzer in the worker process.
        Creates a requests session and checks if the remote service is ready.
        """
        print(f"SAM2RTRemoteAnalyzer worker process ({mp.current_process().pid}): Initializing...")
        self._request_session = requests.Session() # Create a new session for this worker process
        
        if not self.use_proxy:
            self._request_session.trust_env = False # Disable proxy detection
            self._request_session.proxies = {
                "http": None,
                "https": None,
            }
            print(f"SAM2RTRemoteAnalyzer worker process ({mp.current_process().pid}): Proxy disabled for requests session.")

        try:
            if not self._check_service_ready():
                raise RuntimeError(f"Remote SAM2RT service at {self.remote_address} is not ready.")
            print(f"SAM2RTRemoteAnalyzer worker process ({mp.current_process().pid}): Remote service is ready.")
        except Exception as e:
            print(f"SAM2RTRemoteAnalyzer worker process ({mp.current_process().pid}): FATAL Error during initialization: {e}")
            raise # Re-raise to signal failure to NNAnalyzer base class

    def _handle_set_prompts(self, frame: np.ndarray, prompts: dict, obj_id: int) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]], int]:
        """
        Handles the core logic for setting prompts in the worker process.
        This method is called via submit_frame with target='_handle_set_prompts'.
        """
        # Basic input validation
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            print(f"SAM2RTRemoteAnalyzer worker ({mp.current_process().pid}): Invalid frame input for _handle_set_prompts.")
            return None, None, 0
        if not isinstance(prompts, dict) or "points" not in prompts or "labels" not in prompts:
            print(f"SAM2RTRemoteAnalyzer worker ({mp.current_process().pid}): Invalid prompts input for _handle_set_prompts.")
            return None, None, 0
        if not isinstance(obj_id, int) or obj_id <= 0:
            print(f"SAM2RTRemoteAnalyzer worker ({mp.current_process().pid}): Invalid obj_id input for _handle_set_prompts.")
            return None, None, 0

        try:
            # Encode image to PNG bytes
            _, img_encoded = cv2.imencode('.png', frame)
            if img_encoded is None:
                print(f"SAM2RTRemoteAnalyzer worker ({mp.current_process().pid}): Could not encode image to PNG.")
                return None, None, 0
            img_bytes = img_encoded.tobytes()

            files = {'image': ('image.png', img_bytes, 'image/png')}
            data = {'prompt': json.dumps({"obj_id": obj_id, **prompts})}

            prompt_url = f"http://{self.remote_address}/prompt_frame"
            response = self._request_session.post(prompt_url, files=files, data=data, timeout=self.prompt_timeout)
            response.raise_for_status()
            json_response = response.json()

            mask_image, center_coordinate, recommand_reprompt = self._parse_remote_response(json_response, obj_id)

            if mask_image is not None and center_coordinate is not None: # If prompt was successful
                self._is_tracking_active.value = True
                self._current_obj_id.value = obj_id
            else:
                self._is_tracking_active.value = False # Ensure tracking is not active on failure
                self._current_obj_id.value = 0
            
            return mask_image, center_coordinate, recommand_reprompt

        except requests.exceptions.Timeout:
            print(f"SAM2RTRemoteAnalyzer worker ({mp.current_process().pid}): /prompt_frame request timed out after {self.prompt_timeout} seconds.")
            if self._is_tracking_active is not None:
                self._is_tracking_active.value = False
            if self._current_obj_id is not None:
                self._current_obj_id.value = 0
            return None, None, 0
        except requests.exceptions.RequestException as e:
            print(f"SAM2RTRemoteAnalyzer worker ({mp.current_process().pid}): Error sending /prompt_frame request: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            if self._is_tracking_active is not None:
                self._is_tracking_active.value = False
            if self._current_obj_id is not None:
                self._current_obj_id.value = 0
            return None, None, 0
        except Exception as e:
            print(f"SAM2RTRemoteAnalyzer worker ({mp.current_process().pid}): Unexpected error during _handle_set_prompts: {e}")
            if self._is_tracking_active is not None:
                self._is_tracking_active.value = False
            if self._current_obj_id is not None:
                self._current_obj_id.value = 0
            return None, None, 0

    def _analyze(self, frame: np.ndarray) -> Tuple[Tuple[Optional[np.ndarray], Optional[Tuple[int, int]], int], Optional[int]]:
        """
        Analyzes a single frame by sending it to the remote SAM2RT service for tracking.
        This method is called in the worker process.
        Args:
            frame (np.ndarray): The input frame, potentially with appended timecode data.
                                Shape: (H (with_tc), W, C).
        Returns:
            Tuple[
                  Tuple[Optional[np.ndarray], Optional[Tuple[int, int]], int], 
                  Optional[int]
                 ]: A nested-tuple containing:
                - mask_image (Optional[np.ndarray]): Decoded mask image, or None on failure.
                - center_coordinate (Optional[Tuple[int, int]]): (x, y) center coordinate, or None on failure.
                - recommand_reprompt (int): 0 for no re-prompt, 1 for re-prompt recommended.
        """
        # 检查共享变量是否已初始化
        if self._is_tracking_active is None or self._current_obj_id is None:
            print(f"SAM2RTRemoteAnalyzer worker ({mp.current_process().pid}): Shared tracking state not initialized. Call start() first.")
            return None, None, 0

        if not self._is_tracking_active.value:
            print(f"SAM2RTRemoteAnalyzer worker ({mp.current_process().pid}): Tracking not active. Call set_prompts() first.")
            return None, None, 0

        analysis_start_time = time.perf_counter_ns() // 1000
        
        mask_image: Optional[np.ndarray] = None
        center_coordinate: Optional[Tuple[int, int]] = None
        recommand_reprompt: int = 0

        try:
            # 1. Extract image data and original timecode
            if self._is_timecode_appended:
                original_images_batch, tc_array = extract_tc_from_frames(
                    np.expand_dims(frame, axis=0),
                    self.effective_image_height,
                    self.effective_image_width,
                    self.channels
                )
                if tc_array is not None and tc_array.size > 0:
                    original_timecode = int(tc_array[0])
                else:
                    print(f"SAM2RTRemoteAnalyzer worker ({mp.current_process().pid}): Warning - extract_tc_from_frames returned None or empty timecode.")
                    original_timecode = None
                frame_image = original_images_batch[0]
            else:
                frame_image = frame[:self.effective_image_height, :, :]
                # Timecodes remain None

            # 2. Encode image to PNG bytes
            _, img_encoded = cv2.imencode('.png', frame_image)
            if img_encoded is None:
                print(f"SAM2RTRemoteAnalyzer worker ({mp.current_process().pid}): Could not encode image to PNG.")
                return None, None, 0
            img_bytes = img_encoded.tobytes()

            files = {'image': ('image.png', img_bytes, 'image/png')}
            
            # 3. Send POST request to /track_frame
            track_url = f"http://{self.remote_address}/track_frame"
            # 使用工作进程的会话
            response = self._request_session.post(track_url, files=files, timeout=self.track_timeout)
            response.raise_for_status() # Raise an exception for HTTP errors
            json_response = response.json()

            mask_image, center_coordinate, recommand_reprompt = self._parse_remote_response(json_response, self._current_obj_id.value)

        except requests.exceptions.Timeout:
            print(f"SAM2RTRemoteAnalyzer worker ({mp.current_process().pid}): WARNING: /track_frame request timed out after {self.track_timeout} seconds.")
            return None, None, 0
        except requests.exceptions.RequestException as e:
            print(f"SAM2RTRemoteAnalyzer worker ({mp.current_process().pid}): Error sending /track_frame request: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None, None, 0
        except Exception as e:
            print(f"SAM2RTRemoteAnalyzer worker ({mp.current_process().pid}): Unexpected error during _analyze: {e}")
            return None, None, 0

        # 4. Calculate time cost (without wrapper)
        analysis_end_time = time.perf_counter_ns() // 1000
        analysis_duration_seconds = analysis_end_time - analysis_start_time
        print(f"SAM2RTRemoteAnalyzer worker ({mp.current_process().pid}): Analysis duration: {analysis_duration_seconds} us.")
        
        return mask_image, center_coordinate, recommand_reprompt
    

    def _uninitialize_analyzer(self):
        """
        Uninitializes the analyzer in the worker process.
        Closes the requests session.
        """
        print(f"SAM2RTRemoteAnalyzer worker process ({mp.current_process().pid}): Uninitializing...")
        if self._request_session:
            self._request_session.close()
            self._request_session = None
        print(f"SAM2RTRemoteAnalyzer worker process ({mp.current_process().pid}): Uninitialization complete.")

    # Define a constant for the re-prompt threshold
    REPROMPT_THRESHOLD = 5 # Example: recommend re-prompt after 5 consecutive lost frames

    def _parse_remote_response(self, json_response: Dict[str, Any], obj_id: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]], int]:
        """
        Parses the JSON response from the remote SAM2RT service.
        This function centralizes the parsing logic for both prompt and track responses,
        making it easier to adapt to future API changes.

        Args:
            json_response (Dict[str, Any]): The raw JSON response from the remote service.
            obj_id (Optional[int]): The object ID being tracked/prompted for, used to extract specific data.

        Returns:
            Tuple[Optional[np.ndarray], Optional[Tuple[int, int]], int]:
            - mask_image (Optional[np.ndarray]): Decoded mask image, or None on failure.
            - center_coordinate (Optional[Tuple[int, int]]): (x, y) center coordinate, or None on failure.
            - recommand_reprompt (int):
                - 0: No re-prompt needed (success or minor warning).
                - 1: Re-prompt recommended (e.g., tracking lost for multiple frames).
        """
        mask_image: Optional[np.ndarray] = None
        center_coordinate: Optional[Tuple[int, int]] = None
        recommand_reprompt: int = 0
        
        remote_status = json_response.get("status")
        remote_message = json_response.get("message", "")
        tracked_objects = json_response.get("tracked_objects", {})

        target_obj_data: Optional[Dict[str, Any]] = None
        if obj_id is not None and str(obj_id) in tracked_objects:
            target_obj_data = tracked_objects[str(obj_id)]
        elif len(tracked_objects) == 1: # If only one object, assume it's the target
            target_obj_data = next(iter(tracked_objects.values()))

        if target_obj_data:
            mask_info = target_obj_data.get("mask_info", {})
            center_list = mask_info.get("center")
            if center_list and len(center_list) == 2:
                center_coordinate = tuple(center_list)

            mask_b64 = target_obj_data.get("mask_image_png_b64")
            if mask_b64:
                try:
                    mask_bytes = base64.b64decode(mask_b64)
                    mask_np = np.frombuffer(mask_bytes, np.uint8)
                    mask_image = cv2.imdecode(mask_np, cv2.IMREAD_UNCHANGED)
                except Exception as e:
                    print(f"Error decoding mask image from response: {e}")
                    mask_image = None

        if remote_status == "success":
            # mask_image and center_coordinate should be populated if successful
            pass
        elif remote_status == "warning" and "tracking lost" in remote_message.lower():
            # Extract consecutive_lost_frames from the message
            # Example message: "SAM returned empty mask, tracking lost for 5 consecutive frames"
            try:
                # Find the number before "consecutive frames"
                match = re.search(r"for (\d+) consecutive frames", remote_message)
                if match:
                    consecutive_lost_frames = int(match.group(1))
                    if consecutive_lost_frames >= self.REPROMPT_THRESHOLD:
                        recommand_reprompt = 1
            except Exception as e:
                print(f"Error parsing consecutive_lost_frames from message: {e}")
            
            # For tracking lost, mask_image and center_coordinate might be None
            mask_image = None
            center_coordinate = None

        elif remote_status == "failure":
            # For failure, mask_image and center_coordinate are None
            mask_image = None
            center_coordinate = None
            recommand_reprompt = 0 # No specific re-prompt recommendation for general failure

        return mask_image, center_coordinate, recommand_reprompt



    def set_prompts(self, image: np.ndarray, prompts: dict, obj_id: int) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]], int]:
        """
        Sends image and prompts to the remote service to initialize or 
        re-initialize tracking. 
        This method now uses submit_frame to delegate the actual prompt handling
        to the worker process.

        Args:
            image (np.ndarray): The image frame for prompting.
            prompts (dict): Dictionary containing prompt data (e.g., "points", "labels").
            obj_id (int): The ID of the object to track.
        Returns:
            Tuple[Optional[np.ndarray], Optional[Tuple[int, int]], int]:
            - mask_image (Optional[np.ndarray]): Decoded mask image, or None on failure.
            - center_coordinate (Optional[Tuple[int, int]]): (x, y) center coordinate, or None on failure.
            - recommand_reprompt (int): 0 for no re-prompt, 1 for re-prompt recommended.
        """
        # Check if shared variables are initialized
        if self._is_tracking_active is None or self._current_obj_id is None:
            print(f"SAM2RTRemoteAnalyzer process ({mp.current_process().pid}): Shared tracking state not initialized. Call NNAnalyzer.start() first.")
            return None, None, 0

        # Submit the prompt command to the worker process
        submission_timestamp = self.submit_frame(
            frame=image,
            target='_handle_set_prompts',
            prompts=prompts,
            obj_id=obj_id
        )

        if submission_timestamp is None:
            print(f"SAM2RTRemoteAnalyzer process ({mp.current_process().pid}): Failed to submit prompt frame.")
            return None, None, 0
        
        # Get the result from the worker process
        result = self.get_result(timeout=self.prompt_timeout) # Add a buffer to timeout

        if result:
            (mask_image, center_coord, recommand_reprompt), submission_timestamp, analysis_duration = result
            return mask_image, center_coord, recommand_reprompt
        else:
            print(f"SAM2RTRemoteAnalyzer process ({mp.current_process().pid}): No result received for set_prompts (timeout or error).")
            return None, None, 0

    def _check_service_ready(self) -> bool:
        """
        Checks if the remote SAM2RT service is ready.
        This method is called in the worker process during initialization.
        Returns:
            bool: True if the service is ready, False otherwise.
        """
        ready_url = f"http://{self.remote_address}/ready"
        try:
            # Use a short timeout for readiness check
            # Check if request session is initialized
            if self._request_session is None:
                print(f"SAM2RTRemoteAnalyzer worker: Request session not initialized.")
                return False
            response = self._request_session.get(ready_url, timeout=1)
            response.raise_for_status()
            json_response = response.json()
            return json_response.get("status") == "ready"
        except requests.exceptions.RequestException as e:
            print(f"SAM2RTRemoteAnalyzer worker: Error checking service readiness at {ready_url}: {e}")
            return False
        except Exception as e:
            print(f"SAM2RTRemoteAnalyzer worker: Unexpected error during service readiness check: {e}")
            return False
    
    def stop(self):
        """
        Stops the SAM2RTRemoteAnalyzer.
        Ensures the request session is closed before calling the base class stop.
        """
        if self._request_session:
            self._request_session.close()
            self._request_session = None
        super().stop()

if __name__ == "__main__":
    print("SAM2RTRemoteAnalyzer Example Usage (Main Thread)")

    # 1. Configuration
    REMOTE_ADDRESS = "222.29.33.185:5000" # Assuming the remote service is running locally
    # TODO: Disable Proxy in code?
    
    # Example frame dimensions (replace with actual camera/image dimensions)
    IMG_HEIGHT = 480
    IMG_WIDTH = 640
    CHANNELS = 3
    
    # Frame shape for NNAnalyzer's shared memory buffer (no timecode appended for simplicity in this example)
    INPUT_FRAME_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    
    # Timecode timebase (optional, set to None if not using timecodes)
    TIMECODE_TIMEBASE = 10000 # Example: 10000 ticks per second

    # 2. Create Analyzer Instance
    analyzer = SAM2RTRemoteAnalyzer(
        remote_address=REMOTE_ADDRESS,
        frame_shape=INPUT_FRAME_SHAPE,
        effective_image_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), # Explicitly state effective image shape
        timecode_timebase=TIMECODE_TIMEBASE,
        use_proxy=False, # Example: Disable proxy for this instance
        track_timeout=1,
    )

    # 3. Start the Analyzer (this launches the worker process and initializes it)
    print("\n--- Starting SAM2RTRemoteAnalyzer ---")
    try:
        analyzer.start()
        print("SAM2RTRemoteAnalyzer started successfully.")
    except Exception as e:
        print(f"Error starting analyzer: {e}")
        exit()

    # 4. Create a dummy image for prompting
    # A simple white square on a black background
    dummy_image_prompt = np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.uint8)
    square_size = 50
    center_x, center_y = IMG_WIDTH // 2, IMG_HEIGHT // 2
    dummy_image_prompt[center_y - square_size//2 : center_y + square_size//2,
                       center_x - square_size//2 : center_x + square_size//2] = [255, 255, 255]
    
    # Define prompt points (center of the square)
    prompt_points = [[center_x, center_y]]
    prompt_labels = [1] # 1 for foreground
    obj_id = 1 # Object ID to track

    prompts_data = {
        "points": prompt_points,
        "labels": prompt_labels
    }

    # 5. Call set_prompts to initialize tracking
    print("\n--- Calling set_prompts to initialize tracking ---")
    mask_image_prompt, center_coord_prompt, recommand_reprompt_prompt = analyzer.set_prompts(
        image=dummy_image_prompt,
        prompts=prompts_data,
        obj_id=obj_id
    )

    if mask_image_prompt is not None:
        print(f"Prompt successful. Mask shape: {mask_image_prompt.shape}, Center: {center_coord_prompt}")
        cv2.imwrite("prompt_mask_result.png", mask_image_prompt)
        print("Prompt mask saved to prompt_mask_result.png")
    else:
        print("Prompt failed or returned no mask.")
        analyzer.stop()
        exit()

    # 6. Simulate submitting frames for tracking
    print("\n--- Simulating frame submission for tracking ---")
    num_frames_to_track = 5
    for i in range(num_frames_to_track):
        # Create a dummy image for tracking (e.g., slightly shifted)
        dummy_image_track = np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.uint8)
        shift_x = 5 * i
        shift_y = 5 * i
        dummy_image_track[center_y - square_size//2 + shift_y : center_y + square_size//2 + shift_y,
                           center_x - square_size//2 + shift_x : center_x + square_size//2 + shift_x] = [255, 255, 255]

        # Embed a dummy timecode (optional, if timecode_timebase is set)
        current_timecode = int(time.time() * TIMECODE_TIMEBASE)
        # For simplicity, we're not actually embedding timecode into the image data here,
        # as NNAnalyzer's submit_frame appends it.
        # If you need to test timecode extraction from image, you'd need to modify dummy_image_track.

        # Submit frame to the analyzer
        print(f"Submitting frame {i+1}...")
        submission_timestamp = analyzer.submit_frame(dummy_image_track)
        
        if submission_timestamp is not None:
            # Get result from the analyzer
            result = analyzer.get_result(timeout=0.5) # Wait for up to 10 seconds for result
            if result:
                (mask_image_track, center_coord_track, recommand_reprompt_track), submission_timestamp, analysis_duration = result
                print(f"Frame {i+1} analysis result:")
                if mask_image_track is not None:
                    print(f"  Mask shape: {mask_image_track.shape}, Center: {center_coord_track}")
                    cv2.imwrite(f"track_mask_result_{i+1}.png", mask_image_track)
                    print(f"  Track mask saved to track_mask_result_{i+1}.png")
                else:
                    print("  Tracking returned no mask.")
                print(f"  Re-prompt recommended: {recommand_reprompt_track}")
                if submission_timestamp is not None:
                    print(f"  Retrieved timestamp: {submission_timestamp}")
                else:
                    print("  Timecode not available.")
            else:
                print(f"Frame {i+1} analysis: No result received (timeout or error).")
        else:
            print(f"Frame {i+1} submission failed (timeout or analyzer not ready).")
        time.sleep(0.1) # Simulate some delay between frames

    # 7. Stop the Analyzer
    print("\n--- Stopping SAM2RTRemoteAnalyzer ---")
    analyzer.stop()
    print("SAM2RTRemoteAnalyzer stopped.")

    print("\nSAM2RTRemoteAnalyzer Example Usage Finished.")
