import time
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from nnanalyzer import NNAnalyzer
from huateng_camera_tc import extract_tc_from_frames
from sleap_peak_postprocessing import find_local_peaks
from typing import Optional, Tuple # 引入类型提示

class SleapAnalyzer(NNAnalyzer):
    """
    SLEAP model analyzer that inherits from NNAnalyzer.
    Handles model loading, frame preprocessing, inference, and optionally timecode management.
    It can automatically detect if timecode is present based on provided frame shapes.

    Note on frame handling in underlying NNAnalyzer:
    The shared memory IPC in NNAnalyzer uses a single buffer for the input frame.
    If frames are submitted faster than they are analyzed, new frames will overwrite
    previous ones in the shared buffer before analysis. In real-time systems,
    this is often an acceptable or even desired behavior ("favouring latest"),
    especially when timecodes are used, as they allow tracking which specific frame's
    analysis result is returned.
    """
    def __init__(self,
                 model_path: str,
                 frame_shape: tuple[int, int, int], # This is for the NNAnalyzer's shared memory buffer
                 effective_image_shape: Optional[tuple[int, int, int]] = None, # H, W, C of the image itself
                 timecode_timebase: Optional[int] = None, # Denominator of timebase.
                 padding_target_shape: tuple[int, int] = (1024, 1280),
                 model_input_shape: tuple[int, int] = (512, 640),  # TODO: Try to get this from model directly
                 ):
        """
        Initializes the SleapAnalyzer.

        Args:
            model_path (str): Path to the SLEAP model file (.h5).
            frame_shape (tuple[int, int, int]): Shape of the buffer used for shared memory in NNAnalyzer
                                               (H_buffer, W_buffer, C_buffer). This buffer is used to
                                               transfer frame data. It should be large enough to hold
                                               the image data, potentially including appended metadata
                                               like timecodes.
            effective_image_shape (Optional[tuple[int, int, int]]): The actual dimensions (height, width, channels)
                                                                of the image data itself, excluding any
                                                                appended metadata. If None, it's assumed
                                                                that frame_shape implies the effective
                                                                image shape (e.g. by subtracting appended rows).
            padding_target_shape (tuple[int, int], optional): Target shape (height, width) for padding
                                                             the image before resizing.
                                                             Defaults to (1024, 1280).
            model_input_shape (tuple[int, int], optional): Target input shape (height, width)
                                                          for the SLEAP model.
                                                          Defaults to (512, 640).
        """
        super().__init__(model_path, frame_shape) # frame_shape here is for the shared memory buffer
        self.padding_target_shape = padding_target_shape
        self.model_input_shape = model_input_shape
        self.frame_shape = frame_shape # Store for internal use, e.g. getting channels
        self.timecode_timebase = timecode_timebase # Denominator of timebase.
        self._is_timecode_appended = None # Indicating the presence of timecode.

        if effective_image_shape:
            self.effective_image_height = effective_image_shape[0]
            self.effective_image_width = effective_image_shape[1]
            self.channels = effective_image_shape[2]
            # Verify consistency if both are provided
            if frame_shape[0] < self.effective_image_height or \
               frame_shape[1] != self.effective_image_width or \
               frame_shape[2] != self.channels:
                # If frame_shape[0] is exactly effective_image_height, it implies no TC rows in buffer
                # If frame_shape[0] is effective_image_height + APPENDED_ROWS_FOR_TIMECODE, it's consistent for TC
                # Allow frame_shape[0] to be effective_image_height (no TC) or effective_image_height + APPENDED_ROWS (TC)
                raise ValueError(
                    f"Provided frame_shape {frame_shape} is inconsistent with effective_image_shape {effective_image_shape}, "
                    f"where the height of frame_shape ({frame_shape[0]}) should be not less than effective_image_height ({self.effective_image_height}) "
                    f"and other dimension should be the same."
                )
            else:
                self._is_timecode_appended = True
        else:
            # No effective_image_shape provided, assume no timecode appended
            self.effective_image_height = frame_shape[0] # Assume no TC rows appended
            self.effective_image_width = frame_shape[1] # Assume width is consistent
            self.channels = frame_shape[2] # Assume channels are consistent
            self._is_timecode_appended = False # No timecode appended

        # Placeholder for the loaded model, will be initialized in _initialize_analyzer
        # self.model is already defined in NNAnalyzer and initialized to None

    def _initialize_analyzer(self):
        """
        Initializes the analyzer in the worker process.
        Loads the SLEAP model and performs a warmup inference.
        """
        print(f"SleapAnalyzer ({mp.current_process().pid if 'mp' in globals() else 'main'}): Initializing...")
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print(f"SleapAnalyzer ({mp.current_process().pid if 'mp' in globals() else 'main'}): Model loaded from {self.model_path}")
            # TODO: Maybe we can extract model_input_shape and padding_target_shape from model?

            # Perform a warmup inference
            # Create a dummy frame matching the model's expected input dimensions and type
            # The model expects (batch_size, height, width, channels)
            # Channels are typically 3 for RGB images.
            # The dtype after normalization is float32.
            dummy_input_shape = (1, self.model_input_shape[0], self.model_input_shape[1], self.channels) # Assuming same channels as input frame
            dummy_frame = np.zeros(dummy_input_shape, dtype=np.float32)
            
            print(f"SleapAnalyzer ({mp.current_process().pid if 'mp' in globals() else 'main'}): Performing warmup inference...")
            dummy_prediction = self.model.predict(dummy_frame)
            _, _, _, _ = find_local_peaks(dummy_prediction)
            print(f"SleapAnalyzer ({mp.current_process().pid if 'mp' in globals() else 'main'}): Warmup inference complete.")
            print(f"SleapAnalyzer ({mp.current_process().pid if 'mp' in globals() else 'main'}): Initialization complete.")

        except Exception as e:
            print(f"SleapAnalyzer ({mp.current_process().pid if 'mp' in globals() else 'main'}): FATAL Error during initialization: {e}")
            self.model = None # Ensure model is None if loading fails
            raise # Re-raise the exception to signal failure to the NNAnalyzer base class

    def _analyze(self, input_frame: np.ndarray) -> Tuple[Optional[Tuple], Optional[int]]:
        """
        Analyzes a single frame using the loaded SLEAP model.
        Extracts image data and possible timecode (determined by __init__), 
        preprocesses the image, performs inference, and calculates
        the timecode of completion.

        Args:
            frame (np.ndarray): The input frame, could have appended timecode data.
                                                Shape: (H (with_tc), W, C).

        Returns:
            tuple[Tuple, int]: A tuple containing:
                - peak_point: The coordinate of the detected peak (tuple), 
                              or None if no peak is found.
                - new_timecode: The updated timecode (int) after adding analysis duration, 
                                or None if no timecode is provided in input.
        """
        if self.model is None:
            print(f"SleapAnalyzer ({mp.current_process().pid if 'mp' in globals() else 'main'}): Error - Model not loaded. Cannot analyze.")
            # Raise or return None
            # Here, NNAnalyzer's worker loop will skip putting None results in the queue.
            return None, None # Return None

        analysis_start_time = time.perf_counter()

        # 1. Extract image data and original timecode
        original_timecode: Optional[int] = None
        new_timecode: Optional[int] = None
        peak_point: Optional[Tuple] = None # Initialize peak_point

        try:
            # 1. Extract image data and original timecode
            if self._is_timecode_appended:
                original_images_batch, tc_array = extract_tc_from_frames(
                    np.expand_dims(input_frame, axis=0),
                    self.effective_image_height,
                    self.effective_image_width,
                    self.channels
                )
                if tc_array is not None and tc_array.size > 0:
                    original_timecode = int(tc_array[0])
                else:
                    print(f"SleapAnalyzer ({mp.current_process().pid if 'mp' in globals() else 'main'}): Warning - extract_tc_from_frames returned None or empty timecode.")
                    original_timecode = None
                frame_image = original_images_batch[0]
            else:
                frame_image = input_frame[:self.effective_image_height, :, :]
                # Timecodes remain None

        except Exception as e:
            print(f"SleapAnalyzer ({mp.current_process().pid if 'mp' in globals() else 'main'}): Error during timecode/image extraction: {e}")
            # If extraction fails, we cannot proceed with analysis
            return None, None

        try:
            # 2. Preprocess the image (frame_image)
            h, w, _ = frame_image.shape
            pad_h = max(0, self.padding_target_shape[0] - h)
            pad_w = max(0, self.padding_target_shape[1] - w)
            padded_frame = np.pad(frame_image, ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=0)
            expanded_frame = np.expand_dims(padded_frame, axis=0)
            normalized_frame = expanded_frame / 255.0
            input_tensor = tf.image.resize(tf.convert_to_tensor(normalized_frame, dtype=tf.float32), self.model_input_shape)

        except Exception as e:
            print(f"SleapAnalyzer ({mp.current_process().pid if 'mp' in globals() else 'main'}): Error during image preprocessing: {e}")
            return None, None # Cannot proceed if preprocessing fails

        predictions = None # Initialize predictions outside try block
        try:
            # 3. Perform inference
            predictions = self.model.predict(input_tensor)

        except Exception as e:
            print(f"SleapAnalyzer ({mp.current_process().pid if 'mp' in globals() else 'main'}): Error during model inference: {e}")
            # If inference fails, predictions will be None or incomplete, handle in post-processing or return None
            return None, None # Return None if inference fails

        try:
            # 4. Post-processing to get coordinates
            if predictions is not None:
                peak_points, peak_vals, peak_sample_inds, peak_channel_inds = find_local_peaks(predictions)
                if len(peak_points) > 1:
                    print(f"SleapAnalyzer ({mp.current_process().pid if'mp' in globals() else'main'}): Found {len(peak_points)} peaks. Returning the one with max probability.")
                    peak_point = tuple(peak_points[np.argmax(peak_vals)])
                elif len(peak_points) == 1:
                    print(f"SleapAnalyzer ({mp.current_process().pid if'mp' in globals() else'main'}): Found 1 peak.")
                    peak_point = tuple(peak_points[0])
                else:
                    print(f"SleapAnalyzer ({mp.current_process().pid if'mp' in globals() else'main'}): No peaks found.")
                    peak_point = None
            else:
                 # This case should ideally be caught by the inference try block, but as a safeguard:
                print(f"SleapAnalyzer ({mp.current_process().pid if'mp' in globals() else'main'}): Error - Predictions were None after inference.")
                peak_point = None # Ensure peak_point is None

        except Exception as e:
            print(f"SleapAnalyzer ({mp.current_process().pid if'mp' in globals() else'main'}): Error during post-processing: {e}")
            peak_point = None # Ensure peak_point is None if post-processing fails

        # 5. Calculate new timecode (after inference) - This part is less likely to fail
        # but could be included in a broader try block if desired.
        # For now, assuming time calculation itself is robust.
        analysis_end_time = time.perf_counter()
        analysis_duration_seconds = analysis_end_time - analysis_start_time

        if original_timecode is not None and self.timecode_timebase is not None:
             try:
                analysis_duration_ticks = int(analysis_duration_seconds * self.timecode_timebase)
                new_timecode = original_timecode + analysis_duration_ticks
             except Exception as e:
                print(f"SleapAnalyzer ({mp.current_process().pid if 'mp' in globals() else 'main'}): Error calculating new timecode: {e}")
                new_timecode = None
        else:
            print(f"SleapAnalyzer ({mp.current_process().pid if'mp' in globals() else'main'}): No timecode available or timebase not set.")
            new_timecode = None

        # 6. Return predictions and new timecode
        return peak_point, new_timecode

    def _uninitialize_analyzer(self):
        """
        Uninitializes the analyzer in the worker process.
        Currently, no specific cleanup is needed for TensorFlow/Keras models here.
        """
        print(f"SleapAnalyzer ({mp.current_process().pid if 'mp' in globals() else 'main'}): Uninitializing...")
        # TensorFlow/Keras models loaded this way usually don't require explicit cleanup
        # Python's garbage collection handles it when the process exits.
        self.model = None # Clear the model reference
        print(f"SleapAnalyzer ({mp.current_process().pid if 'mp' in globals() else 'main'}): Uninitialization complete.")

if __name__ == '__main__':
    # Example Usage (for testing purposes, typically run by CameraSystem)
    print("SleapAnalyzer Example Usage (Main Thread - for testing structure)")

    # --- Configuration ---
    MODEL_PATH = 'nntest/best_model.h5' # Path to your .h5 model
    
    # Frame dimensions (example, replace with actual dimensions from camera)
    ORIGINAL_IMG_HEIGHT = 720
    ORIGINAL_IMG_WIDTH = 1280
    CHANNELS = 3
    
    # Frame size including the appended row for timecode
    FRAME_HEIGHT_WITH_TC = ORIGINAL_IMG_HEIGHT + 1 # 1 row for appended timecode
    INPUT_FRAME_SHAPE = (FRAME_HEIGHT_WITH_TC, ORIGINAL_IMG_WIDTH, CHANNELS)

    # SLEAP specific parameters (from sleaptest.py or your model's requirements)
    PADDING_TARGET = (1024, 1280) # (height, width) for padding
    MODEL_INPUT = (512, 640)    # (height, width) for model input

    # --- Create Analyzer Instance (simulating how NNAnalyzer base class works) ---
    # In a real scenario, NNAnalyzer.start() would create IPC and a worker process.
    # Here, we'll call methods directly for a simple structural test.

    # Assuming a timebase of 10000 for the test case
    TEST_TIMECODE_TIMEBASE = 10000
    TEST_TIMECODE_DTYPE = np.dtype('uint32') # Example timecode dtype
    
    analyzer = SleapAnalyzer(
        model_path=MODEL_PATH,
        frame_shape=INPUT_FRAME_SHAPE,
        effective_image_shape=(ORIGINAL_IMG_HEIGHT, ORIGINAL_IMG_WIDTH, CHANNELS), # Provide effective shape for TC detection
        timecode_timebase=TEST_TIMECODE_TIMEBASE, # Provide timebase
        padding_target_shape=PADDING_TARGET,
        model_input_shape=MODEL_INPUT
    )

    # --- Test Initialization (simulates what happens in the worker process) ---
    print("\n--- Testing _initialize_analyzer ---")
    try:
        analyzer._initialize_analyzer() # Manually call for testing
        if analyzer.model is not None:
            print("SleapAnalyzer initialized and model loaded successfully.")
        else:
            print("SleapAnalyzer initialization failed or model not loaded.")
            exit()
    except Exception as e:
        print(f"Error during manual initialization test: {e}")
        exit()

    # --- Test Analysis (simulates receiving a frame and analyzing) ---
    print("\n--- Testing _analyze ---")
    # Create a dummy frame with appended timecode data
    dummy_frame_with_tc = np.zeros(INPUT_FRAME_SHAPE, dtype=np.uint8)

    # Embed a dummy timecode (e.g., 12345)
    # Use the TEST_TIMECODE_DTYPE for embedding
    dummy_tc_value = np.array([12345], dtype=TEST_TIMECODE_DTYPE)

    # Embed the timecode bytes directly into the appended row using numpy
    # The appended row starts at index analyzer.effective_image_height
    # The timecode is stored at the beginning of this row.
    # Ensure the appended area is large enough for the timecode bytes.
    timecode_bytes_size = TEST_TIMECODE_DTYPE.itemsize
    appended_area_start_idx = analyzer.effective_image_height * analyzer.effective_image_width * analyzer.channels

    # Create a view of the appended area as uint8 bytes
    appended_area_view = dummy_frame_with_tc.ravel()[appended_area_start_idx:].view(np.uint8)

    # Ensure the view is large enough to hold the timecode bytes
    if appended_area_view.size >= timecode_bytes_size:
        # Copy the timecode bytes into the beginning of the appended area view
        # little-endian as `tobytes()` default
        # TODO: Here the bytestream writing has some error.
        appended_area_view[:timecode_bytes_size] = np.frombuffer(dummy_tc_value.tobytes(), dtype=np.uint8)
        print(f"Embedded dummy timecode {int(dummy_tc_value[0])} into the test frame.")
    else:
        print(f"Warning: Appended area ({appended_area_view.size} bytes) is too small to embed timecode ({timecode_bytes_size} bytes).")


    try:
        start_analysis_test = time.perf_counter()
        predictions, new_timecode = analyzer._analyze(dummy_frame_with_tc)
        end_analysis_test = time.perf_counter()

        if predictions is not None:
            print(f"Analysis successful. Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else 'N/A'}")
            print(f"Time taken for _analyze call: {end_analysis_test - start_analysis_test:.4f} seconds")
        else:
            print("Analysis returned None, check logs for errors.")
        # Check if new_timecode is not None and print it
        if new_timecode is not None:
                print(f"New timecode: {new_timecode}")
                # Example: Check if new_timecode is greater than original (it should be)
                # Note: Comparison is now between Python int and numpy uint32 (dummy_tc_value)
                if new_timecode > int(dummy_tc_value[0]):
                    print("New timecode is greater than original, as expected.")
                else:
                    print("Warning: New timecode is not greater than original.")
        else:
                print("New timecode: None (Timecode not extracted or calculated)")

    except Exception as e:
        print(f"Error during manual analysis test: {e}")
        import traceback
        traceback.print_exc()

    # --- Test Uninitialization ---
    print("\n--- Testing _uninitialize_analyzer ---")
    try:
        analyzer._uninitialize_analyzer()
        if analyzer.model is None:
            print("SleapAnalyzer uninitialized successfully (model set to None).")
        else:
            print("SleapAnalyzer uninitialization might not have fully cleared the model.")
    except Exception as e:
        print(f"Error during manual uninitialization test: {e}")

    print("\nSleapAnalyzer Example Usage Test Finished.")
