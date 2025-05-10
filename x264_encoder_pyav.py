import abc
import multiprocessing as mp
import multiprocessing.shared_memory as mp_shm
import multiprocessing.synchronize as mp_sync
import numpy as np
from fractions import Fraction
from typing import Tuple, Any, Optional, List
from shared_ring_buffer import ProcessSafeSharedRingBuffer
from videoencoder import BaseVideoEncoder # Import BaseVideoEncoder

# Add imports for subprocess, sys, threading, and os here
import av
import time # Import time for the example

# Confirming file state after user feedback
class X264Encoder(BaseVideoEncoder):
    """
    Concrete implementation of BaseVideoEncoder for x264 encoding using PyAV.
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

        # Attributes to hold pyav container and stream
        self._container: Optional[av.container.Container] = None
        self._stream: Optional[av.stream.Stream] = None


    def _initialize_encoder(self):
        """
        Initializes the x264 encoder using PyAV.
        This runs in the worker process.
        """
        print(f"X264Encoder worker ({mp.current_process().pid}): Initializing x264 encoder with PyAV...")

        height, width, channels = self._frame_size

        try:
            # Open the output file for writing
            self._container = av.open(self._output_path, mode='w')

            # Add a video stream using libx264
            self._stream = self._container.add_stream('libx264', rate=Fraction(self._fps).limit_denominator())
            self._stream.width = width
            self._stream.height = height
            self._stream.pix_fmt = 'yuv444p' # H.264 encoding typically uses yuv420p

            # Set encoder options
            options = {
                'preset': self._preset,
                'threads': str(self._threads),
            }
            if self._crf is not None:
                options['crf'] = str(self._crf)
            elif self._bitrate is not None:
                options['b:v'] = str(self._bitrate) # Use 'b:v' for bitrate

            self._stream.options = options

            print(f"X264Encoder worker ({mp.current_process().pid}): PyAV encoder initialized.")

        except Exception as e:
            print(f"FATAL: Error initializing PyAV encoder: {e}")
            self._container = None
            self._stream = None
            raise # Re-raise the exception to indicate initialization failure


    def _encode_frames(self, frames: List[np.ndarray]):
        """
        Encodes a batch of frames using the initialized PyAV encoder.
        This runs in the worker process.
        Args:
            frames: (List[np.ndarray[f,h,w,c]]), a list of input frames to encode.
        """
        if not frames:
            return # Nothing to encode

        total_from_ndarray_time = 0
        total_encode_time = 0
        total_mux_time = 0
        total_frames_processed = 0

        if self._container and self._stream:
            try:
                # frames is List[np.ndarray].
                # Each np.ndarray in the list is a batch of frames, typically shape (N, H, W, C).
                for frame_batch_ndarray in frames:
                    # Iterate through each frame in the current ndarray batch
                    for i in range(frame_batch_ndarray.shape[0]):
                        total_frames_processed += 1
                        single_frame_data = frame_batch_ndarray[i] # This is (H, W, C)
                        
                        # Time VideoFrame.from_ndarray()
                        start_time = time.perf_counter()
                        av_frame = av.VideoFrame.from_ndarray(single_frame_data, format='bgr24')
                        total_from_ndarray_time += (time.perf_counter() - start_time)

                        # Time stream.encode()
                        start_time = time.perf_counter()
                        packets = list(self._stream.encode(av_frame)) # Materialize to measure encode time accurately
                        total_encode_time += (time.perf_counter() - start_time)

                        # Time muxing
                        start_time = time.perf_counter()
                        for packet in packets:
                            # Mux the packet into the container
                            self._container.mux(packet)
                        total_mux_time += (time.perf_counter() - start_time)
                
                if total_frames_processed > 0:
                    print(f"\n--- Internal Timings for _encode_frames ({total_frames_processed} frames) ---")
                    print(f"Total time in VideoFrame.from_ndarray: {total_from_ndarray_time:.4f}s (Avg: {total_from_ndarray_time/total_frames_processed:.6f}s/frame)")
                    print(f"Total time in stream.encode:          {total_encode_time:.4f}s (Avg: {total_encode_time/total_frames_processed:.6f}s/frame)")
                    print(f"Total time in container.mux:          {total_mux_time:.4f}s (Avg: {total_mux_time/total_frames_processed:.6f}s/frame)")
                    print(f"Combined internal avg per frame:      {(total_from_ndarray_time + total_encode_time + total_mux_time)/total_frames_processed:.6f}s/frame\n")


            except Exception as e:
                print(f"X264Encoder worker ({mp.current_process().pid}): Error encoding or muxing frame with PyAV: {e}")
                # The worker loop in the base class will handle exiting if _running is cleared

        else:
            print("Error: PyAV container or stream not available in _encode_frames.")
            # The worker loop in the base class will handle exiting if _running is cleared


    def _uninitialize_encoder(self):
        """
        Uninitializes the PyAV encoder by flushing and closing the container.
        This runs in the worker process.
        """
        print(f"X264Encoder worker ({mp.current_process().pid}): Uninitializing x264 encoder with PyAV...")

        if self._container and self._stream:
            try:
                # Flush the encoder
                print(f"X264Encoder worker ({mp.current_process().pid}): Flushing PyAV encoder...")
                for packet in self._stream.encode(): # Call encode() with no args to flush
                    self._container.mux(packet)

                # Close the container
                print(f"X264Encoder worker ({mp.current_process().pid}): Closing PyAV container...")
                self._container.close()
                print(f"X264Encoder worker ({mp.current_process().pid}): PyAV container closed.")

            except Exception as e:
                print(f"Error during PyAV uninitialization: {e}")

            finally:
                self._container = None # Clear references
                self._stream = None
        else:
            print("Warning: PyAV container or stream not available during uninitialization.")

        print(f"X264Encoder worker ({mp.current_process().pid}): X264 encoder uninitialized.")


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
        'threads': 0,
        # "x264-params": "cabac=1:ref=3:deblock=1:0:0:analyse=0x3:0x113:me=hex:subme=7:psy=1:psy_rd=1.00:0.00:mixed_ref=1:me_range=16:chroma_me=1:trellis=1:8x8dct=1:cqm=0:deadzone=21,11:fast_pskip=1:chroma_qp_offset=4:threads=15:lookahead_threads=2:sliced_threads=0:nr=0:decimate=1:interlaced=0:bluray_compat=0:constrained_intra=0:bframes=3:b_pyramid=2:b_adapt=1:b_bias=0:direct=1:weightb=1:open_gop=0:weightp=2:keyint=250:keyint_min=25:scenecut=40:intra_refresh=0:rc_lookahead=40:rc=crf:mbtree=1:crf=23.0:qcomp=0.60:qpmin=0:qpmax=69:qpstep=4:ip_ratio=1.40:aq=1:1.00",
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
