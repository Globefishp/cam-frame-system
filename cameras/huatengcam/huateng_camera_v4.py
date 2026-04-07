# Gemini 3.1 Pro rewrite, not tested.

# Timecode related part is a mess, extract_tc_from_frames is called from 
# both analyzer and encoder.
# By lastest AbstractCamera API, should pass a handle of AbstractCamera Obj
# to analyzer and encoder to handle timecode, rather than depends on 
# specific camera impl.

# Several module should be refactored, that's not my priority for huateng camera.

import sys
from pathlib import Path
import numpy as np
import platform
import ctypes
import time
import pickle
from typing import Optional, Tuple, Union, Any, List

from . import mvsdk_mod as mvsdk
from cameras.huatengcam.mvsdk_mod import CameraException as MvCamException
from .extensions.unpack_12bit_raw import unpack_12bit_to_16bit_fast
from .extensions import PrecisionTimer
from .extensions.raw_processing_cy_V11 import RawV11Processor
import warnings

from ..abstractcamera import AbstractCamera as AC
from ..abstractcamera import CameraFeatures  # 引入您的基类和枚举
from .huatengcam_types import HuatengSDKException, TriggerMode, BitDepth, BayerPattern
from loguru import logger

NDArray = np.ndarray

# Default values
_FRAME_TIME = 10
_GAIN = 1.0
_BAYER_PATTERN = BayerPattern.BGGR
_HUATENGCAM_CURR_DIR = Path(__file__).resolve().parent
_DEFAULT_CORRECTION_PATH: Path = _HUATENGCAM_CURR_DIR / "corrections/correction_results_D50_250805.npy"

_EXTRA_ROWS_FOR_METADATA = 1

class HuatengCamera(AC):
    """
    华腾相机 (适配 AbstractCamera 接口)
    hibitdepth: 0: 8bit, 1: 12bit(Packed模式:实际占用1.5bpp)
    """

    @staticmethod
    def enumerate_cameras() -> List[mvsdk.tSdkCameraDevInfo]:
        """Enumerate Huateng Cameras by mvsdk.CameraEnumerateDevice()"""
        DevList: List[mvsdk.tSdkCameraDevInfo] = mvsdk.CameraEnumerateDevice()
        logger.info(f"Enumerated {len(DevList)} cameras.")
        return DevList

    def __init__(self,
                 DevInfo: mvsdk.tSdkCameraDevInfo,
                 fps: Optional[float] = None, # None=freerun
                 bitdepth: BitDepth = BitDepth._8,
                 bayer_pattern: BayerPattern = _BAYER_PATTERN, # For Our ISP?
                 ):
        """Initialize HuatengCamera chosen by DevInfo
        
        :param DevInfo: mvsdk.tSdkCameraDevInfo object chosen by enumerate_cameras()
        :param fps: Target FPS, If `None`, the camera will run in freerun mode.
        :param hibitdepth: Define the raw bit depth grabbed, affect internal processing.
            0 = 8bit, 1 = 12bit(if possible)
        """
        self._features_enabled = CameraFeatures.TIMECODE | CameraFeatures.GAIN

        self._DevInfo: mvsdk.tSdkCameraDevInfo = DevInfo
        self._hCamera: int = 0 # Camera handle in sdk.
        self._cap: Optional[mvsdk.tSdkCameraCapbility] = None
        self._pFrameBuffer: Optional[ctypes.c_void_p] = None
        self._frames_captured: int = 0
        self._is_capturing: bool = False

        self._trigger_mode: TriggerMode
        if fps is None:
            self._trigger_mode = TriggerMode.FREERUN
        else:
            self._trigger_mode = TriggerMode.SOFT_TRIGGER
        self._timer: Optional[PrecisionTimer.PrecisionTimer] = None # PrecisionTimer for SW trigger
        self._target_fps: Optional[float] = fps
        self._exposure_time_ms: float = _FRAME_TIME
        self._gain: float = _GAIN

        self._image_width: Optional[int] = None
        self._image_height: Optional[int] = None
        self._image_channels: Optional[int] = None
        self._bit_depth: BitDepth = bitdepth # 在SDK中被称为media_type，详见open时枚举。
        self._bayer_pattern: BayerPattern = bayer_pattern

        self._correction_path = _DEFAULT_CORRECTION_PATH
        self._XYZ_TO_SRGB: NDArray = np.array(
            [[ 3.2404542, -1.5371385, -0.4985314],
             [-0.9692660,  1.8760108,  0.0415560],
             [ 0.0556434, -0.2040259,  1.0572252]])

        self._processor: Optional[RawV11Processor] = None
        
        self._extra_rows: int = _EXTRA_ROWS_FOR_METADATA
    
    def _check_last_err(self) -> None:
        """Check last error by calling SDK function and raise exception if any"""
        caller_frame = sys._getframe(1) # get caller.
        caller_func_name = caller_frame.f_code.co_name
        caller_class_name = caller_frame.f_code.__class__.__name__
        if (err := mvsdk.GetLastError()) != 0:
            raise HuatengSDKException(MvCamException(err), src_func=f"{caller_class_name}.{caller_func_name}")

    # === Camera Lifecycle ===
    def open(self, target_media_type: Optional[int] = None) -> bool:
        """
        Open the camera.

        :param target_media_type: Force target media type index. Default is None (auto detect).

        :return: True if success.
        :raises HuatengSDKException: when failed.
        """
        if self.is_opened():
            return True
        logger.info(f"Opening camera {self._DevInfo.GetFriendlyName()} ({self._DevInfo.GetProductName()})")
        try:
            hCamera = mvsdk.CameraInit(self._DevInfo, -1, -1)
        except MvCamException as e:
            raise HuatengSDKException(e, src_func="HuatengCamera.open") from e
        self._hCamera = hCamera

        self._cap = mvsdk.CameraGetCapability(hCamera)

        ResolutionRange: mvsdk.tSdkResolutionRange = self._cap.sResolutionRange
        monoCamera = (self._cap.sIspCapacity.bMonoSensor != 0)

        self._image_width: int = ResolutionRange.iWidthMax
        self._image_height: int = ResolutionRange.iHeightMax
        self._image_channels: int = 1 if monoCamera else 3

        # # SDK ISP parameter
        # if monoCamera:
        #     mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        # else:
        #     mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
        
        # Get bitdepth supported & Set bit depth
        logger.info(f"Camera supports {self._cap.iMediaTypeDesc} pixel format(s)")
        for i in range(self._cap.iMediaTypeDesc):
            logger.info(f"Pixel format {self._cap.pMediaTypeDesc[i].iIndex}: {self._cap.pMediaTypeDesc[i].GetDescription()}")
        # SDK目前支持8/12/16; hibitdepth就当作是12bit。
        if target_media_type is None:
            if self._bit_depth == BitDepth._8:
                logger.info(f"Using 8bit pixel format. Targeting Index 0")
                mvsdk.CameraSetMediaType(hCamera, 0) # 一般来说是索引0
            elif self._bit_depth == BitDepth._12:
                logger.info(f"Using 12bit pixel format. Targeting Index 1")
                mvsdk.CameraSetMediaType(hCamera, 1)
        else:
            logger.info(f"Using specified media format, Index {target_media_type}")
            mvsdk.CameraSetMediaType(hCamera, target_media_type)
        
        mvsdk.CameraSetRawStartBit(hCamera, -1) # Get full dynamic range

        # Set trigger mode
        if self._trigger_mode == TriggerMode.SOFT_TRIGGER:
            mvsdk.CameraSetTriggerMode(hCamera, 1)  # 软件触发
            mvsdk.CameraSetTriggerCount(hCamera, 1) # 每次触发1帧
            interval_s = 1.0 / self._target_fps
            self._timer = PrecisionTimer.PrecisionTimer(
                interval_s=interval_s,
                c_trigger_func=mvsdk._sdk.CameraSoftTrigger,
                hCamera=hCamera,
                busy_wait_us=2000,
                priority=2
            )
            self._timer.start()
        elif self._trigger_mode == TriggerMode.FREERUN:
            mvsdk.CameraSetTriggerMode(hCamera, 0)

        # Camera "Opened", call property setters
        mvsdk.CameraSetAeState(hCamera, 0) # 手动曝光
        self.exposure_time_ms = self._exposure_time_ms
        self.gain = self._gain

        # Initialize Aux components
        self._init_raw_processor()

        logger.success(f"Camera {self._DevInfo.GetFriendlyName()} ({self._DevInfo.GetProductName()}) opened.")
        return True
    def _init_raw_processor(self):
        # Initialize Hibitdepth Raw Processor
        if self._correction_path.exists():
            correction_info = np.load(self._correction_path, allow_pickle=True).item()
        else:
            logger.warning(f"Color correction file not found: {self._correction_path}, "
                           "color correction disabled.")
            correction_info['wb_params'] = np.array((1,1,1,0,0,0), dtype=np.float64)
            correction_info['fwd_mtx'] = np.eye(3, dtype=np.float64)
        self._processor = RawV11Processor(
            self.height, self.width, black_level=32,
            ADC_max_level=4094 if self._bit_depth == BitDepth._12 else 255,
            bayer_pattern='BGGR' if self._bayer_pattern == BayerPattern.BGGR else 'RGGB',
            wb_params=correction_info['wb_params'],
            fwd_mtx=correction_info['fwd_mtx'],
            render_mtx=self._XYZ_TO_SRGB,
            gamma='BT709'
        )

    def close(self) -> bool:
        if not self.is_opened():
            return True
        if self._timer is not None:
            self._timer.stop()
            self._timer.join()
            self._timer = None

        if self.is_capturing():
            self.stop_capture()

        if self.is_opened():
            mvsdk.CameraUnInit(self._hCamera)
            self._hCamera = 0

        if self._pFrameBuffer != 0:
            mvsdk.CameraAlignFree(self._pFrameBuffer)
            self._pFrameBuffer = 0
        logger.success(f"Camera {self._DevInfo.GetFriendlyName()} ({self._DevInfo.GetProductName()}) closed.")
        return True
    
    def is_opened(self) -> bool:
        if self._hCamera > 0:
            return mvsdk.CameraIsOpened(self._DevInfo)
        else:
            return False
    
    @AC.require_open
    def start_capture(self):
        # Buffer allocation
        # always preserve extra rows regardless of TIMECODE enable.
        # 分配的空间总是有余量, 按照RGB+extra line来分配, 对于raw的话extra info实际上用不了3通道. 
        try:
            self._pFrameBuffer = mvsdk.CameraAlignMalloc(self.frame_size_bytes, 16)

            mvsdk.CameraRstTimeStamp(self._hCamera) # 重置时间戳
            mvsdk.CameraPlay(self._hCamera)
        except MvCamException as e:
            raise HuatengSDKException(e, src_func="HuatengCamera.start_capture") from e
        self._is_capturing = True
        logger.info(f"Camera {self._DevInfo.GetProductName()} started capturing.")
    
    @AC.require_open
    def stop_capture(self):
        try:
            mvsdk.CameraStop(self._hCamera)
        except MvCamException as e:
            raise HuatengSDKException(e, src_func="HuatengCamera.stop_capture") from e
        self._is_capturing = False
        logger.info(f"Camera {self._DevInfo.GetProductName()} stopped capturing.")
    
    def is_capturing(self) -> bool:
        return self._is_capturing

    # ==========================
    # 能力与属性查询
    # ==========================
    @property
    def features(self) -> CameraFeatures: 
        return CameraFeatures.TIMECODE | CameraFeatures.GAIN

    @property
    def features_enabled(self) -> CameraFeatures: return self._features_enabled

    @property
    def width(self) -> int: return self._image_width
    @property
    def height(self) -> int: return self._image_height
    @property
    def full_width(self) -> int: return self._image_width
    @property
    def full_height(self) -> int: return self._image_height

    @property
    def channels(self) -> int: return self._image_channels

    @property
    def dtype(self) -> np.dtype: 
        return np.dtype("uint8") if self._bit_depth == BitDepth._8 else np.dtype("uint16")

    @property
    def target_fps(self) -> float:
        if self._trigger_mode == TriggerMode.SOFT_TRIGGER:
            return self._target_fps
        else:
            return 1000.0 / self.exposure_time_ms 
            # TODO: Freerun, max fps cannot be acquired from SDK...
        
    @property
    @AC.require_open
    def frame_count(self) -> int:
        return self._frames_captured

    @property
    @AC.require_open
    def actual_fps(self) -> float:
        # TODO: deprecated?
        """A proxy, not real FPS..."""
        if self._trigger_mode == TriggerMode.SOFT_TRIGGER:
            return self._target_fps
        else:
            return 1000.0 / self.exposure_time_ms
        
    @property
    def frame_count(self) -> int:
        return self._frames_captured

    @property
    def exposure_time_ms(self) -> float:
        """Return the exposure time in ms, if the camera is not opened, 
        return the cached/default exposure time."""
        if self.is_opened():
            exposure_time_ms = mvsdk.CameraGetExposureTime(self._hCamera) / 1000.0
            self._check_last_err()
            self._exposure_time_ms = exposure_time_ms # Cache value
        return self._exposure_time_ms
    @exposure_time_ms.setter
    def exposure_time_ms(self, exposure_ms: float):
        """Set the exposure time in ms, if the camera is not opened, 
        cache the setting value for new default."""
        self._exposure_time_ms = exposure_ms
        if self.is_opened():
            mvsdk.CameraSetExposureTime(self._hCamera, exposure_ms * 1000)
            self._check_last_err()
            self._exposure_time_ms = mvsdk.CameraGetExposureTime(self._hCamera) / 1000.0 # Cache real value
        logger.info(f"Requested exposure time {exposure_ms:.2f} ms has been set to {self._exposure_time_ms:.2f} ms.")
    @property
    @AC.require_open
    def exposure_time_ms_range(self):
        exp_range = mvsdk.CameraGetExposureTimeRange(self._hCamera)
        self._check_last_err()
        return exp_range[0] / 1000.0, exp_range[1] / 1000.0

    @property
    def gain(self) -> float:
        """Return the analog gain, if the camera is not opened, 
        return the cached/default gain."""
        if self.is_opened():
            gain = mvsdk.CameraGetAnalogGainX(self._hCamera)
            self._check_last_err()
            self._gain = gain # Cache value
        return self._gain
    @gain.setter
    def gain(self, gain: float):
        """Set the analog gain, if the camera is not opened, 
        cache the setting value for new default."""
        self._gain = gain
        if self.is_opened():
            mvsdk.CameraSetAnalogGainX(self._hCamera, gain)
            self._check_last_err()
            self._gain = mvsdk.CameraGetAnalogGainX(self._hCamera) # Cache real value
        logger.info(f"Requested gain {gain:.2f} has been set to {self._gain:.2f}.")

    @property
    @AC.require_open
    def gain_range(self):
        gain_range = mvsdk.CameraGetAnalogGainXRange(self._hCamera)
        self._check_last_err()
        return gain_range[0:2]

    @property
    def hw_timecode_timebase(self) -> float:
        return 10000.0  # 0.1ms

    def get_hw_timecode(self) -> int:
        raise NotImplementedError("Huateng SDK only provides timecode attached to fetched frames.")

    # === Grab Frame ===
    def _grab_extendedbuf_metadata(self) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Grab a raw frame with extended lines, and return with metadata.
        If 12bit raw is enabled, NDArray is np.uint16, otherwise np.uint8.

        :return: (raw_data_np, timecode_val) if success, (None, None) otherwise.
            raw_data_np (NDArray) has shape (H + extra_lines, W).
            timecode_val (int) is the HW timecode value in 0.1ms.
        """
        if self.is_opened() == False or self._pFrameBuffer == 0: # Fully Initialized
            return None, None
        
        # property access use internal cache, rather than `property`.
        try:
            # Timeout = 2 * exposure time.
            timeout = max(200, int(self._exposure_time_ms * 2))
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self._hCamera, timeout) 
            timecode_val: int = FrameHead.uiTimeStamp

            # Raw数据实际仅占用buffer前部一部分空间
            # Create a numpy array view of RawData buffer.
            # Image size before debayer is HxWx(Bpp), for 12bit packed, Bpp=1.5
            if self._bit_depth == BitDepth._8:
                raw_data_ctypes = (mvsdk.c_ubyte * (
                    self._image_width * (self._image_height + self._extra_rows))).from_address(pRawData)
            elif self._bit_depth == BitDepth._12: # 12bit = 1.5Bpp
                raw_data_ctypes = (mvsdk.c_ubyte * (
                    self._image_width * self._image_height * 3 // 2)).from_address(pRawData)

            # No need for copy here, see speed test below.
            raw_data_np = np.frombuffer(raw_data_ctypes, dtype=np.uint8)

            if self._bit_depth == BitDepth._8:
                raw_data_np = raw_data_np.copy().reshape((self._image_height + self._extra_rows), self._image_width)
            elif self._bit_depth == BitDepth._12:
                raw_data_np = unpack_12bit_to_16bit_fast(raw_data_np, 
                                                         self._image_height, self._image_width,
                                                         extra_height=self._extra_rows)
                # <3ms @ 2448*2048, memory-bound, throughput~7GB/s, FYI: np.copy(): 3.53ms/frame
                # TODO: Fuse it to raw_processing pipeline to reduce copy.
            
            mvsdk.CameraReleaseImageBuffer(self._hCamera, pRawData)

        except MvCamException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                raise HuatengSDKException(e, "Failed to grab frame", extra_info="HuatengCamera.grab_metadata") from e
            else:
                logger.warning(f"Timeout to grab frame ({self._frames_captured + 1}).")
                return None, None
        
        self._frames_captured += 1
        return raw_data_np, timecode_val # timestamp is always supported in HuatengCam
    
    def grab(self) -> Optional[np.ndarray]:
        # TODO: Impl. ISP here
        frame = self.grab_raw()
        if frame is None: return None
        frame = self._processor.process(frame)
        return frame


    def grab_raw(self) -> Optional[np.ndarray]:
        frame, _ = self._grab_extendedbuf_metadata()
        frame = self.strip_extended_info(frame)
        return frame

    def grab_metadata(self) -> Tuple[Optional[NDArray], Any]:
        frame, tc_val = self._grab_extendedbuf_metadata()
        if frame is None: return None, {}
        frame = self.strip_extended_info(frame)
        return frame, {'hw_timecode': tc_val}

    def grab_extended_info(self) -> Optional[np.ndarray]:
        frame, tc_val = self._grab_extendedbuf_metadata()
        if frame is None: return None

        # Write metadata to extra_line, 
        print(tc_val)
        metadata = HuatengCamera.HuatengCamMetadata(hw_timecode=tc_val)
        frame = self._append_metadata_to_image(frame, metadata)

        return frame
    
    # === Metadata related ===

    # Private use, will be packed into extractor.
    class HuatengCamMetadata(ctypes.Structure):
        _pack_ = 1 # 1 byte alignment
        _fields_ = [
            ("hw_timecode", ctypes.c_uint32),
        ]

    def _append_metadata_to_image(self, image: NDArray, metadata: HuatengCamMetadata) -> NDArray:
        """
        Append metadata to the image with extra lines. To make the process
        zero-copy, the input image should be C-contiguous.

        :param image: Input image with shape (H + extra_lines, W).
        :type image: NDArray
        :param metadata: Metadata to append.
        :type metadata: HuatengCamMetadata
        :return: Image with metadata appended. (if zero-copy, same ptr as input)
        :rtype: NDArray
        """
        length = ctypes.sizeof(metadata)

        extra_lines = self.extra_lines
        if extra_lines == 0:
            return image

        extra_rows_np = image[-extra_lines:]
        extra_byte_view = extra_rows_np.view(np.uint8).ravel() # C-contiguous to avoid copy.

        if length > len(extra_byte_view):
            raise RuntimeError(f"Metadata bytes ({length}B) is too large to fit in "
                         f"{extra_lines} extra lines ({len(extra_byte_view)}B)."
                        "Metadata is truncated.")
        # TODO: pickled data usually have > 20B/field + 20B baseline, so if a 
        # camera crop ROI, 1 line may be insufficient to store the metadata. 
        # Need more robust mechanism to set extra_lines.
        # Currently no HW ROI implemented.
        # Currently use ctypes.Structure to save space.
        
        # Data layout: HuatengCamMetadata
        extra_byte_view[:length] = np.frombuffer(metadata, dtype=np.uint8)

        return image
        

    def extract_extended_info(self, image: NDArray) -> dict[str, Any]:
        return self._extract_extended_info(image, self.extra_lines, self.__class__.HuatengCamMetadata)
    def strip_extended_info(self, image):
        return super().strip_extended_info(image)
    
    @staticmethod
    def _extract_extended_info(image: NDArray, 
            extra_lines: int, metadata_class: type[ctypes.Structure]) -> dict[str, Any]:
        """
        Extract metadata from the image

        :param image: Input image with shape (H + extra_lines, W).
        :type image: NDArray
        :param extra_lines: Number of extra lines reserved for metadata.
        :type extra_lines: int
        :param metadata_class: Metadata class to extract.
        :type metadata_class: type[ctypes.Structure]
        :return: Metadata dictionary.
        :rtype: dict[str, Any]
        """
        if image.shape[0] <= extra_lines or extra_lines <= 0:
            # No extra lines to extract.
            return {}
        extra_rows = image[-extra_lines:, ...]
        extra_byte_view = extra_rows.view(np.uint8).ravel()
        
        metadata_struct: ctypes.Structure = metadata_class()
        # dst_addr = ctypes.addressof(metadata_struct)
        if len(extra_byte_view) >= ctypes.sizeof(metadata_class):
            # ctypes.memmove(dst_addr, extra_byte_view.ctypes.data, ctypes.sizeof(metadata_class))
            metadata_struct = metadata_class.from_buffer_copy(extra_byte_view)

        # Translate back to dict.
        return {field[0]: getattr(metadata_struct, field[0]) for field in metadata_struct._fields_}
    
    @staticmethod
    def _strip_metadata_from_image(image, extra_lines):
        return image[:-extra_lines, ...]
    
    def _get_decode_ext_info_func(self):
        return self.__class__._extract_extended_info # TODO: a wrapper to enable multi frame processing like V3 API?
    def _get_decode_ext_info_func_kwargs(self):
        return {'extra_lines': self.extra_lines, 'metadata_class': self.__class__.HuatengCamMetadata}
    def _get_strip_ext_info_func(self):
        return super()._get_strip_ext_info_func()
    def _get_strip_ext_info_func_kwargs(self):
        return super()._get_strip_ext_info_func_kwargs()

    @property
    def extra_lines(self) -> int:
        """Return the number of extra lines reserved for timecode."""
        return self._extra_rows

    # deprecated
    @staticmethod
    def unpack_12bit_to_16bit_naive(packed_data: np.ndarray) -> np.ndarray:
        assert packed_data.size % 3 == 0, "Input data size must be a multiple of 3."
        three_byte_chunks = packed_data.reshape(-1, 3).astype(np.uint16)

        byte0, byte1, byte2 = three_byte_chunks[:, 0], three_byte_chunks[:, 1], three_byte_chunks[:, 2]

        pixel0 = (byte0 << 4) | (byte1 & 0x0F)
        pixel1 = (byte2 << 4) | (byte1 >> 4)

        decoded_data = np.empty(pixel0.size + pixel1.size, dtype=np.uint16)
        decoded_data[0::2] = pixel0
        decoded_data[1::2] = pixel1

        return decoded_data
    
if __name__ == '__main__':
    import cv2
    cam_enum = HuatengCamera.enumerate_cameras()
    print(cam_enum)
    cam = HuatengCamera(cam_enum[0], fps=None, bitdepth=BitDepth._12)
    cam.open()
    print(cam.height)
    print(cam.exposure_time_ms_range)
    print(cam.gain_range)
    cam.exposure_time_ms = 10
    cam.gain = 4
    try:
        cam.start_capture()
        frame = cam.grab()
        cv2.imshow("frame", frame[1024:, :])
        cv2.waitKey(0)
        frame = cam.grab_raw()
        print(frame.shape, np.max(frame))
        cv2.imshow("frame", frame[1024:, :]*16)
        cv2.waitKey(0)
        frame, timecode = cam.grab_metadata()
        print(timecode)
        cv2.imshow("frame", frame[1024:, :]*16)
        cv2.waitKey(0)
        frame = cam.grab_extended_info()
        print(frame.shape)
        cv2.imshow("frame", frame[1024:, :]*16)
        cv2.waitKey(0)
        ext_info_extractor = cam.get_extended_info_extractor()
        frame, ext_info = ext_info_extractor(frame)
        print(ext_info)
        print(frame.shape)
        cv2.imshow("frame", frame[1024:, :]*16)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()
    finally:
        cam.stop_capture()
        cam.close()
