# Gemini 3.1 Pro rewrite, not tested.

# Timecode related part is a mess, extract_tc_from_frames is called from 
# both analyzer and encoder.
# By lastest AbstractCamera API, should pass a handle of AbstractCamera Obj
# to analyzer and encoder to handle timecode, rather than depends on 
# specific camera impl.

# Several module should be refactored, that's not my priority for huateng camera.


import numpy as np
import platform
import ctypes
import time
from typing import Optional, Tuple, Union, Any

from . import mvsdk
from .extensions.huatengcam.unpack_12bit_raw import unpack_12bit_to_16bit_fast
from .extensions.huatengcam import PrecisionTimer
from .abstractcamera import AbstractCamera, CameraFeature  # 引入您的基类和枚举
import warnings

FRAME_TIME = 10
GAIN = 1.0
TIMECODE_DTYPE = np.dtype("uint32")
TIMECODE_BYTES = TIMECODE_DTYPE.itemsize
APPENDED_ROWS_FOR_TIMECODE = 1


class HuatengCamera(AbstractCamera):
    """
    华腾相机 (适配 AbstractCamera 接口)
    hibitdepth: 0: 8bit, 1: 12bit(Packed模式:实际占用1.5bpp)
    """

    def __init__(self,
                 DevInfo,
                 exposure_time_ms: float = 10.0,
                 trigger_mode: str = 'freerun',
                 fps: Optional[float] = None,
                 gain: float = GAIN,
                 tc: bool = False,
                 hibitdepth: int = 0,
                 **kwargs):
        if trigger_mode not in ['freerun', 'soft_trigger']:
            raise ValueError("trigger_mode must be 'freerun' or 'soft_trigger'")
        if trigger_mode == 'soft_trigger' and fps is None:
            raise ValueError("fps must be specified for 'soft_trigger' mode.")
        
        self._features_enabled = CameraFeature.TIMECODE if tc else CameraFeature.NONE

        self.DevInfo = DevInfo
        self.hCamera = 0
        self.cap = None
        self.pFrameBuffer = 0
        self.frames_captured = 0

        self.trigger_mode = trigger_mode
        self.timer = None
        self._target_fps = fps
        self._target_exposure_time_ms = exposure_time_ms
        self.acutal_exposure_time_ms = None
        self.target_gain = gain
        self.actual_gain = None

        self.actual_grab_fps = 0.0
        self._last_grab_time = None
        self._ema_alpha = 0.1

        self.image_width = None
        self.image_height = None
        self.image_channels = None
        self.bit_depth = hibitdepth # 在SDK中被称为media_type，详见open时枚举。
        self.pixel_bytes = 1 if hibitdepth == 0 else 2 # 每pixel占用pixel_bytes内存。
        self.image_buffer_size_byte = 0
        
        self.appended_rows = APPENDED_ROWS_FOR_TIMECODE
        self.frame_data_height = None
        self.frame_data_buffer_size_byte = 0

    # ==========================
    # SDK 生命周期管理
    # ==========================
    def open(self) -> bool:
        if self.is_opened():
            return True

        try:
            hCamera = mvsdk.CameraInit(self.DevInfo, -1, -1)
        except mvsdk.CameraException as e:
            print("CameraInit Failed({}): {}".format(e.error_code, e.message))
            return False
        self.hCamera = hCamera

        cap = mvsdk.CameraGetCapability(hCamera)
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        if monoCamera:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        if self.bit_depth == 0:
            mvsdk.CameraSetMediaType(hCamera, 0)
        elif self.bit_depth == 1:
            mvsdk.CameraSetMediaType(hCamera, 1)

        self.cap = mvsdk.CameraGetCapability(hCamera)

        # Buffer allocation
        self.image_width = self.cap.sResolutionRange.iWidthMax
        self.image_height = self.cap.sResolutionRange.iHeightMax
        self.image_channels = 1 if monoCamera else 3
        
        if CameraFeature.TIMECODE in self.features_enabled:
            self.frame_data_height = self.image_height + self.appended_rows
        else:
            self.frame_data_height = self.image_height
        self.image_buffer_size_byte      = self.image_height      * self.image_width * self.image_channels * self.pixel_bytes
        self.frame_data_buffer_size_byte = self.frame_data_height * self.image_width * self.image_channels * self.pixel_bytes
        # 分配的空间总是有余量, 按照RGB+extra line来分配, 对于raw的话实际上用不了3通道. 如果要用CameraSDK来做RGB, 这样写兼容性好.

        self.pFrameBuffer = mvsdk.CameraAlignMalloc(self.frame_data_buffer_size_byte, 16)
        
        mvsdk.CameraSetRawStartBit(hCamera, -1)

        if self.trigger_mode == 'soft_trigger':
            mvsdk.CameraSetTriggerMode(hCamera, 1)
            mvsdk.CameraSetTriggerCount(hCamera, 1)
            interval_s = 1.0 / self._target_fps
            self.timer = PrecisionTimer.PrecisionTimer(
                interval_s=interval_s,
                c_trigger_func=mvsdk._sdk.CameraSoftTrigger,
                hCamera=hCamera,
                busy_wait_us=2000,
                priority=2
            )
            self.timer.start()
        elif self.trigger_mode == 'freerun':
            mvsdk.CameraSetTriggerMode(hCamera, 0)

        mvsdk.CameraSetAeState(hCamera, 0)
        self.exposure_time_ms = self._target_exposure_time_ms
        mvsdk.CameraSetAnalogGainX(hCamera, self.target_gain)
        self.actual_gain = mvsdk.CameraGetAnalogGainX(hCamera)

        mvsdk.CameraRstTimeStamp(hCamera)
        mvsdk.CameraPlay(hCamera)

        return True

    def close(self) -> bool:
        if self.timer is not None:
            self.timer.stop()
            self.timer.join()
            self.timer = None

        if self.is_opened():
            mvsdk.CameraUnInit(self.hCamera)
            self.hCamera = 0

        if self.pFrameBuffer != 0:
            mvsdk.CameraAlignFree(self.pFrameBuffer)
            self.pFrameBuffer = 0
            
        return True

    def is_opened(self) -> bool:
        return self.hCamera > 0

    # ==========================
    # 能力与属性查询
    # ==========================
    @property
    def features(self) -> CameraFeature: return CameraFeature.TIMECODE

    @property
    def features_enabled(self) -> CameraFeature: return self._features_enabled

    @property
    def width(self) -> int: return self.image_width

    @property
    def height(self) -> int: return self.image_height

    @property
    def channels(self) -> int: return self.image_channels

    @property
    def dtype(self) -> np.dtype: return np.dtype("uint8")

    @property
    def actual_fps(self) -> float:
        if self.trigger_mode == 'soft_trigger':
            return self._target_fps
        else:
            return 1000.0 / self.acutal_exposure_time_ms if self.acutal_exposure_time_ms else 0.0

    @property
    def exposure_time_ms(self) -> float:
        if self.is_opened():
            self.acutal_exposure_time_ms = mvsdk.CameraGetExposureTime(self.hCamera) / 1000.0
            return self.acutal_exposure_time_ms
        return self._target_exposure_time_ms

    @exposure_time_ms.setter
    def exposure_time_ms(self, exposure_ms: float):
        self._target_exposure_time_ms = exposure_ms
        if self.is_opened():
            mvsdk.CameraSetExposureTime(self.hCamera, exposure_ms * 1000)
            self.acutal_exposure_time_ms = mvsdk.CameraGetExposureTime(self.hCamera) / 1000.0

    @property
    def hw_timecode_timebase(self) -> float:
        if not self.supports_feature(CameraFeature.TIMECODE):
            raise AttributeError("This camera does not support hardware timecode.")
        return 10000.0  # 0.1ms

    def get_hw_timecode(self) -> int:
        raise NotImplementedError("Huateng SDK only provides timecode attached to fetched frames.")

    def _update_grab_fps(self):
        current_time = time.perf_counter()
        if self._last_grab_time is not None:
            interval = current_time - self._last_grab_time
            if interval > 0:
                current_fps = 1.0 / interval
                self.actual_grab_fps = (current_fps * self._ema_alpha) + (self.actual_grab_fps * (1 - self._ema_alpha))
        self._last_grab_time = current_time

    # ==========================
    # 核心图像抓取逻辑
    # ==========================
    def _grab_raw_internal(self, timeout_ms=2000) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """私有统一方法，负责向 SDK 索要RAW图像和时间戳"""
        if self.is_opened() == False or self.pFrameBuffer == 0: # Fully Initialized
            return None, None
            
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, timeout_ms)
            timecode_val = FrameHead.uiTimeStamp

            # Raw数据实际仅占用buffer前部一部分空间
            # Create a numpy array view of RawData buffer.
            # Image size before debayer is HxWx(Bpp), for 12bit packed, Bpp=1.5
            if self.bit_depth == 0:
                # TODO: 8 bit raw is currently not supported.
                raw_data_ctypes = (mvsdk.c_ubyte * (self.image_width * self.image_height)).from_address(pRawData)
            elif self.bit_depth == 1: # 12bit = 1.5Bpp
                raw_data_ctypes = (mvsdk.c_ubyte * (self.image_width * self.image_height * 3 // 2)).from_address(pRawData)

            # No need for copy, see speed test below.
            raw_data_np = np.frombuffer(raw_data_ctypes, dtype=np.uint8)
            if self.bit_depth == 1:
                raw_data_np = unpack_12bit_to_16bit_fast(raw_data_np, self.image_height, self.image_width) 
                # <3ms @ 2448*2048, memory-bound, throughput~7GB/s, FYI: np.copy(): 3.53ms/frame
                # TODO: Fuse it to raw_processing pipeline
            
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)
            if CameraFeature.TIMECODE in self.features_enabled:
                return raw_data_np, timecode_val
            else:
                return raw_data_np, None

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print(f"Camera GetImageBuffer/Process failed ({e.error_code}): {e.message}")
            return None, None

    def grab_raw(self, **kwargs) -> Optional[np.ndarray]:
        frame, _ = self._grab_raw_internal(**kwargs)
        if frame is None: return None
        
        self.frames_captured += 1
        self._update_grab_fps()
        
        return frame

    def grab_metadata(self, **kwargs) -> Tuple[Optional[np.ndarray], Any]:
        frame, tc_val = self._grab_raw_internal(**kwargs)
        if frame is None: return None, {}
        if tc_val is None and not CameraFeature.TIMECODE in self.features_enabled:
            print(f"Warning: Camera does not enable hardware timecode. Timecode will be None.")
        
        self.frames_captured += 1
        self._update_grab_fps()
        
        return frame, {'hw_timecode': tc_val}

    @warnings.deprecated("Use grab_raw and grab_metadata instead.")
    def grab_sensor_raw(self, return_timecode: bool = False) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[int]]]:
        """Original API, with return_timecode option."""
        if self.hCamera == 0 or self.pFrameBuffer == 0:
            return (None, None) if return_timecode else None

        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
            if self.bit_depth == 0:
                raw_data_ctypes = (mvsdk.c_ubyte * (self.image_width * self.image_height)).from_address(pRawData)
            elif self.bit_depth == 1:
                raw_data_ctypes = (mvsdk.c_ubyte * (self.image_width * self.image_height * 3 // 2)).from_address(pRawData)

            raw_data_np = np.frombuffer(raw_data_ctypes, dtype=np.uint8)
            if self.bit_depth == 1:
                raw_data_np = unpack_12bit_to_16bit_fast(raw_data_np, self.image_height, self.image_width) 
            
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)
            self._update_grab_fps()
            
            if not return_timecode:
                return raw_data_np
            return raw_data_np, FrameHead.uiTimeStamp

        except mvsdk.CameraException:
            return (None, None) if return_timecode else None

    # ==========================
    # Extended Info (联动重写区)
    # ==========================
    def grab_extended_info(self, **kwargs) -> Optional[np.ndarray]:
        if not self.supports_feature(CameraFeature.TIMECODE):
            return super().grab_extended_info(**kwargs)

        frame, tc_val = self._grab_raw_internal()
        if frame is None: return None

        # 将硬件时间码直接以 C-Bytes 形式写入预分配好的附加行
        timecode_as_bytes = tc_val.to_bytes(TIMECODE_BYTES, byteorder='little')
        ctypes.memmove(self.pFrameBuffer + self.image_buffer_size_byte, timecode_as_bytes, TIMECODE_BYTES)

        self.frames_captured += 1
        self._update_grab_fps()
        return frame

    def extract_extended_info(self, image: np.ndarray, **kwargs) -> Any:
        if not self.supports_feature(CameraFeature.TIMECODE):
            return super().extract_extended_info(image, **kwargs)
            
        if image.shape[0] <= self.height:
            return {}
            
        # 提取第一行附加行的前 4 个字节，反解为 uint32
        extra_rows = image[self.height:, ...]
        extra_byte_view = extra_rows.view(np.uint8).ravel()
        
        if len(extra_byte_view) >= TIMECODE_BYTES:
            tc_val = extra_byte_view[:TIMECODE_BYTES].view(TIMECODE_DTYPE)[0]
            return {'hw_timecode': int(tc_val)}
        return {}

    @property
    def extra_lines(self) -> int:
        if not self.supports_feature(CameraFeature.TIMECODE):
            return super().extra_lines
        return self.appended_rows

    # ==========================
    # 静态工具方法 (收编原有的散落函数)
    # ==========================
    @staticmethod
    def batch_extract_timecode(
        combined_frames: np.ndarray,
        original_height: int,
        original_width: int, # Useless param?
        channels: int,
        timecode_dtype: np.dtype = TIMECODE_DTYPE,
        expected_appended_rows: int = APPENDED_ROWS_FOR_TIMECODE
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        从包含嵌入时间码的帧数据中提取原始图像和时间码。
        时间码数据附加在帧末尾，以小端字节序存储。
        如果无法提取时间码，则时间码部分返回 None。

        Args:
            combined_frames: 包含原始图像数据和附加时间码的 NumPy 数组 (n, H+appended, W, C)。
                            n 是帧的数量，H 是原始图像高度，W 是宽度，C 是通道数。
            original_height: 原始图像的高度。
            original_width: 原始图像的宽度。
            channels: 图像的通道数。
            timecode_dtype: 时间码的 NumPy 数据类型 (默认为 np.uint32)。

        Returns:
            一个元组，包含：
            - 原始图像数据 (np.ndarray)，形状为 (n, H, W, C) 或 (n, combined_H, W, C) 如果 combined_H < H。
            - 提取到的时间码 (Optional[np.ndarray])，形状为 (n,)，数据类型为 timecode_dtype，如果无法提取则为 None。
        """
        if combined_frames.ndim != 4 or combined_frames.shape[3] != channels:
            raise ValueError(f"Invalid combined_frames shape. Expected (n, H+appended, W, C), got {combined_frames.shape} with channels {channels}.")

        n_frames, combined_height, width, _ = combined_frames.shape
        timecode_bytes = timecode_dtype.itemsize

        actual_original_height_to_slice = min(original_height, combined_height)
        original_images = combined_frames[:, :actual_original_height_to_slice, :, :]

        if width != original_width:
            # Useless branch?
            raise ValueError(f"Combined frame width ({width}) does not match original width ({original_width}).")

        if combined_height <= original_height:
            return original_images, None

        try:
            appended_data = combined_frames[:, original_height:, :, :]
            appended_data_uint8_view = appended_data.view(np.uint8)
            appended_data_flat_uint8 = appended_data_uint8_view.reshape(n_frames, -1)
            timecode_byte_batch = appended_data_flat_uint8[:, :timecode_bytes]
            extracted_timecodes = timecode_byte_batch.view(timecode_dtype).ravel()
            return original_images, extracted_timecodes
        except Exception:
            return original_images, None

    @staticmethod
    def decode_12bit_packed_to_16bit_numpy(packed_data: np.ndarray) -> np.ndarray:
        """原有外置的 decode 逻辑，已转为类静态方法。"""
        assert packed_data.size % 3 == 0, "Input data size must be a multiple of 3."
        three_byte_chunks = packed_data.reshape(-1, 3).astype(np.uint16)

        byte0, byte1, byte2 = three_byte_chunks[:, 0], three_byte_chunks[:, 1], three_byte_chunks[:, 2]

        pixel0 = (byte0 << 4) | (byte1 & 0x0F)
        pixel1 = (byte2 << 4) | (byte1 >> 4)

        decoded_data = np.empty(pixel0.size + pixel1.size, dtype=np.uint16)
        decoded_data[0::2] = pixel0
        decoded_data[1::2] = pixel1

        return decoded_data
    
    @staticmethod
    def enumerate_cameras():
        # 枚举相机保持不变
        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            print("No camera was found!")
            return

        for i, DevInfo in enumerate(DevList):
            print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))

        cams =[]
        for i in map(lambda x: int(x), input("Select cameras: ").split()):
            cam = HuatengCamera(DevList[i])
            if cam.open():
                cams.append(cam)
        return cams