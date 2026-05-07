# Author: Haiyun Huang 2026 with Google Gemini 3.1 Pro

# Notes:
# 1. For @abstractmethod and raise NotImplementedError: 
#    - If a method MUST be implemented, @abstractmethod
#    - If a method is optional but recommended, base class will have a scratch
#      that raise NotImplementedError.

# 2. About error handling:
#    - Subclass should add its own ErrorType in its `types.py` for public use.
#    - These ErrorType should describe errors outside python language level 
#      (external error)
#    - Subclass should raise error explicitly and better translate lower level 
#      error type to its own error type.
#    - For predictable errors such as invalid parameters, also raise Error 
#      but not logging (instructions can not execute, that's NOT recoverable.)
#      See below.
#    - By raising errors, all public API functions should NOT use return False to 
#      indicate failure, because that has no useful info.
#      Instead, return a bool can indicate a full success or some warnings occurs.
#      (this design seems better)

# 3. About logging:
#    - Log upto Warning level(recoverable exceptions), raise Error or Critical
#      (but no logging, let caller to handle)
#    - Subclass should use logger provided by loguru, to better handling in 
#      multi-process environment.
#    - For lower-level modules, they may use logging. TODO: We may translate it 
#      to loguru logger? A: It should be done by top-level configuration.

# 4. About property:
#    - Its better to operate camera settings via python property, instead of
#      methods, since this represents a status.



from abc import ABC, abstractmethod
import numpy as np
import pickle
import time
from typing import Optional, Tuple, Union, Any, TypeVar, ParamSpec, Concatenate, Callable, cast
from enum import IntFlag

from loguru import logger as file_logger
from loguru._logger import Logger # for type hint only

# for @require_open
from functools import wraps
F = TypeVar('F', bound=Callable[..., Any])

P = ParamSpec('kwargs') # for labeling Callable[NDArray, **kwargs]

# third-party libraries
# from loguru import logger # Recommend to use loguru to handle multi-process logging.

from .types import CameraFeatures, CamException

NDArray = np.ndarray

class AbstractCamera(ABC):
    # This is a HAL class, so any error should be raised explicitly, 
    # using CamException or your own ErrorType that inherit from CamException.
    """
    Abstract camera object that defines common methods for camera interaction.
    """
    def __init_subclass__(cls, **kwargs):
        # python >= 3.6
        super().__init_subclass__(**kwargs)
        
        # 检查当前子类自身的 __dict__ 中是否同时重写了如下属性/方法
        # Extract 和 Append 相对, 有Append逻辑(grab_extended_info), 就要有Extract逻辑(供其他grab组函数调用)
        has_extract = 'extract_extended_info' in cls.__dict__
        has_decoder_getter = '_get_decode_ext_info_func' in cls.__dict__
        has_decoder_payload_getter = '_get_decode_ext_info_func_kwargs' in cls.__dict__
        has_lines = 'extra_lines' in cls.__dict__
        has_grab_ext = 'grab_extended_info' in cls.__dict__
        if not (has_decoder_getter == has_decoder_payload_getter == 
                has_extract == has_lines == has_grab_ext):
            raise TypeError(
                f"Should overwrite `_get_decode_ext_info_func`, `_get_decode_ext_info_func_kwargs`, "
                f"`extract_extended_info`, `extra_lines`, and `grab_extended_info` in "
                f"{cls.__name__} at the same time."
            )
        # 使用Extra line的时候, strip逻辑是一样的, 如果要重写, 则重写私有方法(入口通常可以直接super())
        has_strip = 'strip_extended_info' in cls.__dict__
        has_strip_getter = '_get_strip_ext_info_func' in cls.__dict__
        has_strip_payload_getter = '_get_strip_ext_info_func_kwargs' in cls.__dict__
        if not (has_strip_getter == has_strip_payload_getter == has_strip):
            raise TypeError(
                f"Should overwrite `_get_strip_ext_info_func`, `_get_strip_ext_info_func_kwargs`, "
                f"`strip_extended_info` in "
                f"{cls.__name__} at the same time."
            )
        has_gain = 'gain' in cls.__dict__
        has_gain_range = 'gain_range' in cls.__dict__
        if not (has_gain == has_gain_range):
            raise TypeError(
                f"Should overwrite `gain` and `gain_range` in "
                f"{cls.__name__} at the same time. "
            )
    
    @abstractmethod
    def __init__(self, *args, inject_logger: Logger, **kwargs):
        """
        Init the camera with optional logger injection. Whether it is used 
        depends on subclass implementation. If needed, call 
        `super().__init__(inject_logger=logger)`
        """
        # args and kwargs should be as less as possible.
        # Here may pass in some camera specific parameters. The caller may use
        # isinstance to decide pass what parameters.
        self._logger: Logger
        if inject_logger is not None:
            if isinstance(inject_logger, Logger):
                self._logger = inject_logger
            else:
                raise TypeError("inject_logger must be a loguru.Logger instance.")
        else:
            self._logger = file_logger
        super().__init__(*args, **kwargs)

    # === Camera Obj Lifecycle Management ===
    @abstractmethod
    def open(self) -> bool:
        """
        Open the camera and initialize resources. Blocking method.

        :return: True if success, False if any warning occurs.
        :raise CamException: if unrecoverable error occurs.
        """
        pass
    
    @abstractmethod
    def start_capture(self) -> bool:
        """
        Start capturing frames. 

        For some cameras, the camera already starts capturing frames when open.
        This function could be explicitly be null(pass) if a camera starts its
        capturing when opened.

        :return: True if success, False if any warning occurs.
        :raise CamException: if unrecoverable error occurs.
        """
        pass
    
    @abstractmethod
    def is_capturing(self) -> bool:
        """
        Check whether the camera is capturing frames.

        :return: True if the camera is capturing frames, False otherwise
        """
        pass

    @abstractmethod
    def stop_capture(self) -> bool:
        """
        Stop capturing frames.

        :return: True if success, False if any warning occurs.
        :raise CamException: if unrecoverable error occurs.
        """
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """
        Check whether the camera is open.

        :return: True if the camera is open, False otherwise
        """
        pass
        
    @staticmethod
    def require_open(func: F) -> F:
        # Decorator for instance method to check if the camera is opened. 
        # require python >= 3.10 if defined as staticmethod. put it outside the class
        # to compatible with lower version.
        @wraps(func)
        def wrapper(self: "AbstractCamera", *args, **kwargs):
            # should have exactly the same signature as the original method.
            # thus `@staticmethod` is not allowed.
            cls_name = self.__class__.__name__
            func_name = func.__name__
            if not self.is_opened():
                raise CamException(f"Operation failed, Camera is not open.", src_func=f"{cls_name}.{func_name}")
            return func(self, *args, **kwargs)
        return cast(F, wrapper)

    @abstractmethod
    def close(self) -> bool:
        """
        Close the camera and release resources. Blocking method.

        :return: True if success, False if any warning occurs.
        :raise CamException: if unrecoverable error occurs.
        """
        pass


    @property
    def features(self) -> CameraFeatures:
        """
        Get the features supported by the camera.

        :return: The features supported by the camera.
        """
        return CameraFeatures.NONE
    
    @property
    def features_enabled(self) -> CameraFeatures:
        """
        Get the features enabled by the camera.

        :return: The features enabled by the camera.
        """
        return CameraFeatures.NONE

    def supports_feature(self, feature: CameraFeatures) -> bool:
        """
        Check whether the camera supports the given feature.

        :param feature: The feature to check.
        :return: True if the camera supports the feature, False otherwise
        """
        return feature in self.features
    
    def is_feature_enabled(self, feature: CameraFeatures) -> bool:
        """
        Check whether the camera has enabled the given feature.

        :param feature: The feature to check.
        :return: True if the camera has enabled the feature, False otherwise
        """
        return feature in self.features_enabled

    # === Camera Actions ===
    @abstractmethod
    def grab_raw(self, **kwargs) -> Optional[NDArray]:
        """
        Grab a raw image frame from the camera. Core function.
                
        :return: NDArray of image data. `None` if failed.
        """
        pass

    def grab(self, **kwargs) -> Optional[NDArray]:
        """
        Grab a processed image frame from the camera. Default is to return 
        a raw image from `grab_raw`.

        Any ISP stage can be added here, to reduce user space memory copy.
        
        :return: NDArray of image data. `None` if failed.
        """
        image = self.grab_raw(**kwargs)
        return image

    def grab_metadata(self, **kwargs) -> Tuple[Optional[NDArray], Any]:
        """
        Grab a processed image with metadata. Sometimes useful for immediate parsing.
                
        :return: (NDArray, Any). 
                 First element is the NDArray of image data. `None` if failed. 
                 Second element is the metadata. Usually `dict`. 
        """
        raise NotImplementedError("Subclass should implement this method.")

    def grab_extended_info(self, **kwargs) -> Optional[NDArray]:
        """
        Grab processed image frame with extended info line (e.g. pickled metadata).
        Default is to append a software timecode with a full frame copy action.
        Overwrite this method to use native larger buffer provided by 
        camera SDK that fits the extended info line.
        A reference implementation is provided, see `_append_metadata_to_image`.
        
        .. note:: If requiring the raw image with extended info, overwrite 
            `grab_raw` to add extended info from the beginning, and **keep it** in 
            following steps.

        :return: NDArray of image data. None if failed. 
        """
        image = self.grab(**kwargs)
        metadata = {'sw_timecode': time.time()}
        if image is None:
            return None
        image_with_metadata = self._append_metadata_to_image(image, metadata)
        return image_with_metadata

    # === Extended Info Handling ===
    def extract_extended_info(self, image: NDArray, **kwargs) -> Any:
        """
        Extract the extended info from the image with extended info line.

        The core extraction logic should be implemented in a staticmethod
        such as `_extract_metadata_from_image`, to meet the demand of 
        generating a serializable extractor.

        Usually, subclass can directly call super().extract_extended_info(image).

        :param image: NDArray of image data.
        :return: The extended info.
        """
        # Usually, subclass can directly call super().extract_extended_info(image)
        # but you may call feature specific staticmethod here.
        return self._get_decode_ext_info_func()(image, self.extra_lines)
        raise NotImplementedError("Subclass should implement this method.")
    def strip_extended_info(self, image: NDArray, **kwargs) -> NDArray:
        """
        Strip the extended info line from the image data.
        Usually, subclass can directly call super().strip_extended_info(image),
        but also could be replaced by subclass implementation.

        :param image: NDArray of image data.
        :return: NDArray of image data without extended info line.
        """
        return self._get_strip_ext_info_func()(image, self.extra_lines)
    
    def get_extended_info_extractor(self) -> Callable[[NDArray], dict[str, Any]]:
        """
        Get the extractor for the extended info. Should be serializable.

        Subclass should implement following private functions:
        - _get_ext_info_decode_func() -> Callable[[NDArray, **kwargs], dict[str, Any]]: 
            returns a serializable function that decodes the extended info from
            the image.
        - _get_ext_info_decode_func_kwargs() -> dict[str, Any]: 
            returns the kwargs for the decode function.
        
        Upon calling this function, the extractor is created, decode function 
        and kwargs are snapshoted, and further __call__ on the extractor will 
        use the snapshoted values. Use the extractor with care since it won't 
        be sync'ed with the camera class status anymore.

        You may not need to rewrite this method.
        
        :return: The extractor function. Receives a NDArray of image data.
                 Returns a dict[str, Any] of metadata.
        """
        decode_payload: dict = self._get_decode_ext_info_func_kwargs()
        strip_payload: dict = self._get_strip_ext_info_func_kwargs()
        extractor = ExtInfoExtractor(
            static_decode_func=self._get_decode_ext_info_func(),
            static_strip_func=self._get_strip_ext_info_func(),
            decode_kwargs=decode_payload, 
            strip_kwargs=strip_payload
        )
        return extractor
    
    # === Hook function that needs to rewrite in subclass ===
    def _get_decode_ext_info_func(self) -> Callable[Concatenate[NDArray, P], dict[str, Any]]:
        """
        Hook function to get the static decode function for the extended info.
        Subclass should overwrite this method if use its own extended info.

        :return func (Callable[[NDArray, **kwargs], dict[str, Any]]): 
            A static function that decodes the extended info from the image.
        """
        return self.__class__._extract_metadata_from_image
    def _get_decode_ext_info_func_kwargs(self) -> dict[str, Any]:
        """
        Hook function to get the context payload for the static decode function.
        Subclass should overwrite this method if use its own extended info.

        :return payload (dict[str, Any]): The kwargs for the decode function.
        """
        return {'extra_lines': self.extra_lines}
    def _get_strip_ext_info_func(self) -> Callable[[NDArray], NDArray]:
        """
        Hook function to get the static function for stripping the extended info.
        Subclass should overwrite this method if use its own extended info.

        :return func (Callable[[NDArray], NDArray]): 
            A static function that strips the extended info line from the image.
        """
        return self.__class__._strip_metadata_from_image
    def _get_strip_ext_info_func_kwargs(self) -> dict[str, Any]:
        """
        Hook function to get the context payload for the static strip function.
        Subclass should overwrite this method if use its own extended info.

        :return payload (dict[str, Any]): The kwargs for the strip function.
        """
        return {'extra_lines': self.extra_lines}
    
    # === Pure Camera SDK Calling ===
    @property
    @abstractmethod
    def target_fps(self) -> float:
        """Get target FPS of the camera."""
        pass

    @target_fps.setter
    def target_fps(self, fps: float) -> None:
        """Set target FPS of the camera."""
        raise NotImplementedError("Subclass should implement this method.")

    @property
    def actual_fps(self) -> float:
        """
        Get actual FPS of the camera. Only available after capture started.
        Use `target_fps` instead for general use.
        """
        # 原始实现中,许多下游用了这个属性,按照道理来讲,这个属性必须在录制之后才能获取到真值
        # 其余时候只能是返回一些虚假值. 对于录制开始前的需求, 重构时请使用target_fps
        # TODO: Remove this property.
        return self.target_fps # A good default value for base class.
    
    @property
    @abstractmethod
    def frame_count(self) -> int:
        """Get the number of frames grabbed since the camera opened."""
        pass

    @property
    @abstractmethod
    def exposure_time_ms(self) -> float:
        """Get current exposure time (milliseconds)."""
        pass

    @exposure_time_ms.setter
    @abstractmethod
    def exposure_time_ms(self, value: float) -> None:
        """Set exposure time (milliseconds)."""
        pass

    @property
    @abstractmethod
    def exposure_time_ms_range(self) -> Tuple[float, float]:
        """Get exposure time (ms) range (min, max)"""
        pass

    @property
    def gain(self) -> float:
        """Get current gain value."""
        raise NotImplementedError("Subclass should implement this method.")

    @gain.setter
    def gain(self, value: float) -> None:
        """Set gain value."""
        raise NotImplementedError("Subclass should implement this method.")
    
    @property
    def gain_range(self) -> Tuple[float, float]:
        """Get gain range (min, max)."""
        raise NotImplementedError("Subclass should implement this method.")
    
    # === Memory Allocation-related Property ===
    @property
    @abstractmethod
    def width(self) -> int:
        """Get image width (pixels)"""
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        """Get image height (pixels)"""
        pass

    @property
    def extra_lines(self) -> int:
        """
        Get the number of extended info lines (e.g. pickled metadata) appended 
        to the image.
        The indices of the extra lines are: [self.height, self.height + extra_lines].

        The default value is 1, where a demo metadata dict with software timecode is appended.
        """
        return 1

    @property
    @abstractmethod
    def channels(self) -> int:
        """Get image channels (e.g. 3 for RGB)"""
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Get image data type (e.g. np.uint8)"""
        return np.dtype(np.uint8)
    
    @property
    def frame_size_bytes(self) -> int:
        """Get the size of each frame in bytes. Useful for user to allocate memory."""
        return self.width * (self.height + self.extra_lines) * self.channels * np.dtype(self.dtype).itemsize

    # === Camera Hardware Description ===
    @property
    @abstractmethod
    def full_width(self)  -> int: pass
    @property
    @abstractmethod
    def full_height(self) -> int: pass
    
    # === Feature: Hardware Timecode ===
    @property
    def hw_timecode_timebase(self) -> float:
        """Returns the timebase (denominator) for timecode. 1/timebase = time per tick."""
        if not self.supports_feature(CameraFeatures.TIMECODE):
            raise AttributeError("This camera does not support hardware timecode.")
        raise NotImplementedError("Subclass should implement this property since the camera supports timecode.")

    # === Private Functions ===
    # TODO: Subclass possible optimization: Zero-copy; Use ctypes.Structure 
    # instead of dict for metadata.
    def _append_metadata_to_image(self, image: NDArray, metadata: dict) -> NDArray:
        """
        Private helper to serialize dict and append it to image as extra lines.
        Returns a copy of NDArray with extended info line.
        """
        serialized = pickle.dumps(metadata)
        length = len(serialized)

        extra_shape = list(image.shape)
        extra_shape[0] = self.extra_lines
        extra_rows = np.zeros(extra_shape, dtype=image.dtype)
        
        extra_byte_view = extra_rows.view(np.uint8).ravel()
        if length + 4 > len(extra_byte_view):
            raise ValueError(f"Metadata bytes ({length}B) is too large to fit in "
                             f"{self.extra_lines} extra lines ({len(extra_byte_view)}B)."
                             "Check or overwrite the `extra_lines` property.")
        
        # Data layout: 4 bytes for metadata length (little-endian uint32), 
        # followed by metadata bytes.
        extra_byte_view[:4] = np.array([length], dtype=np.uint32).view(np.uint8)
        extra_byte_view[4:4+length] = np.frombuffer(serialized, dtype=np.uint8)

        # Return a copy of frame with extra lines
        return np.concatenate((image, extra_rows), axis=0)
    
    @staticmethod
    def _extract_metadata_from_image(image: NDArray, extra_lines: int) -> dict:
        """
        Static method to extract serialized dict from the image with given 
        `extra_lines` (positive integer). 

        Rewrite this staticmethod for subclass specific metadata extraction.
        Make sure this method(function) is serializable.
        """
        if image.shape[0] <= extra_lines or extra_lines <= 0:
            # No extra lines to extract.
            return {}
        extra_rows = image[-extra_lines:, ...]
        extra_byte_view = extra_rows.view(np.uint8).ravel()

        length = extra_byte_view[:4].view(np.uint32)[0]
        
        # Simple check to make sure length is not corrupted.
        if length == 0 or length > len(extra_byte_view) - 4:
            return {}
            
        serialized_bytes = extra_byte_view[4:4+length].tobytes()

        try:
            return pickle.loads(serialized_bytes)
        except pickle.PickleError:
            return {}
    
    @staticmethod
    def _strip_metadata_from_image(image: NDArray, extra_lines: int) -> NDArray:
        """
        Static method to strip extended info from the image with given 
        `extra_lines` (positive integer). 

        Rewrite this staticmethod for subclass specific metadata extraction.
        Make sure this method(function) is serializable.
        """
        return image[:-extra_lines, ...]

# TODO: use frozen=True to have clearer immutable semantics?
class ExtInfoExtractor:
    """
    A universal extended info extractor that allows extracting subclass defined
    extended info from the image. Can be serialized thus is multi-process safe.
    """
    def __init__(self, static_decode_func: Callable[Concatenate[NDArray, P], dict[str, Any]], 
                 static_strip_func: Callable[[NDArray], NDArray],
                 decode_kwargs: dict[str, Any] = {},
                 strip_kwargs: dict[str, Any] = {}):
        """Save the static decode function and its context."""
        self.decode_func = static_decode_func
        self.strip_func = static_strip_func
        self.decode_kwargs: dict[str, Any] = decode_kwargs # payload for extraction, such as ctype.Structure etc.
        self.strip_kwargs: dict[str, Any] = strip_kwargs # payload for stripping

    def __call__(self, image: NDArray) -> tuple[NDArray, dict[str, Any]]:
        """Call the static decode function with the image and kwargs(Context)."""
        extended_info = self.decode_func(image, **self.decode_kwargs)
        stripped_image = self.strip_func(image, **self.strip_kwargs)
        return stripped_image, extended_info