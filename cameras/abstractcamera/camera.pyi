import abc
import numpy as np
from .types import CamException as CamException, CameraFeatures as CameraFeatures
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntFlag as IntFlag
from loguru._logger import Logger
from typing import Any, Callable, Concatenate, ParamSpec, TypeVar

F = TypeVar('F', bound=Callable[..., Any])
P = ParamSpec('kwargs')
NDArray = np.ndarray

class AbstractCamera(ABC, metaclass=abc.ABCMeta):
    """
    Abstract camera object that defines common methods for camera interaction.
    """
    def __init_subclass__(cls, **kwargs) -> None: ...
    @abstractmethod
    def __init__(self, *args, inject_logger: Logger, **kwargs):
        """
        Init the camera with optional logger injection. Whether it is used 
        depends on subclass implementation. If needed, call 
        `super().__init__(inject_logger=logger)`
        """
    @abstractmethod
    def open(self) -> bool:
        """
        Open the camera and initialize resources. Blocking method.

        :return: True if success, False if any warning occurs.
        :raise CamException: if unrecoverable error occurs.
        """
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
    @abstractmethod
    def is_capturing(self) -> bool:
        """
        Check whether the camera is capturing frames.

        :return: True if the camera is capturing frames, False otherwise
        """
    @abstractmethod
    def stop_capture(self) -> bool:
        """
        Stop capturing frames.

        :return: True if success, False if any warning occurs.
        :raise CamException: if unrecoverable error occurs.
        """
    @abstractmethod
    def is_opened(self) -> bool:
        """
        Check whether the camera is open.

        :return: True if the camera is open, False otherwise
        """
    @staticmethod
    def require_open(func: F) -> F: ...
    @abstractmethod
    def close(self) -> bool:
        """
        Close the camera and release resources. Blocking method.

        :return: True if success, False if any warning occurs.
        :raise CamException: if unrecoverable error occurs.
        """
    @property
    def features(self) -> CameraFeatures:
        """
        Get the features supported by the camera.

        :return: The features supported by the camera.
        """
    @property
    def features_enabled(self) -> CameraFeatures:
        """
        Get the features enabled by the camera.

        :return: The features enabled by the camera.
        """
    def supports_feature(self, feature: CameraFeatures) -> bool:
        """
        Check whether the camera supports the given feature.

        :param feature: The feature to check.
        :return: True if the camera supports the feature, False otherwise
        """
    def is_feature_enabled(self, feature: CameraFeatures) -> bool:
        """
        Check whether the camera has enabled the given feature.

        :param feature: The feature to check.
        :return: True if the camera has enabled the feature, False otherwise
        """
    @abstractmethod
    def grab_raw(self, **kwargs) -> NDArray | None:
        """
        Grab a raw image frame from the camera. Core function.
                
        :return: NDArray of image data. `None` if failed.
        """
    def grab(self, **kwargs) -> NDArray | None:
        """
        Grab a processed image frame from the camera. Default is to return 
        a raw image from `grab_raw`.

        Any ISP stage can be added here, to reduce user space memory copy.
        
        :return: NDArray of image data. `None` if failed.
        """
    def grab_metadata(self, **kwargs) -> tuple[NDArray | None, Any]:
        """
        Grab a processed image with metadata. Sometimes useful for immediate parsing.
                
        :return: (NDArray, Any). 
                 First element is the NDArray of image data. `None` if failed. 
                 Second element is the metadata. Usually `dict`. 
        """
    def grab_extended_info(self, **kwargs) -> NDArray | None:
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
    def strip_extended_info(self, image: NDArray, **kwargs) -> NDArray:
        """
        Strip the extended info line from the image data.
        Usually, subclass can directly call super().strip_extended_info(image),
        but also could be replaced by subclass implementation.

        :param image: NDArray of image data.
        :return: NDArray of image data without extended info line.
        """
    def get_extended_info_extractor(self) -> Callable[[NDArray], dict[str, Any]]:
        """
        Get the serializable extractor for the extended info.

        Subclass should implement following private functions:
        - _get_ext_info_decode_func() -> Callable[[NDArray, **kwargs], dict[str, Any]]: 
            returns a serializable function that decodes the extended info from
            the image.
        - _get_ext_info_decode_func_kwargs() -> dict[str, Any]: 
            returns the kwargs(serializable dict) for the decode function.
        
        Upon calling this function, the extractor is created, decode function 
        and kwargs are snapshoted, and further __call__ on the extractor will 
        use the snapshoted values. Use the extractor with care since it won't 
        be sync'ed with the camera class status anymore.

        You may not need to rewrite this method.
        
        :return: The extractor object which is callable and serializable. 
        Receives a NDArray of image data. Returns a dict[str, Any] of metadata.
        """
    @property
    @abstractmethod
    def target_fps(self) -> float:
        """Get target FPS of the camera."""
    @target_fps.setter
    def target_fps(self, fps: float) -> None:
        """Set target FPS of the camera."""
    @property
    def actual_fps(self) -> float:
        """
        Get actual FPS of the camera. Only available after capture started.
        Use `target_fps` instead for general use.
        """
    @property
    @abstractmethod
    def frame_count(self) -> int:
        """Get the number of frames grabbed since the camera opened."""
    @property
    @abstractmethod
    def exposure_time_ms(self) -> float:
        """Get current exposure time (milliseconds)."""
    @exposure_time_ms.setter
    @abstractmethod
    def exposure_time_ms(self, value: float) -> None:
        """Set exposure time (milliseconds)."""
    @property
    @abstractmethod
    def exposure_time_ms_range(self) -> tuple[float, float]:
        """Get exposure time (ms) range (min, max)"""
    @property
    def gain(self) -> float:
        """Get current gain value."""
    @gain.setter
    def gain(self, value: float) -> None:
        """Set gain value."""
    @property
    def gain_range(self) -> tuple[float, float]:
        """Get gain range (min, max)."""
    @property
    @abstractmethod
    def width(self) -> int:
        """Get image width (pixels)"""
    @property
    @abstractmethod
    def height(self) -> int:
        """Get image height (pixels)"""
    @property
    def extra_lines(self) -> int:
        """
        Get the number of extended info lines (e.g. pickled metadata) appended 
        to the image.
        The indices of the extra lines are: [self.height, self.height + extra_lines].

        The default value is 1, where a demo metadata dict with software timecode is appended.
        """
    @property
    @abstractmethod
    def channels(self) -> int:
        """Get image channels (e.g. 3 for RGB)"""
    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Get image data type (e.g. np.uint8)"""
    @property
    def frame_size_bytes(self) -> int:
        """Get the size of each frame in bytes. Useful for user to allocate memory."""
    @property
    @abstractmethod
    def full_width(self) -> int: ...
    @property
    @abstractmethod
    def full_height(self) -> int: ...
    @property
    def hw_timecode_timebase(self) -> float:
        """Returns the timebase (denominator) for timecode. 1/timebase = time per tick."""

@dataclass
class ExtInfoExtractor:
    """
    A universal extended info extractor that allows extracting subclass defined
    extended info from the image. Can be serialized thus is multi-process safe.

    Should be compatible to extract batch of frames (NDArray: NHWC or HWC) and 
    return a list of extended information dict.

    `static_decode_func` and `static_strip_func` should be serializable.
    """
    static_decode_func: Callable[Concatenate[NDArray, P], list[dict[str, Any]]]
    static_strip_func: Callable[Concatenate[NDArray, P], NDArray]
    decode_kwargs: dict[str, Any] = field(default_factory=dict)
    strip_kwargs: dict[str, Any] = field(default_factory=dict)
    def __call__(self, image: NDArray) -> tuple[NDArray, list[dict[str, Any]]]:
        """
        Call the static decode function with the image and kwargs(Context).
        
        Args:
            image (NDArray): Image (or batch of images) to extract extended info from.
        
        Returns:
            tuple[NDArray, list[dict[str, Any]]]: Image (or batch of images) 
                without extended info and List of extended info dict.
        """
