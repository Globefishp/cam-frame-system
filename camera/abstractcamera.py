from abc import ABC, abstractmethod
import numpy as np
import pickle
import time
from typing import Optional, Tuple, Union, Any
from enum import IntFlag

NDArray = np.ndarray

class CameraFeature(str, IntFlag):
    NONE          = 0      # Default capability.
    TIMECODE      = 1 << 0


class AbstractCamera(ABC):
    """
    Abstract camera object that defines common methods for camera interaction.
    """
    def __init_subclass__(cls, **kwargs):
        # python >= 3.6
        super().__init_subclass__(**kwargs)
        
        # 检查当前子类自身的 __dict__ 中是否重写了这三个属性/方法
        has_extract = 'extract_extended_info' in cls.__dict__
        has_lines = 'extra_lines' in cls.__dict__
        has_grab_ext = 'grab_extended_info' in cls.__dict__
        if not (has_extract == has_lines == has_grab_ext):
            raise TypeError(
                f"Should overwrite `extract_extended_info`, `extra_lines`, and "
                f"`grab_extended_info` in {cls.__name__} at the same time."
            )

    # === Camera Obj Management ===
    @abstractmethod
    def open(self) -> bool:
        """
        Open the camera and initialize resources. Blocking method.

        :return: True if success, False otherwise
        """
        pass

    @abstractmethod
    def close(self) -> bool:
        """
        Close the camera and release resources. Blocking method.

        :return: True if success, False otherwise
        """
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """
        Check whether the camera is open.

        :return: True if the camera is open, False otherwise
        """
        pass

    @property
    def features(self) -> CameraFeature:
        """
        Get the features supported by the camera.

        :return: The features supported by the camera.
        """
        return CameraFeature.NONE
    
    @property
    def features_enabled(self) -> CameraFeature:
        """
        Get the features enabled by the camera.

        :return: The features enabled by the camera.
        """
        return CameraFeature.NONE

    def supports_feature(self, feature: CameraFeature) -> bool:
        """
        Check whether the camera supports the given feature.

        :param feature: The feature to check.
        :return: True if the camera supports the feature, False otherwise
        """
        return feature in self.features
    
    def is_feature_enabled(self, feature: CameraFeature) -> bool:
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
        Grab a raw image frame from the camera.
                
        :return: NDArray of image data. `None` if failed.
        """
        pass

    def grab_metadata(self, **kwargs) -> Tuple[Optional[NDArray], Any]:
        """
        Grab a image with metadata. Sometimes useful for immediate parsing.
                
        :return: (NDArray, Any). 
                 First element is the NDArray of image data. `None` if failed. 
                 Second element is the metadata. Usually `dict`. 
        """
        raise NotImplementedError("Subclass should implement this method.")

    def grab_extended_info(self, **kwargs) -> Optional[NDArray]:
        """
        Grab image frame with extended info line (e.g. pickled metadata).
        Default is to append a software timecode with a full frame copy action.
        Overwrite this method to use native larger buffer provided by 
        camera SDK that fits the extended info line.
        A reference implementation is provided, see `_append_metadata_to_image`.

        :return: NDArray of image data. None if failed. 
        """
        image = self.grab_raw(**kwargs)
        metadata = {'sw_timecode': time.time()}
        if image is None:
            return None
        image_with_metadata = self._append_metadata_to_image(image, metadata)
        return image_with_metadata

    # === Extended Info Handling ===
    def extract_extended_info(self, image: NDArray, **kwargs) -> Any:
        """
        Extract the extended info from the image with extended info line.

        :param image: NDArray of image data.
        :return: The extended info.
        """
        # You may call feature specific methods here.
        raise NotImplementedError("Subclass should implement this method.")
    def strip_extended_info(self, image: NDArray, **kwargs) -> NDArray:
        """
        Strip the extended info line from the image data.

        :param image: NDArray of image data.
        :return: NDArray of image data without extended info line.
        """
        return image[:self.height, :]
    
    # === Pure Camera SDK Calling ===
    @property
    @abstractmethod
    def actual_fps(self) -> float:
        """Get actual FPS of the camera."""
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
    def dtype(self) -> np.dtype:
        """Get image data type (e.g. np.uint8)"""
        pass
    
    # === Feature: Hardware Timecode ===
    @property
    def hw_timecode_timebase(self) -> float:
        if not self.supports_feature(CameraFeature.TIMECODE):
            raise AttributeError("This camera does not support hardware timecode.")
        raise NotImplementedError("Subclass should implement this property since the camera supports timecode.")
           
    def get_hw_timecode(self) -> int:
        """
        Get the current timecode value.

        :return: The current timecode value.
        """
        if not self.supports_feature(CameraFeature.TIMECODE):
            raise AttributeError("This camera does not support hardware timecode.")
        raise NotImplementedError("Subclass should implement this method since the camera supports timecode.")

    # === Private Functions ===
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
            raise ValueError(f"Metadata bytes ({length}B) is too large to fit in {self.extra_lines} extra lines ({len(extra_byte_view)}B)."
                             "Check or overwrite the `extra_lines` property.")
        
        # Data layout: 4 bytes for metadata length (little-endian uint32), 
        # followed by metadata bytes.
        extra_byte_view[:4] = np.array([length], dtype=np.uint32).view(np.uint8)
        extra_byte_view[4:4+length] = np.frombuffer(serialized, dtype=np.uint8)

        # Return a copy of frame with extra lines
        return np.concatenate((image, extra_rows), axis=0)

    def _extract_metadata_from_image(self, image: NDArray) -> dict:
        """Private helper to extract serialized dict from the image."""
        if image.shape[0] <= self.height:
            # Maybe useless check since it is a private function.
            return {}
        extra_rows = image[self.height:self.height + self.extra_lines, ...]
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