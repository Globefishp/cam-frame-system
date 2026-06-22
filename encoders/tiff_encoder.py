# encoders/tiff_encoder.py
# Scratched/modified by Gemini 3.1 Pro, modified/reviewed by Haiyun Huuang 260622

import os
import multiprocessing as mp
import numpy as np
from numpy.typing import NDArray, DTypeLike
from typing import Tuple, Any, Optional, List, Dict, Callable
from .videoencoder_v3 import BaseVideoEncoder
from .videoencoder_types import EncoderException, TimecodeExtractor
from ..frameserver.v3.frameserver_v3 import FrameServer

from loguru import logger as file_logger
from loguru._logger import Logger # For Type Hinting Only

import time
from datetime import datetime
import tifffile
from ome_types import OME, model

_OME_XML_BASE_BYTES = 4096
_MAX_PLANES = 640000
_OME_XML_BYTES_PER_PLANE = 100

class TiffEncoder(BaseVideoEncoder):
    """
    Concrete implementation of BaseVideoEncoder for streaming OME-TIFF encoding.
    
    Supports BigTIFF streaming write, OME-XML metadata generation, and automatic 
    Z-stack/Channel padding upon unexpected stops. Optimized for biomedical 
    imaging compatibility (e.g., ImageJ/Fiji Bio-Formats).
    """
    
    # Pre-allocate bytes for OME-XML header enabling in-place modification.
    XML_PADDING_BYTES = _OME_XML_BASE_BYTES + _OME_XML_BYTES_PER_PLANE * _MAX_PLANES # ~64 MB

    # Map numpy dtypes to OME-XML pixel types
    _OME_DTYPE_MAP = {
        "uint8": "uint8",
        "uint16": "uint16",
        "uint32": "uint32",
        "int8": "int8",
        "int16": "int16",
        "int32": "int32",
        "float32": "float",
        "float64": "double",
        "bool": "bit"
    }

    def __init__(self,
                 frame_server: FrameServer,
                 output_path: str,
                 frame_size: Tuple[int, ...],
                 z_slices: int = 1,
                 ch_colors: Optional[List[str]] = None,
                 pixel_size_um: float = 1.0,
                 dimension_order: str = 'XYZTC',
                 timecode_posix: bool = False,
                 dtype: DTypeLike = np.uint16,
                 batch_size: int = 5,
                 target_fps: Optional[float] = None,
                 stat_interval: float = 1.0,
                 extinfo_extractor: Optional[TimecodeExtractor] = None,
                 inject_logger: Optional[Logger] = None,
                 **kwargs
        ):
        """
        Args:
            frame_server: (FrameServer), the frame server instance.
            output_path: (str), path to the output .ome.tif file.
            frame_size: (Tuple[int, int] or Tuple[int, int, int]), (H, W) or (H, W, C).
            z_slices: (int), number of Z planes in a full stack. Defaults to 1.
            ch_colors: (List[str]), list of channel colors in pydantic compatible
                color format. The length of list should equals to channel number. 
                Defaults to ["#00FF00"].
            pixel_size_um: (float), physical pixel size in micrometers. Defaults to 1.0.
            dimension_order: (str), OME dimension order. Defaults to 'XYZTC'.
            timecode_posix: (bool), if True, the timecode will be interpreted as posix 
                timestamp for calculating absolute `AcquisitionDate` in tiff metadata. 
                Default `False`.
            dtype: (DTypeLike), dtype of the frames. Defaults to np.uint16.

            batch_size: (int), number of frames to get from the buffer at once. Defaults to 5.
            target_fps: (Optional[float]), expected encoding speed.
            stat_interval: (float), interval between status updates.
            extinfo_extractor: (Optional[TimecodeExtractor]), the timecode extractor 
                instance for extracting timecodes from frames. If None, timecodes will
                not be extracted. Default to None. 
            inject_logger: (Optional[Logger]), loguru logger instance.
            kwargs: Any other arguments for BaseVideoEncoder.
        """
        super().__init__(frame_server=frame_server, output_path=output_path, 
                         batch_size=batch_size, target_fps=target_fps, 
                         stat_interval=stat_interval, extinfo_extractor=extinfo_extractor, 
                         inject_logger=inject_logger, **kwargs)
        self._logger: Logger = self._logger.bind(friendly_name="TiffEncoder")
        
        self._frame_size = frame_size
            
        self._z_slices: int = z_slices
        self._ch_colors: List[str] = ch_colors if ch_colors is not None else ["#00FF00"]
        self._num_channels: int = len(self._ch_colors)
        self._pixel_size_um: float = pixel_size_um
        self._dimension_order: str = dimension_order
        # `_timecode_timebase`, `_timecode_key` and `ext_info` in `_encode_frames` should co-exists. 
        self._timecode_timebase: Optional[float] = extinfo_extractor.timebase if extinfo_extractor else None
        self._timecode_key: Optional[str] = extinfo_extractor.timecode_key if extinfo_extractor else None
        self._timecode_posix: bool = timecode_posix
        self._dtype = np.dtype(dtype)
        
        self._timecodes: List[float] = []
        self._acquisition_start_time: Optional[datetime] = None
        
        # Calculate samples_per_pixel (e.g., 1 for grayscale, 3 for RGB)
        self._samples_per_pixel = self._frame_size[2] if len(self._frame_size) == 3 else 1
        
        # Encoder state
        self._tiff_writer: Optional[tifffile.TiffWriter] = None
        self._total_frames_written: int = 0

        # Metadata constructs
        self._resolution_tags = (1e4 / self._pixel_size_um, 1e4 / self._pixel_size_um, 'CENTIMETER')

    def _create_ome_metadata_bytes(self, total_t: int) -> bytes:
        """
        Helper method to generate compliant OME-XML bytes.
        
        Args:
            total_t: (int), total time point. if _MAX_FRAMES, will treat as
                initializing file.
        
        Returns:
            (bytes), UTF-8 encoded OME-XML metadata.
        """
        logger = self._logger
        total_planes = total_t * self._z_slices * self._num_channels
        
        # Define channels
        channels =[
            model.Channel(id=f"Channel:0:{i}", 
                          samples_per_pixel=self._samples_per_pixel,
                          color=color)
            for i, color in enumerate(self._ch_colors)
        ]
        
        # Define TiffData mapping for ImageJ compatibility
        tiff_data = model.TiffData(ifd=0, plane_count=total_planes)

        # Construct planes metadata (including timecode and ZTC info)
        planes = []
        if self._timecodes and total_planes <= _MAX_PLANES:
            timebase = self._timecode_timebase
            if timebase is None:
                logger.warning("Timebase is not set, using default timebase 1.0s.")
                timebase = 1.0
            t0 = self._timecodes[0]
            for i, tc in enumerate(self._timecodes):
                if i >= total_planes:
                    break
                z = i % self._z_slices
                c = (i // self._z_slices) % self._num_channels
                t = i // (self._z_slices * self._num_channels)
                delta_t = (tc - t0) / timebase
                planes.append(model.Plane(the_c=c, the_z=z, the_t=t, delta_t=delta_t, delta_t_unit="s"))
        pixels = model.Pixels(
            id="Pixels:0",
            dimension_order=self._dimension_order,
            size_x=self._frame_size[1],
            size_y=self._frame_size[0],
            size_z=self._z_slices,
            size_c=self._num_channels,
            size_t=total_t,
            type=self._OME_DTYPE_MAP.get(self._dtype.name, "uint16"),
            physical_size_x=self._pixel_size_um,
            physical_size_y=self._pixel_size_um,
            physical_size_z=1.0, # TODO: Add Z-step size
            physical_size_x_unit="µm", 
            physical_size_y_unit="µm",
            physical_size_z_unit="µm",
            channels=channels,
            tiff_data_blocks=[tiff_data],
            planes=planes
        )
        
        ome = OME()
        image_kwargs = {"id": "Image:0", "pixels": pixels}
        if self._acquisition_start_time:
            image_kwargs["acquisition_date"] = self._acquisition_start_time.isoformat()
        ome.images.append(model.Image(**image_kwargs))
        
        xml_str = ome.to_xml()
        # Forces UTF-8 encoding and padding to bypass tifffile ASCII checks.
        return xml_str.encode('utf-8').ljust(self.XML_PADDING_BYTES, b' ')

    def _initialize_encoder(self):
        """
        Initializes the TiffWriter in BigTIFF mode.
        Runs in the worker process.
        """
        pid, friendly_name = mp.current_process().pid, "TiffEncoder"
        logger = self._logger
        
        logger.info(f"Initializing TiffWriter for {self._output_path} (BigTIFF=True)...")
        self._total_frames_written = 0
        
        try:
            # Keep the writer open for streaming
            self._tiff_writer = tifffile.TiffWriter(self._output_path, bigtiff=True)
        except Exception as e:
            raise EncoderException(f"Failed to open TiffWriter: {e}", pid=pid, name=friendly_name) from e

    def _encode_frames(self, frames_list: List[NDArray], ext_info: Optional[List[dict]] = None, **kwargs) -> bool:
        """
        Writes a chunk of frames to the TIFF file consecutively.
        """
        pid, friendly_name = mp.current_process().pid, "TiffEncoder"
        logger = self._logger

        if not frames_list:
            return False

        if self._tiff_writer is None:
            raise EncoderException("TiffWriter is not initialized.", pid=pid, name=friendly_name)

        try:
            # dimensions for unpacking
            height, width = self._frame_size[0], self._frame_size[1]
            channels = self._samples_per_pixel
            expected_pixels = height * width * channels
            
            # Extract timecode (len(ext_info) should equals to frames in chunks)
            if ext_info and self._timecode_key:
                # Check ext_info size
                batch_size = sum(chunk.shape[0] if chunk.ndim > 2 else 1 for chunk in frames_list)
                if len(ext_info) != batch_size:
                    logger.warning("Extended info size mismatch with the frames in "
                        f"current batch, starting from frame encoded #{self._total_frames_written}."
                        f"batch size: {batch_size}, ext_info size: {len(ext_info)}")
                for i, item in enumerate(ext_info):
                    tc = item.get(self._timecode_key)
                    if tc is not None:
                        self._timecodes.append(float(tc))
                    else:
                        # use the last known timecode, or 0.0 if not available.
                        if self._timecodes:
                            self._timecodes.append(self._timecodes[-1])
                        else:
                            self._timecodes.append(0.0)
                        logger.warning(f"Timecode missing for frame {self._total_frames_written + i}. Using last known value.")
                        

            for frame_chunk in frames_list:
                if frame_chunk is None or frame_chunk.size == 0:
                    continue
                
                # frame_chunk shape is typically (N, H, W) or (N, H, W, C)
                # Extract effective pixels.
                if expected_pixels > frame_chunk[0].size:
                    raise EncoderException(f"Frame data mismatch (less) than declared. "
                        f"slot pixels: {frame_chunk[0].size}, expected: {expected_pixels}")
                frame_chunk = frame_chunk.reshape(frame_chunk.shape[0], -1)[:, :expected_pixels]
                if channels > 1:
                    frame_chunk = frame_chunk.reshape(-1, height, width, channels)
                else:
                    frame_chunk = frame_chunk.reshape(-1, height, width)

                for i in range(frame_chunk.shape[0]):
                    frame = frame_chunk[i]
                    
                    if self._total_frames_written == 0:
                        # First batch: run start time estimation
                        if self._timecodes:
                            if self._timecode_posix:
                                self._acquisition_start_time = datetime.fromtimestamp(self._timecodes[0])
                            else:
                                last_tc = self._timecodes[-1]
                                sys_time = time.time()
                                timebase = self._timecode_timebase
                                if timebase is None:
                                    logger.warning("Timebase is not set, using default timebase 1.0s.")
                                    timebase = 1.0
                                delta_ticks = last_tc - self._timecodes[0]
                                delta_seconds = delta_ticks / timebase
                                start_sys_time = sys_time - delta_seconds
                                self._acquisition_start_time = datetime.fromtimestamp(start_sys_time)
                        
                        # First frame: allocate huge dummy SizeT and pad XML
                        dummy_xml = self._create_ome_metadata_bytes(total_t=_MAX_PLANES) # Dummy value
                        self._tiff_writer.write(
                            frame,
                            description=dummy_xml,
                            resolution=self._resolution_tags,
                            contiguous=True
                        )
                    else:
                        self._tiff_writer.write(frame, contiguous=True)
                        
                    self._total_frames_written += 1
                    
        except Exception as e:
            raise EncoderException(
                f"Error writing frame to TIFF. Frame index: {self._total_frames_written}. Error: {e}", 
                pid=pid, name=friendly_name
            ) from e

        return True

    def _uninitialize_encoder(self):
        """
        Pads missing frames if the sequence is incomplete, closes the file, 
        and updates the OME-XML header in-place with the correct dimensions.
        """
        pid, friendly_name = mp.current_process().pid, "TiffEncoder"
        logger = self._logger

        logger.info(f"Uninitializing TiffWriter. Total frames written {self._total_frames_written}")
        
        if self._tiff_writer is None:
            return

        try:
            # 1. Pad missing frames (Black frames) to complete the last Z-stack/Channel sequence
            full_stack_size = self._z_slices * self._num_channels
            remainder = self._total_frames_written % full_stack_size
            
            if remainder != 0 and self._total_frames_written > 0:
                pad_count = full_stack_size - remainder
                logger.warning(f"Incomplete Z/C stack detected. Padding {pad_count} black frames.")
                
                # Determine correct dtype based on the instance variable
                black_frame = np.zeros(self._frame_size, dtype=self._dtype)
                last_tc = self._timecodes[-1] if self._timecodes else 0.0
                for _ in range(pad_count):
                    self._tiff_writer.write(black_frame, contiguous=True)
                    self._total_frames_written += 1
                    self._timecodes.append(last_tc)
            
            # 2. Close the file handle
            self._tiff_writer.close()
            self._tiff_writer = None
            
            # 3. Update the OME-XML Metadata in-place instantly
            if self._total_frames_written > 0:
                logger.info("Updating actual SizeT in OME-XML header...")
                actual_t = self._total_frames_written // full_stack_size
                final_xml_bytes = self._create_ome_metadata_bytes(total_t=actual_t)
                
                # tiffcomment uses the padding to overwrite the header instantaneously
                tifffile.tiffcomment(self._output_path, final_xml_bytes)
                logger.success(f"OME-TIFF saved successfully. Final dimensions: T={actual_t}, Z={self._z_slices}, C={self._num_channels}")
                
        except Exception as e:
            logger.error(f"Error during TiffEncoder uninitialization: {e}")
            raise EncoderException(f"Cleanup failed: {e}", pid=pid, name=friendly_name) from e