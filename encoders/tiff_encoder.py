# encoders/tiff_encoder.py
# Gemini 3.1 Pro

import os
import multiprocessing as mp
import numpy as np
from typing import Tuple, Any, Optional, List
from ringbuffers.shared_ring_buffer_v2a import ProcessSafeSharedRingBuffer
from encoders.videoencoder import BaseVideoEncoder
from encoders.videoencoder_types import EncoderException

from loguru import logger as file_logger
from loguru._logger import Logger # For Type Hinting Only

import tifffile
from ome_types import OME, model

class TiffEncoder(BaseVideoEncoder):
    """
    Concrete implementation of BaseVideoEncoder for streaming OME-TIFF encoding.
    
    Supports BigTIFF streaming write, OME-XML metadata generation, and automatic 
    Z-stack/Channel padding upon unexpected stops. Optimized for biomedical 
    imaging compatibility (e.g., ImageJ/Fiji Bio-Formats).
    """
    
    # Pre-allocate 16KB for OME-XML header to ensure in-place modification is always possible
    XML_PADDING_BYTES = 16384 

    def __init__(self,
                 shared_buffer: ProcessSafeSharedRingBuffer,
                 output_path: str,
                 batch_size: int = 5,
                 expected_fps: Optional[float] = None,
                 inject_logger: Optional[Logger] = None,
                 **kwargs
        ):
        """
        Args:
            shared_buffer: (ProcessSafeSharedRingBuffer), the shared buffer instance.
            output_path: (str), path to the output .ome.tif file.
            batch_size: (int), number of frames to get from the buffer at once.
            expected_fps: (Optional[float]), expected encoding speed.
            inject_logger: (Optional[Logger]), loguru logger instance.
            kwargs:
                frame_size: (Tuple[int, int] or Tuple[int, int, int]), (H, W) or (H, W, C).
                z_slices: (int), number of Z planes in a full stack. Defaults to 1.
                ch_colors: (List[str]), list of channel colors in pydantic compatible
                    color format. The length of list should equals to channel number. 
                    Defaults to ["#00FF00"].
                pixel_size_um: (float), physical pixel size in micrometers. Defaults to 1.0.
                dimension_order: (str), OME dimension order. Defaults to 'XYZTC'.
        """
        super().__init__(
            shared_buffer, output_path, 
            batch_size=batch_size, expected_fps=expected_fps, inject_logger=inject_logger
        )
        
        # Parse kwargs
        self._frame_size = kwargs.pop('frame_size', None)
        if self._frame_size is None:
            raise ValueError("frame_size must be provided in kwargs (e.g., (2048, 2048) or (2048, 2048, 3))")
            
        self._z_slices: int = kwargs.pop('z_slices', 1)
        self._ch_colors: List[str] = kwargs.pop('ch_colors', ["#00FF00"])
        self._num_channels: int = len(self._ch_colors)
        self._pixel_size_um: float = kwargs.pop('pixel_size_um', 1.0)
        self._dimension_order: str = kwargs.pop('dimension_order', 'XYZTC')
        
        # Calculate samples_per_pixel (e.g., 1 for grayscale, 3 for RGB)
        self._samples_per_pixel = self._frame_size[2] if len(self._frame_size) == 3 else 1
        
        # Encoder state
        self._tiff_writer: Optional[tifffile.TiffWriter] = None
        self._total_frames_written: int = 0

        # Metadata constructs
        self._resolution_tags = (1e4 / self._pixel_size_um, 1e4 / self._pixel_size_um, 'CENTIMETER')

    def _create_ome_metadata_bytes(self, current_t: int) -> bytes:
        """
        Helper method to generate compliant OME-XML bytes.
        Forces UTF-8 encoding and padding to bypass tifffile ASCII checks.
        """
        total_planes = current_t * self._z_slices * self._num_channels
        
        # Define channels
        channels =[
            model.Channel(id=f"Channel:0:{i}", 
                          samples_per_pixel=self._samples_per_pixel,
                          color=color)
            for i, color in enumerate(self._ch_colors)
        ]
        
        # Define TiffData mapping for ImageJ compatibility
        tiff_data = model.TiffData(ifd=0, plane_count=total_planes)

        pixels = model.Pixels(
            id="Pixels:0",
            dimension_order=self._dimension_order,
            size_x=self._frame_size[1],
            size_y=self._frame_size[0],
            size_z=self._z_slices,
            size_c=self._num_channels,
            size_t=current_t,
            type="uint16", # Assuming 16-bit by default for biomedical, can be parameterized
            physical_size_x=self._pixel_size_um,
            physical_size_y=self._pixel_size_um,
            physical_size_z=1.0,
            physical_size_x_unit="µm", 
            physical_size_y_unit="µm",
            physical_size_z_unit="µm",
            channels=channels,
            tiff_data_blocks=[tiff_data]
        )
        
        ome = OME()
        ome.images.append(model.Image(id="Image:0", pixels=pixels))
        
        xml_str = ome.to_xml()
        return xml_str.encode('utf-8').ljust(self.XML_PADDING_BYTES, b' ')

    def _initialize_encoder(self):
        """
        Initializes the TiffWriter in BigTIFF mode.
        Runs in the worker process.
        """
        pid, friendly_name = mp.current_process().pid, "TiffEncoder"
        logger = self._logger.bind(friendly_name=friendly_name)
        
        logger.info(f"Initializing TiffWriter for {self._output_path} (BigTIFF=True)...")
        self._total_frames_written = 0
        
        try:
            # Keep the writer open for streaming
            self._tiff_writer = tifffile.TiffWriter(self._output_path, bigtiff=True)
        except Exception as e:
            raise EncoderException(f"Failed to open TiffWriter: {e}", pid=pid, name=friendly_name) from e

    def _encode_frames(self, frames_list: List[np.ndarray]) -> bool:
        """
        Writes a chunk of frames to the TIFF file consecutively.
        """
        pid, friendly_name = mp.current_process().pid, "TiffEncoder"
        logger = self._logger.bind(friendly_name=friendly_name)

        if not frames_list:
            return False

        if self._tiff_writer is None:
            raise EncoderException("TiffWriter is not initialized.", pid=pid, name=friendly_name)

        try:
            for frame_chunk in frames_list:
                if frame_chunk is None or frame_chunk.size == 0:
                    continue
                
                # frame_chunk shape is typically (N, H, W) or (N, H, W, C)
                for i in range(frame_chunk.shape[0]):
                    frame = frame_chunk[i]
                    
                    if self._total_frames_written == 0:
                        # First frame: allocate huge dummy SizeT and pad XML
                        dummy_xml = self._create_ome_metadata_bytes(current_t=999999)
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
        logger = self._logger.bind(friendly_name=friendly_name)

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
                
                # Determine correct dtype based on the class logic (fallback to uint16)
                black_frame = np.zeros(self._frame_size, dtype=np.uint16)
                for _ in range(pad_count):
                    self._tiff_writer.write(black_frame, contiguous=True)
                    self._total_frames_written += 1
            
            # 2. Close the file handle
            self._tiff_writer.close()
            self._tiff_writer = None
            
            # 3. Update the OME-XML Metadata in-place instantly
            if self._total_frames_written > 0:
                logger.info("Updating actual SizeT in OME-XML header...")
                actual_t = self._total_frames_written // full_stack_size
                final_xml_bytes = self._create_ome_metadata_bytes(current_t=actual_t)
                
                # tiffcomment uses the padding to overwrite the header instantaneously
                tifffile.tiffcomment(self._output_path, final_xml_bytes)
                logger.success(f"OME-TIFF saved successfully. Final dimensions: T={actual_t}, Z={self._z_slices}, C={self._num_channels}")
                
        except Exception as e:
            logger.error(f"Error during TiffEncoder uninitialization: {e}")
            raise EncoderException(f"Cleanup failed: {e}", pid=pid, name=friendly_name) from e