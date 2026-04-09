# tests/test_tiffencoder.py

import os
import pytest
import numpy as np
import tifffile
from ome_types import from_tiff

from encoders.tiff_encoder import TiffEncoder
from ringbuffers.shared_ring_buffer_v2a import ProcessSafeSharedRingBuffer

@pytest.fixture
def mock_ring_buffer():
    """Mock the shared ring buffer to decouple tests from IPC complexities."""
    buffer = ProcessSafeSharedRingBuffer(create=True, buffer_capacity=10, frame_shape=(10,10,1))
    yield buffer
    
    buffer.close()
    buffer.unlink()

@pytest.fixture
def temp_output_file(tmp_path):
    """Provide a temporary file path for testing."""
    return str(tmp_path / "test_output.ome.tif")

def test_initialization_missing_kwargs(mock_ring_buffer, temp_output_file):
    """Test that missing frame_size raises ValueError."""
    with pytest.raises(ValueError, match="frame_size must be provided"):
        TiffEncoder(mock_ring_buffer, temp_output_file)

def test_initialization_success(mock_ring_buffer, temp_output_file):
    """Test proper kwargs parsing."""
    encoder = TiffEncoder(
        mock_ring_buffer, 
        temp_output_file, 
        frame_size=(512, 512), 
        z_slices=10, 
        pixel_size_um=0.5
    )
    assert encoder._frame_size == (512, 512)
    assert encoder._z_slices == 10
    assert encoder._pixel_size_um == 0.5
    assert encoder._samples_per_pixel == 1

def test_encode_frames_and_padding(mock_ring_buffer, temp_output_file):
    """
    Test the core worker lifecycle: init -> encode -> padding -> uninit.
    We simulate Z=10, but only provide 15 frames. 
    It should pad 5 frames automatically.
    """
    encoder = TiffEncoder(
        mock_ring_buffer, 
        temp_output_file, 
        frame_size=(64, 64), 
        z_slices=10,
        batch_size=5
    )
    
    # 1. Manually call worker methods (bypassing multiprocessing for deterministic test)
    encoder._initialize_encoder()
    
    # Create 15 dummy frames (shape: N, H, W)
    dummy_chunk_1 = np.ones((10, 64, 64), dtype=np.uint16)
    dummy_chunk_2 = np.ones((5, 64, 64), dtype=np.uint16)
    
    # 2. Encode
    success1 = encoder._encode_frames([dummy_chunk_1])
    success2 = encoder._encode_frames([dummy_chunk_2])
    
    assert success1 and success2
    assert encoder._total_frames_written == 15
    
    # 3. Uninitialize (this should trigger padding and XML update)
    encoder._uninitialize_encoder()
    
    # Assert padding logic triggered (15 + 5 pad = 20)
    assert encoder._total_frames_written == 20
    
    # 4. Verify physical file using pure tifffile
    with tifffile.TiffFile(temp_output_file) as tif:
        assert len(tif.pages) == 20 # The file must have exactly 20 pages
        
        # Verify the resolution tags were written correctly on the first page
        page = tif.pages[0]
        assert page.tags['XResolution'].value[0] == int(1e4 / 1.0)

def test_ome_xml_update(mock_ring_buffer, temp_output_file):
    """
    Test that the final OME-XML is valid and reflects the correctly calculated SizeT.
    """
    encoder = TiffEncoder(
        mock_ring_buffer, 
        temp_output_file, 
        frame_size=(32, 32), 
        z_slices=5,
        ch_colors=["#00FF00", "#FF0000"]
    )
    
    encoder._initialize_encoder()
    
    # Provide exactly 2 full logical sequences (T=2)
    # Total frames = T * Z * C = 2 * 5 * 2 = 20
    frames = np.zeros((20, 32, 32), dtype=np.uint16)
    encoder._encode_frames([frames])
    encoder._uninitialize_encoder()
    
    # Parse the resulting TIFF using ome-types
    ome_obj = from_tiff(temp_output_file)
    
    # Assertions on the OME logic
    assert len(ome_obj.images) == 1
    pixels = ome_obj.images[0].pixels
    
    assert pixels.size_z == 5
    assert pixels.size_c == 2
    assert pixels.size_t == 2 # T must be 2
    assert pixels.physical_size_x == 1.0
    
    # Verify TiffData mappings were generated
    assert len(pixels.tiff_data_blocks) == 1
    assert pixels.tiff_data_blocks[0].plane_count == 20