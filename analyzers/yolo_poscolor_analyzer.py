# analyzers/yolo_poscolor_analyzer.py
# Gemini 3.1 pro, reviewed by Haiyun Huang (260507)

# perf test: 5060Ti @ 100W 2.7G, ~50fps for 9 tiles of (640x640).
# Tiling is somewhat waste since YOLO can output multiple detection box at a time.

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import threading as t
import time
import os
from io import TextIOWrapper
from pathlib import Path
from typing import Optional, Any, Tuple, Union, Dict, List, Callable, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .yolo_analyzer import YOLOBaseAnalyzer
from .analyzer_types import DeviceType

class TimecodeExtractor(Protocol):
    def __call__(self, image: NDArray, **kwargs: Any) -> tuple[NDArray, List[Dict[str, Any]]]:
        pass
    @property
    def timebase(self) -> int:
        """1/timebase equals time per tick in second."""
        pass
    @property
    def timecode_key(self) -> str:
        """The key to extract timecode from the extended info dict."""
        pass

class YOLOPosColorAnalyzer(YOLOBaseAnalyzer):
    """
    Leaf analyzer that tracks bounding box position, displacement, and bounding 
    box inner grayscale statistics.
    
    Features:
        - Image Tiling (1 frame -> N tiles) for batch processing multiple ROIs.
        - Continuous execution mode by default, managing its own `frame_counter`.
        - GPU transformation for grayscale.
    """

    def __init__(self,
                 frame_server: Any,
                 model_path: Union[str, Path],
                 batch_size: int,
                 tile_grids: List[Tuple[int, int]],
                 tile_shape: Tuple[int, int],
                 color_weights: Optional[Tuple[float, float, float]] = None,
                 save_path: Optional[Union[str, Path]] = None,
                 extinfo_extractor: Optional[TimecodeExtractor] = None,
                 **kwargs):
        """
        Args:
            frame_server (FrameServer): The FrameServer instance.
            model_path (str | Path): Path to the .pt YOLO model file.
            batch_size (int): Batch size for YOLO inference.
            tile_grids (List[Tuple[int, int]]): A list of (x, y) tuples defining 
                the upper-left of tiles.
            tile_shape (Tuple[int, int]): The shape of all tiles (height, width).
            color_weights (Optional[Tuple[float, float, float]]): RGB transformation 
                weights for grayscale. Default is np.eye(3)*(0.299, 0.587, 0.114).
            save_path (Optional[Path]): CSV file path to save results. 
                If None, no CSV file will be saved.
            tc_extractor (Optional[TimecodeExtractor]): 
                Extractor for hardware timecode. If None, will record POSIX 
                nanoseconds of process finish time.
            kwargs:
                batch_size (int): The number of frames to process in a batch. 
                tensor_type (TensorType): TensorType.TORCH is used.
                device (DeviceType): if CUDA available, use GPU.
                consumer_mode (ConsumerMode): control whether frame drop may happens
                    at the cost of blocking the buffer, see FrameServer for details.
                continuous_mode (bool): control whether the analyzer fetch frames 
                    continuously or upon `step()` call (in BaseAnalyzer)
                fetch_timeout (float): (only used when ConsumerMode.SYNC and 
                    continuous_mode=False) timeout for fetch a batch of frames 
                    after `step()`
                stat_interval (float): interval for statistics update.
                inject_logger (Optional[Logger]): loguru logger instance.
        """
        # Force continuous mode for self-managed frame counter streaming
        kwargs['continuous_mode'] = True 
        kwargs['batch_size'] = batch_size
        kwargs['extinfo_extractor'] = extinfo_extractor
        super().__init__(frame_server=frame_server, model_path=model_path, **kwargs)
        self._timecode_extractor = extinfo_extractor
        
        self._tile_grids: List[Tuple[int, int]] = tile_grids
        self._tile_shape: Tuple[int, int] = tile_shape
        
        # Fallback to standard RGB->Gray luminance weights
        if color_weights is None:
            color_weights = (0.299, 0.587, 0.114)
        self._color_weights: Tuple[float, float, float] = color_weights
        
        self._save_path: Optional[Path] = Path(save_path) if save_path else None
        self._csv_file_handle: Optional[TextIOWrapper] = None
        
        # State Management
        self._frame_counter: int = 0
        self._prev_centers: Dict[int, Tuple[float, float]] = {}
        # pickle problem, defer init to _initialize_analyzer which is run in subprocess
        self._history_lock: Optional[t.Lock] = None 
        
        # Subprocess device-specific tensor (initialized in _initialize_analyzer)
        self._color_matrix: Optional[torch.Tensor] = None

    @property
    def model_path(self) -> Path:
        return Path(self._model_path)
    @model_path.setter
    @YOLOBaseAnalyzer.require_stop
    def model_path(self, path: Union[str, Path]):
        self._model_path = Path(path)

    @property
    def save_path(self) -> Optional[Path]:
        return Path(self._save_path)
    @save_path.setter
    @YOLOBaseAnalyzer.require_stop
    def save_path(self, path: Optional[Union[str, Path]]):
        self._save_path = Path(path)

    def _initialize_analyzer(self):
        import torch
        super()._initialize_analyzer()
        self._history_lock = t.Lock()
        
        logger = self._logger
        device_str = 'cuda' if self._device == DeviceType.CUDA else 'cpu'
        # Shape (3, 1) for matrix multiplication with (H, W, 3)
        self._color_matrix = torch.tensor(self._color_weights, device=device_str, dtype=torch.float32).view(3, 1)
        
        if self._save_path:
            # Create directories if not exist
            try:
                os.makedirs(self._save_path.parent, exist_ok=True)
            except (PermissionError, OSError) as e:
                import tempfile
                self._save_path = Path(f"{tempfile.gettempdir()}/{self._save_path.name}")
                try:
                    os.makedirs(self._save_path.parent, exist_ok=True)
                except Exception as e:
                    raise
                logger.warning(f"Failed to create save directory: {e}, fall back to use {self._save_path}")
            file_exists = self._save_path.exists()
            self._csv_file_handle = open(self._save_path, "a", encoding="utf-8")
            if not file_exists or self._save_path.stat().st_size == 0:
                self._csv_file_handle.write("FrameIndex,TileID,Timestamp,Displacement,Grayscale,BBox\n")
            self._csv_file_handle.flush()

    def _uninitialize_analyzer(self):
        if self._csv_file_handle is not None:
            self._csv_file_handle.close()
            self._csv_file_handle = None
        super()._uninitialize_analyzer()

    def _handle_command(self, cmd_name: str, payload: Any):
        """Handle async commands from main process concurrently."""
        # TODO: Not used and not useful.
        if cmd_name == "reset_tracker":
            with self._history_lock:
                self._prev_centers.clear()
                self._frame_counter = 0  # Optional: Reset counter or keep it going
            # Update status dict via IPC to notify main process
            self._status_update({"tracker_status": "reset_completed"})
        else:
            self._status_subdict_update("error", {"cmd": f"Unknown command {cmd_name}"})

    def _preprocess(self, frames: torch.Tensor, ext_info: Optional[List[dict]] = None, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        Handle arbitrary Tiling (List of (x,y) and (h,w)), Type Casting, and NHWC to NCHW rearrangement.
        """
        import torch
        # frame shape: (B, H, W, C)
        B, H, W, C = frames.shape
        N = len(self._tile_grids)
        H_t, W_t = self._tile_shape
        
        # Pre-allocate the final contiguous float32 tensor in NCHW format
        # Shape: (B * N, C, H_t, W_t)
        tiles = torch.empty((B * N, C, H_t, W_t), dtype=torch.float32, device=frames.device)
        
        for i, (tx, ty) in enumerate(self._tile_grids):
            # Slice the patch (B, H_t, W_t, C) and permute to (B, C, H_t, W_t)
            # view, no extra memory copy
            patch = frames[:, ty:ty+H_t, tx:tx+W_t, :].permute(0, 3, 1, 2)
            
            # Copy to pre-allocated tensor with fused Normalization
            # i::N perfectly maps the batch size for N tiles.
            if frames.dtype == torch.uint8:
                tiles[i::N] = patch.float() / 255.0
            elif frames.dtype in (torch.int16, torch.uint16, torch.int32): 
                # 16-bit images loaded via generic pipelines are often int16/int32
                tiles[i::N] = patch.float() / 65535.0
            else:
                tiles[i::N] = patch.float()
            
        metadata = {
            "batch_size": B,
            "original_h": H,
            "original_w": W
        }
        return tiles, metadata

    def _postprocess(self, results: list, original_frames: torch.Tensor, metadata: Dict, ext_info: Optional[List[dict]] = None, **kwargs):
        """
        Map bounding boxes back to global coordinates, calculate displacement,
        compute grayscale via pure GPU Matrix Ops, and update results via IPC.
        """
        import torch
        B = metadata["batch_size"]
        num_tiles = len(self._tile_grids)
        H_t, W_t = self._tile_shape
        orig_h, orig_w = metadata["original_h"], metadata["original_w"]
        
        batch_results_dict = {}

        for b in range(B):
            # 1. Get Timecode from extended info (parsed in BaseAnalyzer)
            timecode = None
            if ext_info:
                timecode = int(ext_info[b][self._timecode_extractor.timecode_key] *
                               1_000_000_000 / self._timecode_extractor.timebase)
            if timecode is None:
                timecode = time.time_ns()

            frame_results = {"timestamp": timecode}

            current_frame_id = self._frame_counter
            self._frame_counter += 1
            
            # 2. Reconstruct tiles and process each independently
            for i in range(num_tiles):
                tile_idx = b * num_tiles + i
                tile_result = results[tile_idx]
                
                best_conf = -1.0
                tile_best_bbox = None
                if len(tile_result.boxes) > 0:
                    # Find the max confidence box in this tile
                    max_idx = torch.argmax(tile_result.boxes.conf).item()
                    best_conf = tile_result.boxes.conf[max_idx].item()
                    tile_best_bbox = tile_result.boxes.xyxy[max_idx]  # Tensor [x1, y1, x2, y2]
                
                global_bbox = None
                displacement = float('nan')
                gray_value = float('nan')

                if tile_best_bbox is not None:
                    # Calculate global offsets
                    x_offset, y_offset = self._tile_grids[i]
                    x1 = max(0,      tile_best_bbox[0].item() + x_offset)
                    y1 = max(0,      tile_best_bbox[1].item() + y_offset)
                    x2 = min(orig_w, tile_best_bbox[2].item() + x_offset)
                    y2 = min(orig_h, tile_best_bbox[3].item() + y_offset)
                    global_bbox = (x1, y1, x2, y2)
                    
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    
                    # Use lock to safely update history state
                    with self._history_lock: # Avoid corruptted data, in GIL, can be removed.
                        # Displacement calculation per tile
                        prev_c = self._prev_centers.get(i)
                        if prev_c is not None:
                            displacement = np.hypot(cx - prev_c[0], cy - prev_c[1])
                        else:
                            displacement = 0.0 # First occurrence has no displacement
                        self._prev_centers[i] = (cx, cy)

                    # Grayscale Calculation (Pure GPU Matrix operation)
                    # original_frame shape is (B, H, W, C). Slice ROI.
                    roi = original_frames[b, int(y1+0.5):int(y2+0.5), int(x1+0.5):int(x2+0.5), :]
                    if roi.numel() > 0:
                        # Matrix Mult: (h, w, 3) @ (3, 1) -> (h, w, 1)
                        roi_gray = torch.matmul(roi.float(), self._color_matrix)
                        # Squeeze to (h, w) and mean
                        gray_value = roi_gray.mean().item()
                    
                    # 3. Store tile result
                    frame_results[i] = {
                        "status": "detected" if global_bbox else "lost",
                        "bbox": global_bbox,
                        "displacement": displacement,
                        "grayscale": gray_value
                    }
                    
                    # 4. Stream to CSV
                    if self._csv_file_handle is not None:
                        bbox_str = f"{global_bbox[0]:.2f},{global_bbox[1]:.2f},{global_bbox[2]:.2f},{global_bbox[3]:.2f}" if global_bbox else "None"
                        self._csv_file_handle.write(f"{current_frame_id},{i},{timecode},{displacement:.2f},{gray_value:.2f},\"{bbox_str}\"\n")
            
            # Format payload using frame_id as Key
            batch_results_dict[current_frame_id] = frame_results
        
        if self._csv_file_handle is not None:
            self._csv_file_handle.flush()

        # Commit batched updates via IPC to Main Process
        self._result_update(batch_results_dict)