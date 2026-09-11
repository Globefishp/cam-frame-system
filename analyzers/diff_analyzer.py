# analyzers/diff_analyzer.py

import os
import time
import threading as t
from io import TextIOWrapper
from pathlib import Path
from typing import Optional, Any, Tuple, Union, Dict, List, Protocol, TYPE_CHECKING
import torch

from .analyzer import BaseAnalyzer
from .analyzer_types import TensorType, DeviceType, ConsumerMode
from .ops import GridTiling

class TimecodeExtractor(Protocol):
    def __call__(self, image: Any, **kwargs: Any) -> tuple[Any, List[Dict[str, Any]]]:
        pass
    @property
    def timebase(self) -> int:
        """1/timebase equals time per tick in second."""
        pass
    @property
    def timecode_key(self) -> str:
        """The key to extract timecode from the extended info dict."""
        pass

def compute_absdiff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes absolute difference safely for both uint8 and float tensors."""
    if a.dtype == torch.uint8:
        return torch.where(a > b, a - b, b - a)
    else:
        return torch.abs(a - b)

def compute_diff_metrics(diff: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes Mean Absolute Difference (MAD) and pixel changed count."""
    mad = diff.mean(dim=[-2, -1])
    px_changed = (diff > threshold).sum(dim=[-2, -1])
    return mad, px_changed

class DiffAnalyzer(BaseAnalyzer):
    """
    Stateful analyzer that computes pixel-wise differences between consecutive frames per tile.
    
    Features:
        - Image Tiling (1 frame -> N tiles) for processing multiple ROIs.
        - FrameServer V3/V4 Integration using ConsumerMode.SYNC with keep_last_n=1 
          to retain the historical frame buffer without memory copying.
        - Vectorized batch processing to handle batch_size >= 1 efficiently without torch.cat 
          on large image tensors.
    """

    def __init__(self,
                 frame_server: Any,
                 tile_grids: List[Tuple[int, int]],
                 tile_shape: Tuple[int, int],
                 px_changed_threshold: float = 10.0,
                 channel_weights: Optional[List[float]] = None,
                 save_path: Optional[Union[str, Path]] = None,
                 extinfo_extractor: Optional[TimecodeExtractor] = None,
                 **kwargs):
        """
        Initialize the DiffAnalyzer.

        Args:
            frame_server: The FrameServer instance.
            tile_grids: Top-left coordinates (x, y) of each tile.
            tile_shape: The shape of all tiles (height, width).
            px_changed_threshold: Raw value threshold for the tensor dtype to 
                determine if a pixel is considered "changed".
            channel_weights: Optional RGB channel weights for pixel difference.
                Default `None` uses equal weights.
            save_path: Path to save the analysis results.
            extinfo_extractor: Extractor for hardware timecode.
            kwargs: Additional arguments for BaseAnalyzer.
        """
        # Force SYNC mode and keep_last_n=1 to retain previous buffer without copy
        kwargs['consumer_mode'] = ConsumerMode.SYNC
        kwargs['keep_last_n'] = 1
        kwargs['tensor_type'] = TensorType.TORCH
        kwargs['continuous_mode'] = True # Force continuous mode for self-managed frame counter
        kwargs['extinfo_extractor'] = extinfo_extractor
        
        # Determine device
        if 'device' not in kwargs:
            kwargs['device'] = DeviceType.CUDA if torch.cuda.is_available() else DeviceType.CPU

        super().__init__(frame_server=frame_server, **kwargs)

        self._tile_grids = tile_grids
        self._tile_shape = tile_shape
        self._px_changed_threshold = px_changed_threshold
        self._timecode_extractor = extinfo_extractor
        self._ch_weights = channel_weights
        
        self._tile_area = len(self._tile_grids) * self._tile_shape[0] * self._tile_shape[1]
        
        self._tiling_op = GridTiling(
            grid_coords=self._tile_grids,
            tile_shape=self._tile_shape,
            tiling_axes=(1, 2),
            axis=1,
            permute=None
        )

        self._save_path: Optional[Path] = Path(save_path) if save_path else None
        self._csv_file_handle: Optional[TextIOWrapper] = None

        self._data_lock: Optional[t.Lock] = None
        self._prev_tiles: Optional[torch.Tensor] = None # for _tile_first
        self._prev_frame: Optional[torch.Tensor] = None # for _diff_first
        self._frame_counter: int = 0

    @property
    def save_path(self) -> Optional[Path]:
        return Path(self._save_path)

    @save_path.setter
    @BaseAnalyzer.require_stop
    def save_path(self, path: Optional[Union[str, Path]]):
        self._save_path = Path(path)

    def _initialize_analyzer(self):
        """Worker Process: Initialize state variables."""
        self._data_lock = t.Lock()
        self._prev_tiles = None
        self._frame_counter = 0
        self._ch_weights_tensor = None
        self._logger.info("DiffAnalyzer state initialized.")
        
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
                self._logger.warning(f"Failed to create save directory: {e}, fall back to use {self._save_path}")
            file_exists = self._save_path.exists()
            self._csv_file_handle = open(self._save_path, "a", encoding="utf-8")
            if not file_exists or self._save_path.stat().st_size == 0:
                self._csv_file_handle.write("FrameIndex,TileID,Timestamp,MAD,PxChanged\n")
            self._csv_file_handle.flush()

    def _uninitialize_analyzer(self):
        """Worker Process: Cleanup state."""
        if self._csv_file_handle is not None:
            self._csv_file_handle.close()
            self._csv_file_handle = None
        self._prev_tiles = None
        self._prev_frame = None

    def _handle_command(self, cmd_name: str, payload: Any):
        """Handle custom commands asynchronously."""
        if cmd_name == "reset_state":
            with self._data_lock:
                self._prev_tiles = None
                self._prev_frame = None
                self._frame_counter = 0
                # recompute tile area in case tile_grids were updated externally (for future mod)
                self._tile_area = len(self._tile_grids) * self._tile_shape[0] * self._tile_shape[1]
            self._logger.info("DiffAnalyzer state reset requested and completed.")

    def _analyze(self, frames: torch.Tensor, ext_info: Optional[List[dict]] = None, **kwargs) -> Any:
        """
        Worker Process: Core logic to compute difference.
        
        Args:
            frames: Input tensor of shape (B, H, W, C).
            ext_info: Optional extended info for each frame in the batch.
        """
        B, H, W, C = frames.shape
        image_area = H * W
        
        # Dynamic strategy routing based on overlapping area
        if self._tile_area > image_area:
            batch_results_dict = self._diff_first(frames, ext_info)
        else:
            batch_results_dict = self._tile_first(frames, ext_info)
            
        # Commit Results via IPC
        self._result_update(batch_results_dict)

    def _tile_first(self, frames: torch.Tensor, ext_info: Optional[List[dict]] = None) -> dict:
        """Process tiles first, then compute differences. Good for small ROIs."""
        B, H, W, C = frames.shape
        N = len(self._tile_grids)
        H_t, W_t = self._tile_shape

        # 1. Tiling
        # output shape: (B, N, H_t, W_t, C) since permute is removed
        tiles = torch.empty((B, N, H_t, W_t, C), dtype=torch.float32, device=frames.device)
        self._tiling_op.process(frames, out=tiles)

        with self._data_lock:
            if self._ch_weights_tensor is None and self._ch_weights is not None and C == len(self._ch_weights):
                self._ch_weights_tensor = torch.tensor(self._ch_weights, dtype=torch.float32, device=frames.device)

            # 2. Compute first frame difference (without copying image tensors)
            if self._prev_tiles is not None:
                first_diff = torch.abs(tiles[0] - self._prev_tiles) # (N, H_t, W_t, C)
                if self._ch_weights_tensor is not None: # Custom channel mixing
                    first_diff = torch.tensordot(first_diff, self._ch_weights_tensor, dims=([-1], [0])) # (N, H_t, W_t)
                else:
                    first_diff = first_diff.mean(dim=-1) # (N, H_t, W_t)
                
                first_mad, first_px = compute_diff_metrics(first_diff, self._px_changed_threshold)
                first_mad = first_mad.unsqueeze(0)
                first_px = first_px.unsqueeze(0)
            else:
                first_mad = torch.zeros((1, N), dtype=torch.float32, device=frames.device)
                first_px = torch.zeros((1, N), dtype=torch.int64, device=frames.device)

            # 3. Compute internal batch differences if B > 1
            if B > 1:
                internal_diff = torch.abs(tiles[1:] - tiles[:-1]) # (B-1, N, H_t, W_t, C)
                if self._ch_weights_tensor is not None:
                    internal_diff = torch.tensordot(internal_diff, self._ch_weights_tensor, dims=([-1], [0])) # (B-1, N, H_t, W_t)
                else:
                    internal_diff = internal_diff.mean(dim=-1) # (B-1, N, H_t, W_t)
                
                internal_mad, internal_px = compute_diff_metrics(internal_diff, self._px_changed_threshold)
                
                # Concatenate the metric tensors
                mad = torch.cat([first_mad, internal_mad], dim=0) # (B, N)
                px_changed = torch.cat([first_px, internal_px], dim=0) # (B, N)
            else:
                mad = first_mad
                px_changed = first_px

            # 4. Format Results
            batch_results_dict = self._format_results(B, N, mad, px_changed, ext_info)

            # Roll State
            self._prev_tiles = tiles[-1]
            self._prev_frame = None

        return batch_results_dict

    def _diff_first(self, frames: torch.Tensor, ext_info: Optional[List[dict]] = None) -> dict:
        """Process diff first on the whole image, then tile. Good for overlapping ROIs."""
        B, H, W, C = frames.shape
        N = len(self._tile_grids)
        H_t, W_t = self._tile_shape

        with self._data_lock:
            if self._ch_weights_tensor is None and self._ch_weights is not None and C == len(self._ch_weights):
                self._ch_weights_tensor = torch.tensor(self._ch_weights, dtype=torch.float32, device=frames.device)

            # 1. Compute global differences
            if self._prev_frame is not None:
                frames_shifted = torch.cat([self._prev_frame.unsqueeze(0), frames[:-1]], dim=0)
                diff = compute_absdiff(frames, frames_shifted)
                has_first = True
            else:
                if B > 1:
                    frames_shifted = frames[:-1]
                    diff = compute_absdiff(frames[1:], frames_shifted)
                else:
                    diff = None
                has_first = False

            if diff is not None:
                if diff.dtype == torch.uint8:
                    diff_float = diff.float() / 255.0
                elif diff.dtype in (torch.int16, torch.uint16, torch.int32):
                    diff_float = diff.float() / 65535.0
                else:
                    diff_float = diff.float()

                if self._ch_weights_tensor is not None:
                    diff_float = torch.tensordot(diff_float, self._ch_weights_tensor, dims=([-1], [0])) # (B_diff, H, W)
                else:
                    diff_float = diff_float.mean(dim=-1) # (B_diff, H, W)

                px_map = diff_float > self._px_changed_threshold
                combined = torch.stack([diff_float, px_map.float()], dim=-1) # (B_diff, H, W, 2)

                B_diff = diff.shape[0]
                tiled = torch.empty((B_diff, N, H_t, W_t, 2), dtype=torch.float32, device=frames.device)
                self._tiling_op.process(combined, out=tiled, norm_factor=1.0)

                computed_mad = tiled[..., 0].mean(dim=(-2, -1))
                computed_px = tiled[..., 1].sum(dim=(-2, -1)).to(torch.int64)
            else:
                computed_mad = None
                computed_px = None

            if not has_first:
                first_mad = torch.zeros((1, N), dtype=torch.float32, device=frames.device)
                first_px = torch.zeros((1, N), dtype=torch.int64, device=frames.device)
                if computed_mad is not None:
                    mad = torch.cat([first_mad, computed_mad], dim=0)
                    px_changed = torch.cat([first_px, computed_px], dim=0)
                else:
                    mad = first_mad
                    px_changed = first_px
            else:
                mad = computed_mad
                px_changed = computed_px

            batch_results_dict = self._format_results(B, N, mad, px_changed, ext_info)

            # Roll State
            self._prev_frame = frames[-1]
            self._prev_tiles = None

        return batch_results_dict

    def _format_results(self, B: int, N: int, mad: torch.Tensor, px_changed: torch.Tensor, ext_info: Optional[List[dict]]) -> dict:
        """Helper to format and log the analysis results."""
        batch_results_dict = {}
        for b in range(B):
            timecode = None
            if ext_info and self._timecode_extractor:
                timecode = int(ext_info[b][self._timecode_extractor.timecode_key] *
                               1_000_000_000 / self._timecode_extractor.timebase)
            if timecode is None:
                timecode = time.time_ns()

            frame_id = self._frame_counter
            frame_results = {"timestamp": timecode}
            for i in range(N):
                mad_val = mad[b, i].item()
                px_val = px_changed[b, i].item()
                frame_results[i] = {
                    "mad": mad_val,
                    "px_changed": px_val
                }
                if self._csv_file_handle is not None:
                    self._csv_file_handle.write(f"{frame_id},{i},{timecode},{mad_val:.2f},{px_val}\n")
            
            self._frame_counter += 1
            batch_results_dict[frame_id] = frame_results

        if self._csv_file_handle is not None:
            self._csv_file_handle.flush()
            
        return batch_results_dict
