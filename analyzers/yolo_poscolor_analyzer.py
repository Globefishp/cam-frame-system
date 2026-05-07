# analyzers/yolo_poscolor_analyzer.py
# Gemini 3.1 pro, reviewed by Haiyun Huang (260507)

import torch
import numpy as np
import threading as t
from typing import Optional, Any, Tuple, Union, Dict, List
from .yolo_analyzer import YOLOBaseAnalyzer
from .analyzer_types import DeviceType

class YOLOPosColorAnalyzer(YOLOBaseAnalyzer):
    """
    Leaf analyzer that tracks bounding box position, displacement, and bounding 
    box inner grayscale statistics.
    
    Features:
        - Image Tiling (1 frame -> N tiles) to maintain resolution on large images.
        - Continuous execution mode by default, managing its own `frame_counter`.
        - Pure GPU transformation for grayscale.
    """

    def __init__(self,
                 frame_server: Any,
                 model_path: str,
                 batch_size: int,
                 tile_grids: List[Tuple[int, int]],
                 tile_shape: Tuple[int, int],
                 color_weights: Optional[Tuple[float, float, float]] = None,
                 **kwargs):
        """
        Args:
            batch_size (int): Batch size for YOLO inference.
            tile_grids (List[Tuple[int, int]]): A list of (x, y) tuples defining the upper-left of tiles.
            tile_shape (Tuple[int, int]): The shape of all tiles (height, width).
            color_weights (Optional[Tuple[float, float, float]]): RGB transformation weights for grayscale. 
                Default is np.eye(3)*(0.299, 0.587, 0.114).
        """
        # Force continuous mode for self-managed frame counter streaming
        kwargs['continuous_mode'] = True 
        kwargs['batch_size'] = batch_size
        super().__init__(frame_server=frame_server, model_path=model_path, **kwargs)
        
        self._tile_grids = tile_grids
        self._tile_shape = tile_shape
        
        # Fallback to standard RGB->Gray luminance weights
        if color_weights is None:
            color_weights = (0.299, 0.587, 0.114)
        self._color_weights = color_weights
        
        # State Management
        self._frame_counter: int = 0
        self._prev_center: Optional[Tuple[float, float]] = None
        self._history_lock = t.Lock()
        
        # Subprocess device-specific tensor (initialized in _initialize_analyzer)
        self._color_matrix: Optional[torch.Tensor] = None

    def _initialize_analyzer(self):
        super()._initialize_analyzer()
        device_str = 'cuda' if self._device == DeviceType.CUDA else 'cpu'
        # Shape (3, 1) for matrix multiplication with (H, W, 3)
        self._color_matrix = torch.tensor(self._color_weights, device=device_str, dtype=torch.float32).view(3, 1)

    def _handle_command(self, cmd_name: str, payload: Any):
        """Handle async commands from main process concurrently."""
        if cmd_name == "reset_tracker":
            with self._history_lock:
                self._prev_center = None
                self._frame_counter = 0  # Optional: Reset counter or keep it going
            # Update status dict via IPC to notify main process
            self._status_update({"tracker_status": "reset_completed"})
        else:
            self._status_subdict_update("error", {"cmd": f"Unknown command {cmd_name}"})

    def _preprocess(self, frame: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        Handle arbitrary Tiling (List of (x,y) and (h,w)), Type Casting, and NHWC to NCHW rearrangement.
        """
        # frame shape: (B, H, W, C)
        B, H, W, C = frame.shape
        N = len(self._tile_grids)
        H_t, W_t = self._tile_shape
        
        # Pre-allocate the final contiguous float32 tensor in NCHW format
        # Shape: (B * N, C, H_t, W_t)
        tiles = torch.empty((B * N, C, H_t, W_t), dtype=torch.float32, device=frame.device)
        
        for i, (tx, ty) in enumerate(self._tile_grids):
            # Slice the patch (B, H_t, W_t, C) and permute to (B, C, H_t, W_t)
            # view, no extra memory copy
            patch = frame[:, ty:ty+H_t, tx:tx+W_t, :].permute(0, 3, 1, 2)
            
            # Copy to pre-allocated tensor with fused Normalization
            # i::N perfectly maps the batch size for N tiles.
            if frame.dtype == torch.uint8:
                tiles[i::N] = patch.float() / 255.0
            elif frame.dtype in (torch.int16, torch.uint16, torch.int32): 
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

    def _postprocess(self, results: list, original_frame: torch.Tensor, metadata: Dict, **kwargs):
        """
        Map bounding boxes back to global coordinates, calculate displacement,
        compute grayscale via pure GPU Matrix Ops, and update results via IPC.
        """
        B = metadata["batch_size"]
        num_tiles = len(self._tile_grids)
        H_t, W_t = self._tile_shape
        orig_h, orig_w = metadata["original_h"], metadata["original_w"]
        
        batch_results_dict = {}

        for b in range(B):
            best_conf = -1.0
            best_bbox = None
            
            # 1. Reconstruct tiles and find highest confidence globally for this frame
            for i in range(num_tiles):
                tile_idx = b * num_tiles + i
                tile_result = results[tile_idx]
                
                if len(tile_result.boxes) > 0:
                    # Find the max confidence box in this tile
                    max_idx = torch.argmax(tile_result.boxes.conf).item()
                    conf = tile_result.boxes.conf[max_idx].item()
                    
                    if conf > best_conf:
                        best_conf = conf
                        box = tile_result.boxes.xyxy[max_idx]  # Tensor [x1, y1, x2, y2]
                        
                        # Calculate global offsets
                        x_offset, y_offset = self._tile_grids[i]
                        
                        x1 = box[0].item() + x_offset
                        y1 = box[1].item() + y_offset
                        x2 = box[2].item() + x_offset
                        y2 = box[3].item() + y_offset
                        
                        # Clamp to image boundaries just in case
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(orig_w, int(x2)), min(orig_h, int(y2))
                        
                        best_bbox = (x1, y1, x2, y2)
            
            # 2. Logic Extraction (Displacement and Grayscale)
            displacement = float('nan')
            gray_value = float('nan')
            
            # Use lock to safely update history state
            with self._history_lock:
                current_frame_id = self._frame_counter
                self._frame_counter += 1
                
                if best_bbox is not None:
                    x1, y1, x2, y2 = best_bbox
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    
                    # Displacement calculation
                    if self._prev_center is not None:
                        displacement = np.hypot(cx - self._prev_center[0], cy - self._prev_center[1])
                    else:
                        displacement = 0.0 # First occurrence has no displacement
                    
                    self._prev_center = (cx, cy)
            
            if best_bbox is not None:
                x1, y1, x2, y2 = best_bbox
                # Grayscale Calculation (Pure GPU Matrix operation)
                # original_frame shape is (B, H, W, C). Slice ROI.
                roi = original_frame[b, y1:y2, x1:x2, :]
                
                if roi.numel() > 0:
                    # Convert ROI to float (if not already) for matrix multiplication
                    roi_float = roi.float()
                    # Matrix Mult: (h, w, 3) @ (3, 1) -> (h, w, 1)
                    roi_gray = torch.matmul(roi_float, self._color_matrix)
                    # Squeeze to (h, w) and mean
                    gray_value = roi_gray.mean().item()
            
            # 3. Format payload using frame_id as Key
            batch_results_dict[current_frame_id] = {
                "status": "detected" if best_bbox else "lost",
                "bbox": best_bbox,          # Format: Tuple(x1, y1, x2, y2)
                "displacement": displacement,
                "grayscale": gray_value
            }
        
        # Commit batched updates via IPC to Main Process
        self._result_update(batch_results_dict)