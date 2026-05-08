# analyzers/yolo_analyzer.py
# Gemini 3.1 pro, reviewed by Haiyun Huang (260507)

from frameserver import FrameServer
from abc import abstractmethod
import torch
import threading as t
from typing import Optional, Any, Tuple, Union, Dict, List
from pathlib import Path
from ultralytics import YOLO

from .analyzer import BaseAnalyzer
from .analyzer_types import TensorType, DeviceType, AnalyzerException

class YOLOBaseAnalyzer(BaseAnalyzer):
    """
    Intermediate base class for YOLO-driven analysis.
    Handles YOLO model lifecycle, generic batched inference on GPU tensors, 
    and exposes preprocess/postprocess hooks for concrete analyzer classes.
    """

    def __init__(self,
                 frame_server: FrameServer,
                 model_path: str | Path,
                 imgsz: int = 640,
                 conf: float = 0.5,
                 **kwargs):
        """
        Initialize YOLO Base config.
        
        Args:
            frame_server (FrameServer): The FrameServer instance.
            model_path (str | Path): Path to the .pt YOLO model file.
            imgsz (int): Inference image size.
            conf (float): Confidence threshold for YOLO inference.
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
                extinfo_extractor (Optional[Callable[[NDArray,], Tuple[NDArray, List[Dict[str, Any]]]]]):
                    A callable object that accepts a batch of frames with extended 
                    info, returns a tuple of (NDArray of frames, ordered list of 
                    Dict of extended info). If provided, the list of extended info 
                    will be passed to `_preprocess()` and `_postprocess()` in 'ext_info'.
                inject_logger (Optional[Logger]): loguru logger instance.
        """
        # Enforce PyTorch Tensor config and GPU
        kwargs['tensor_type'] = TensorType.TORCH
        kwargs['device'] = DeviceType.CUDA if torch.cuda.is_available() else DeviceType.CPU
        super().__init__(frame_server=frame_server, **kwargs)
        
        self._model_path = model_path
        self._imgsz = imgsz
        self._conf = conf
        self._model: Optional[YOLO] = None

    def _initialize_analyzer(self):
        """Worker Process: Load YOLO model to target device."""
        device_str = 'cuda' if self._device == DeviceType.CUDA else 'cpu'
        self._model = YOLO(self._model_path)
        self._model.to(device_str)
        # Dummy warmup to allocate context
        dummy_input = torch.zeros((1, 3, self._imgsz, self._imgsz), device=device_str)
        self._model.predict(source=dummy_input, imgsz=self._imgsz, verbose=False)

    def _uninitialize_analyzer(self):
        """Worker Process: Cleanup model and release VRAM."""
        del self._model
        self._model = None
        if self._device == DeviceType.CUDA:
            torch.cuda.empty_cache()

    def _analyze(self, frames: torch.Tensor, ext_info: Optional[List[dict]] = None, **kwargs) -> Any:
        """
        Worker Process: Core execution skeleton.
        """
        if not isinstance(frames, torch.Tensor):
            raise AnalyzerException("YOLOBaseAnalyzer expects a torch.Tensor. Check tensor_type config.")

        # 1. Preprocess hook (e.g., format matching, tiling)
        # expected formatted_input shape: (B', C, H, W) normalized to 0.0-1.0
        # ext_info' will be None if `extinfo_extractor` is not provided.
        formatted_input, metadata = self._preprocess(frames, ext_info=ext_info, **kwargs)

        # 2. YOLO standard inference
        # model.predict supports list of results directly returning.
        results = self._model.predict(
            source=formatted_input, 
            imgsz=self._imgsz, 
            conf=self._conf, 
            verbose=False
        )

        # 3. Postprocess hook (e.g., remapping coordinates, logic extraction)
        return self._postprocess(results, frames, metadata, ext_info=ext_info, **kwargs)

    # --- Abstract Hooks for Leaf Classes ---
    @abstractmethod
    def _preprocess(self, frame: torch.Tensor, ext_info: Optional[List[dict]] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        """
        Preprocess NHWC tensor to NCHW normalized in 0.0-1.0 range. 
        Tiling can be done to extend N.
        Result and status handle functions, such as `_result_update` and `_status_update` 
        can be called in this method.

        Args:
            frame (torch.Tensor): NHWC tensor.
            ext_info (Optional[List[dict]]): List of extended information dicts
                if `extinfo_extractor` is provided, else None.
            kwargs (dict): kwargs dict from `.step()` if `continuous_mode=False`.
        
        Returns:
            Tuple[torch.Tensor, Any]: (NCHW Tensor in 0.0-1.0, Metadata to pass to `_postprocess`).

        Raises:
            AnalyzerException: If any error occurs during postprocessing.
        """
        raise NotImplementedError

    @abstractmethod
    def _postprocess(self, results: list, original_frames: torch.Tensor, 
                     metadata: Any, ext_info: Optional[List[dict]] = None, **kwargs) -> None:
        """
        Process YOLO results using metadata and update state/results.
        Result and status handle functions, such as `_result_update` and `_status_update` 
        can be called in this method.
        
        Args:
            results (list[Results]): YOLO results.
            original_frames (torch.Tensor): Original NHWC frame.
            metadata (Any): Metadata returned by `_preprocess`.
            ext_info (Optional[List[dict]]): List of extended information dicts 
                if `extinfo_extractor` is provided, else None.
            kwargs (dict): kwargs dict from `.step()` if `continuous_mode=False`.

        Raises:
            AnalyzerException: If any error occurs during postprocessing.
        """
        raise NotImplementedError

