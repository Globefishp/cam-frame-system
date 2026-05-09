import abc
from .analyzer_types import AnalyzerException as AnalyzerException, ConsumerMode as ConsumerMode, DeviceType as DeviceType, TensorType as TensorType
from _typeshed import Incomplete
from abc import ABC
from frameserver import FrameServer, FrameTicket as FrameTicket
from loguru._logger import Logger
from multiprocessing.shared_memory import SharedMemory as SharedMemory
from numpy.typing import NDArray as NDArray
from typing import Any, Callable, TypeVar

F = TypeVar('F', bound=Callable[..., Any])

class BaseAnalyzer(ABC, metaclass=abc.ABCMeta):
    """
    Abstract base class for a high-performance Analyzer using multiprocessing.

    Features:
        - FrameServer V3 Integration (sync or async consumption).
        - Configurable data interface: Numpy or pytorch (CPU or GPU tensor), maximizing data 
          throughput by using DMA on page-locked memory.
        - Two execution modes:
            continuous_mode=True: Auto-loop, fetching and analyzing frames continuously.
            continuous_mode=False: Step mode, waiting for explicit `step()` calls.
        - Concurrent command handling: `_handle_command` is executed in a background thread
          within the worker process, allowing it to run concurrently with `_analyze`.
          Subclasses MUST USE LOCKS when accessing shared resources in these functions.
    """
    @staticmethod
    def require_stop(func: F) -> F:
        """Decorator to ensure the analyzer is stopped before calling the method."""
    result_: dict
    result_lock_: Incomplete
    result_written_: Incomplete
    def __init__(self, frame_server: FrameServer, batch_size: int = 1, tensor_type: TensorType = ..., device: DeviceType = ..., consumer_mode: ConsumerMode = ..., continuous_mode: bool = True, fetch_timeout: float = 0.1, stat_interval: float = 1.0, extinfo_extractor: Callable[[NDArray], tuple[NDArray, list[dict[str, Any]]]] | None = None, inject_logger: Logger | None = None, **kwargs) -> None:
        """
        Initialize BaseAnalyzer resource management.

        Args:
            frame_server (FrameServer): The frame server instance to get frame from.
            tensor_type (TensorType): Specify the data structure that will be 
                accepted by `_analyze()`
            device (DeviceType): If `tensor_type=TensorType.TORCH`, it is configurable
                where the tensor is stored when passed to `_analyze()`. `DeviceType.CUDA` 
                will accelerate further GPU operations. If `tensor_type=TensorType.NUMPY`, 
                this argument has no effect.
            consumer_mode (ConsumerMode): Control the behaviour to fetch a frame. 
                `ConsumerMode.SYNC` will ensure no frame skipping, but also have the risk
                of filling up the buffer if the analyzer is too slow; `ConsumerMode.ASYNC` 
                will always fetch the newest frame non-blockingly to avoid buffer overflow.
            continuous_mode (bool): Control the behaviour of the worker loop. 
                `continuous_mode=True` will fetch and analyze frames continuously. 
                `continuous_mode=False` will wait for explicit `step()` calls.
            fetch_timeout (float): Timeout for fetching a single frame.
            stat_interval (float): Interval for statistics update.
            extinfo_extractor (Optional[Callable[[NDArray,], Tuple[NDArray, List[Dict[str, Any]]]]]):
                A callable object that accepts a batch of frames with extended info, 
                returns a tuple of (NDArray of frames, ordered list of Dict of extended info).
                If provided, the list of extended info will be add to kwargs that pass in 
                `_analyze()`, as key='ext_info'.
            inject_logger (Optional[Logger]): Loguru logger instance.
        """
    @property
    def status(self) -> dict:
        """Get current analyzer status copy. Synchronized from child process."""
    @property
    def is_ready(self) -> bool: ...
    def get_result(self, key: Any, timeout: float | None = 0) -> Any:
        """
        Get the value for a given key from the result dict.

        .. note:: The result is a view, and will not be removed after retrieval, 
            which is different from `get_error()`.

        :param key: (Any), key to get.
        :param timeout: (Optional[float]), timeout for waiting result. 
            0 = non-blocking, None = blocking.
        :return: (Any), value of the key, None if not found after timeout.
        """
    def get_results(self) -> dict:
        """Get a copy of the accumulated results."""
    def get_latest_result(self) -> Any:
        """
        Get the most recently **added** result (dict insertion order).
        Require python 3.8+. Be care of TOCTOU problem.
        """
    def send_command(self, cmd_name: str, payload: Any):
        """Send an arbitrary command for subclass to the worker process."""
    def step(self, **kwargs) -> int:
        """
        Trigger a single analysis step asynchronously. To check the result, 
        see `get_error()`.

        :params kwargs: Parameters dict that will be added to kwargs that passed
            to the `_analyze()`.
        :return: `step_id` to query error status in get_error(step_id). 
            `-1` if `continuous_mode` is True.
        """
    def get_error(self, step_id: int = -1, timeout: float | None = 0) -> str | None:
        """
        Get the error message for a specific step_id or -1 (continuous mode).

        .. note:: 
            - If query succeeded, the event will be removed. Thus cannot be queried twice.
            - In continuous mode, only failure will be reported.
            - `get_error()` is now synchronized with the `Analyzer.result_`. If step_id
              succeeded, the results dict is guaranteed to be updated.
        
        :param step_id: (int), the step_id to query error status. 
            -1 if `continuous_mode` is True.
        :param timeout: (float), timeout for waiting error status. 
            0 = non-blocking, None = blocking.
        :return: Error message string, or 'success' if no error. None if 
            timeout waiting specified id, or id is not available after worker stopped.
        """
    def start(self) -> None:
        """
        Starts the analyzer background worker process.
        It is not safe to call `start` and `stop` concurrently.
        """
    def stop(self, terminate_timeout: float = 10.0) -> None:
        """
        Stops the analyzer and cleans up resources. Will not consume remaining 
        frames in the buffer (equivalent to `eager_stop` in `VideoEncoder`).
        
        It is not safe to call `start` and `stop` concurrently.
        
        :param terminate_timeout: (float) Timeout in seconds to wait for the 
            worker process to force terminate.
        :raises TimeoutError: If the worker process does not terminate within 
            the specified timeout.
        """
