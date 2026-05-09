import abc
from .videoencoder_types import EncoderException as EncoderException
from abc import ABC
from frameserver import FrameServer
from loguru._logger import Logger
from numpy.typing import NDArray as NDArray
from typing import Any, Callable, TypeVar

F = TypeVar('F', bound=Callable[..., Any])

class BaseVideoEncoder(ABC, metaclass=abc.ABCMeta):
    """
    Abstract base class for a video encoder using multiprocessing.


    Handles IPC resource creation, process management, frame processing and logging
    from a shared buffer.
    Requires subclasses to implement the encoder initialization and frame encoding logic.
    Use `self._logger` in subclass implememted methods if you want.

    Cannot do standard picklization due to internal IPC resources.
    Probably process-safe, but not tested.
    """
    @staticmethod
    def require_stop(func: F) -> F:
        """Decorator to ensure the encoder is stopped before calling the method."""
    def __init__(self, frame_server: FrameServer, output_path: str, batch_size: int = 5, target_fps: float | None = None, stat_interval: float = 1.0, extinfo_extractor: Callable[[NDArray], tuple[NDArray, list[dict[str, Any]]]] | None = None, inject_logger: Logger | None = None, **kwargs) -> None:
        """
        Initialize the base encoder with a pre-created shared buffer instance.

        Args:
            frame_server: (FrameServer), the frame server instance to get frame from.
                The VideoEncoder instance will be registered as a strict consumer.
            output_path: (str), path to the output video file.
            batch_size: (int), number of frames to get from the buffer at once. 
                Defaults to 5.
            target_fps: (Optional[float]), target encoding speed in fps. Used in speed
                warning for base class. Subclass may also use it. Defaults to None.
            stat_interval: (float), interval between status updates and warning checks
                in seconds. Defaults to 1.0.
            extinfo_extractor: (Optional[Callable[[NDArray,], tuple[NDArray, List[Dict[str, Any]]]]]), 
                the extended info extractor instance for extracting timecodes from 
                frames. if None, timecodes will not be extracted. Default to None. 
            inject_logger: (Optional[Logger]), the loguru logger instance for logging.
                enqueue=True is required. if None, a default logger is acquired 
                from *current process* (i.e. could differ from the logger instance 
                from `main` if current processs is not `main`).
        """
    @property
    def status(self) -> dict:
        """Current encoder status, read-only. 
        Will be syncronized between parent and child processes."""
    @property
    def is_ready(self) -> bool:
        """Checks if the encoder worker process has signaled it's ready."""
    @property
    def frame_encoded(self) -> int:
        """Return encoded frame count. Process-safe."""
    @property
    def output_path(self) -> str:
        """The output path of the video encoder."""
    @output_path.setter
    @require_stop
    def output_path(self, value: str):
        """Set the output path. Only allowed when the encoder is stopped."""
    def start(self) -> None:
        """
        Launches the background worker process and starts the video encoder.

        Does nothing if the encoder is already running.
        It is not safe to call `start` and `stop` concurrently.
        """
    def stop(self, resumable: bool = False, exit_timeout=None, terminate_timeout: float = 20.0):
        """
        Stops the video encoder.
            Signals the worker process to stop, waits for it to join,
            and cleans up the created IPC resources (shared ring buffer).
        Overwrite it if you want to control timeout. Usually terminate_timeout is
            set to a large value.

        Does nothing if the encoder is already stopped.
        It is not safe to call `start` and `stop` concurrently.

        Args:
            resumable (bool): If True, will not release the consumer from frame 
                server, so the encoder can be resumed later seamlessly. 
                Useful for file rotation. To fully release resources, call `stop()`
                with `resumable=False` again. Default False.
            exit_timeout (float): Timeout for the worker process to exit. Default None
                represents auto determined by buffer size and `target_fps`. If not
                available, use 10s.
            terminate_timeout (float): Timeout for the worker process to be terminated.

        Raises:
            TimeoutError: If the worker process does not terminate within the timeout.
        """
