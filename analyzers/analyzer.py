# analyzers/analyzer.py
# Refactored by Gemini 3.1 pro, Reviewed & Corrected by Haiyun Huang (260507)

import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.connection import Connection
import threading as t
import time
from abc import ABC, abstractmethod
from typing import Optional, Any, Union, Callable, Tuple, List, Dict, TypeVar, cast, TYPE_CHECKING
from functools import wraps
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING: import torch # Lazy import for env that don't have pytorch.
# CUDAPinner is lazy imported.

from loguru import logger as file_logger
from loguru._logger import Logger

from frameserver import FrameServer, FrameTicket
from .analyzer_types import TensorType, DeviceType, ConsumerMode, AnalyzerException

F = TypeVar('F', bound=Callable[..., Any])


class BaseAnalyzer(ABC):
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
        @wraps(func)
        def wrapper(self: "BaseAnalyzer", *args, **kwargs):
            if self._worker_enable.is_set() or self._worker_process is not None:
                cls_name = self.__class__.__name__
                func_name = func.__name__
                raise RuntimeError(f"Operation failed, Analyzer is currently running. src_func={cls_name}.{func_name}")
            return func(self, *args, **kwargs)
        return cast(F, wrapper)

    def __init__(self,
                 frame_server: FrameServer,
                 batch_size: int = 1,
                 tensor_type: TensorType = TensorType.NUMPY,
                 device: DeviceType = DeviceType.CPU,
                 consumer_mode: ConsumerMode = ConsumerMode.ASYNC,
                 continuous_mode: bool = True,
                 fetch_timeout: float = 0.1,        # Timeout for fetching a single frame.
                 stat_interval: float = 1.0,
                 extinfo_extractor: Optional[Callable[[NDArray,], Tuple[NDArray, List[Dict[str, Any]]]]] = None,
                 inject_logger: Optional[Logger] = None,
                 **kwargs):
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
        if not isinstance(frame_server, FrameServer):
            raise TypeError("frame_server must be a FrameServer instance.")
        
        self._frame_server: FrameServer = frame_server
        self._fs_cid: Optional[int] = None
        
        self._batch_size        = batch_size
        self._tensor_type       = tensor_type
        self._device            = device
        self._consumer_mode     = consumer_mode
        self._continuous_mode   = continuous_mode
        self._fetch_timeout     = fetch_timeout if consumer_mode == ConsumerMode.SYNC else 0.0 # get_async is non-blocking
        self._stat_interval     = stat_interval
        self._extinfo_extractor = extinfo_extractor
        
        if inject_logger is not None:
            if isinstance(inject_logger, Logger):
                self._logger = inject_logger
            else:
                raise TypeError("inject_logger must be a loguru.Logger instance.")
        else:
            self._logger = file_logger
        # Logger for base class.
        self.__logger: Logger = self._logger.bind(friendly_name="BaseAnalyzer")

        # Multiprocessing control signals
        self._worker_enable: mp_sync.Event = mp.Event()
        self._worker_ready: mp_sync.Event = mp.Event()
        self._worker_process: Optional[mp.Process] = None

        # IPC resources
        self._status: dict = {}
        self._status_lock = t.Lock()
        self._status_written = t.Condition(self._status_lock)
        self._response_thread: Optional[t.Thread] = None
        self._response_rx: Optional[Connection] = None
        self._response_tx: Optional[Connection] = None

        self.result_: dict = {}      # Public property, should be protected by result_lock_
        self.result_lock_ = t.Lock()
        self.result_written_ = t.Condition(self.result_lock_)

        self._cmd_rx: Optional[Connection] = None
        self._cmd_tx: Optional[Connection] = None

        self._init_ipc()

        # Statistics
        self._analyzed_count: mp.Value = mp.Value('i', 0, lock=False) # Only accessed by _worker

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove resources belonging to main process or unpicklable
        state.pop('_status_lock', None)
        state.pop('_status_written', None)
        state.pop('_response_thread', None)
        state.pop('_response_rx', None) # Subprocess only need tx end.
        
        state.pop('result_lock_', None)
        state.pop('result_written_', None)
        
        state.pop('_cmd_tx', None)    # Subprocess only need rx end.
        return state

    def _init_ipc(self):
        """Initialize all pipes and locks for IPC."""
        self._status.clear()
        self._status["analyzer_status"] = "idle"
        self._status["error"] = {}
        self.result_.clear()
        
        self._response_rx, self._response_tx = mp.Pipe(duplex=False) # write by base&sub class
        self._cmd_rx, self._cmd_tx = mp.Pipe(duplex=False)

    def _response_upd_thread(self):
        """Daemon thread loop for receiving status updates from child process."""
        logger = self.__logger
        rx_end = self._response_rx
        while True:
            try:
                if rx_end.poll(0.1):
                    target, command, data = rx_end.recv()
                    if target == 'status':
                        with self._status_lock:
                            if command == 'update':
                                self._status.update(data)
                            elif command == 'del':
                                for k in data:
                                    self._status.pop(k, None)
                            elif command == 'overwrite':
                                self._status.clear()
                                self._status.update(data)
                            elif command == 'subdict_update':
                                dict_name, content = data
                                self._status.setdefault(dict_name, {}).update(content)
                            self._status_written.notify_all()
                    elif target == 'result':
                        with self.result_lock_:
                            if command == 'update':
                                self.result_.update(data)
                            elif command == 'del':
                                for k in data:
                                    self.result_.pop(k, None)
                            elif command == 'overwrite':
                                self.result_.clear()
                                self.result_.update(data)
                            self.result_written_.notify_all()
            except EOFError:
                logger.debug("Response receiver exited due to Tx EOF.")
                break
            except OSError:
                logger.debug("Response receiver exited due to Rx closed.")
                break
            except Exception as e:
                logger.opt(exception=e).error("Error in daemon thread for subprocess response.")
                break

    def _start_ipc_threads(self):
        self._response_thread = t.Thread(target=self._response_upd_thread, daemon=True, name="AnalyzerResponseUpd")
        self._response_thread.start()

    def _stop_ipc(self):
        # Tx process end, thus it's safe to close tx.
        # When close rx, OSError will raise during poll.
        if self._response_tx: self._response_tx.close(); self._response_tx = None
        if self._response_rx: self._response_rx.close(); self._response_rx = None

        if self._cmd_tx: self._cmd_tx.close(); self._cmd_tx = None
        if self._cmd_rx: self._cmd_rx.close(); self._cmd_rx = None

        if self._response_thread and self._response_thread.is_alive():
            self._response_thread.join(timeout=1.0)
            if self._response_thread.is_alive():
                self.__logger.warning("Status updater thread failed to join.")
                # keep reference for future handling if needed (currently none)
            else: self._response_thread = None

    # --- Properties and Getters for Main Process ---
    
    @property
    def status(self) -> dict:
        """Get current analyzer status copy. Synchronized from child process."""
        with self._status_lock:
            return self._status.copy()

    @property
    def is_ready(self) -> bool:
        return self._worker_ready.is_set() and self._worker_enable.is_set()

    def get_result(self, key: Any, timeout: Optional[float] = 0) -> Any:
        """
        Get the value for a given key from the result dict.

        .. note:: The result is a view, and will not be removed after retrieval, 
            which is different from `get_error()`.

        :param key: (Any), key to get.
        :param timeout: (Optional[float]), timeout for waiting result. 
            0 = non-blocking, None = blocking.
        :return: (Any), value of the key, None if not found after timeout.
        """
        with self.result_lock_:
            if self.result_written_.wait_for(
              lambda: ((key in self.result_) or (not self.is_ready)), timeout=timeout):
                return self.result_.get(key, None)
            return None

    def get_results(self) -> dict:
        """Get a copy of the accumulated results."""
        with self.result_lock_:
            return self.result_.copy()

    def get_latest_result(self) -> Any:
        """
        Get the most recently **added** result (dict insertion order).
        Require python 3.8+. Be care of TOCTOU problem.
        """
        with self.result_lock_:
            return next(reversed(self.result_.values()), None)

    # --- Control API for Main Process ---

    def send_command(self, cmd_name: str, payload: Any):
        """Send an arbitrary command for subclass to the worker process."""
        if self._cmd_tx:
            self._cmd_tx.send((cmd_name, payload))

    def step(self, **kwargs) -> int:
        """
        Trigger a single analysis step asynchronously. To check the result, 
        see `get_error()`.

        :params kwargs: Parameters dict that will be added to kwargs that passed
            to the `_analyze()`.
        :return: `step_id` to query error status in get_error(step_id). 
            `-1` if `continuous_mode` is True.
        """
        if self._continuous_mode:
            self.__logger.warning("step() called but analyzer is in continuous mode. Ignoring.")
            return -1
        step_id = time.time_ns()
        if self._cmd_tx:
            self._cmd_tx.send(('step', (step_id, kwargs)))
        return step_id

    def get_error(self, step_id: int = -1, timeout: Optional[float] = 0) -> Optional[str]:
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
        with self._status_lock:
            if self._status_written.wait_for(
              lambda: ((step_id in self._status.get("error", {})) or (not self.is_ready)), timeout=timeout):
                errors: dict = self._status.get("error", {})
                # wait_for can be unblocked by (not self.is_ready) the id can be still not available.
                return errors.pop(step_id, None) 

            # wait_for timeout
            return None 

    # ====== Methods for Subprocess ======
    # --- Abstract Methods ---

    @abstractmethod
    def _initialize_analyzer(self):
        """Initialize the specific analyzer within the worker process."""
        pass

    @abstractmethod
    def _uninitialize_analyzer(self):
        """Clean up analyzer resources within the worker process."""
        pass

    @abstractmethod
    def _analyze(self, frame: Union[np.ndarray, "torch.Tensor"], ext_info: Optional[List[dict]] = None, **kwargs) -> Any:
        """
        Analyze a frame.
        Args:
            frame: NDArray or Torch Tensor depending on configuration.
            ext_info: List of extended information dicts for the frames.
            kwargs: Parameters passed via step() if continuous_mode=False.
        """
        pass

    @abstractmethod
    def _handle_command(self, cmd_name: str, payload: Any):
        """
        Handle a custom command. 
        WARNING: This is called from a background thread inside the worker process
        and WILL execute concurrently with `_analyze`. Subclasses must implement thread locks 
        for any shared resources.
        """
        pass

    # --- Methods for Subclasses in Worker Process ---

    def _status_update(self, data: dict):
        """Subclass specific method to update status."""
        with self._response_tx_lock: self._response_tx.send(('status', 'update', data))

    def _status_del(self, keys: list):
        with self._response_tx_lock: self._response_tx.send(('status', 'del', keys))

    def _status_overwrite(self, data: dict):
        with self._response_tx_lock: self._response_tx.send(('status', 'overwrite', data))
    
    def _status_subdict_update(self, dict_name: str, data: dict):
        with self._response_tx_lock: self._response_tx.send(('status', 'subdict_update', (dict_name, data)))

    def _result_update(self, data: dict):
        """Subclass specific method to update results."""
        with self._response_tx_lock: self._response_tx.send(('result', 'update', data))

    def _result_del(self, keys: list):
        with self._response_tx_lock: self._response_tx.send(('result', 'del', keys))

    def _result_overwrite(self, data: dict):
        with self._response_tx_lock: self._response_tx.send(('result', 'overwrite', data))

    # --- Worker Process Core Logic ---

    def _cmd_listener_thread(self):
        """Background thread in worker process to poll for commands concurrently."""
        while self._worker_enable.is_set():
            try:
                if self._cmd_rx.poll(0.5):
                    command, payload = self._cmd_rx.recv()
                    if command == 'step':
                        step_id, kwargs = payload
                        with self._step_cond:
                            if self._step_pending:
                                # Discard and report error.
                                self._status_subdict_update('error', 
                                    {step_id: "Dropped: Analyzer is busy processing a previous step."})
                            else:
                                self._step_id = step_id
                                self._step_kwargs = kwargs
                                self._step_pending = True
                                self._step_cond.notify_all()
                    else:
                        try:
                            self._handle_command(command, payload)
                        except Exception as e:
                            self.__logger.opt(exception=e).error(f"Error handling command {command} in subclass.")
            except (EOFError, BrokenPipeError):
                break
            except Exception as e:
                self.__logger.opt(exception=e).error("Unexpected error in command listener thread.")

    def _worker(self, cid: int):
        """Main worker process loop."""
        self._worker_ready.clear()
        pid = mp.current_process().pid
        logger = self.__logger
        frame_server = self._frame_server

        # Subprocess-specific instance properties
        self._response_tx_lock = t.Lock()
        self._step_cond: t.Condition = t.Condition()
        self._step_id: int = -1
        self._step_kwargs: dict = {}
        self._step_pending: bool = False
        
        # Initialize analyzer
        try:
            logger.info("Initializing subclass...")
            self._initialize_analyzer()
            logger.success("Subclass initialized.")
        except Exception as e:
            logger.opt(exception=e).error("Subclass initialization failed.")
            return

        pinner = None
        # Import pytorch if needed
        if self._tensor_type == TensorType.TORCH:
            import torch
            # Setup CUDAPinner which can accelerate CUDA DMA.
            if self._device == DeviceType.CUDA:
                try:
                    from utils.cuda_shm_pinner_v2 import CUDAPinner
                    pinner = CUDAPinner()
                    pinner.pin(frame_server.buffer._data_shm) # Must use the same virtual address to accelerate DMA
                    logger.info("Registered FrameServer ShM in CUDA driver.")
                except ImportError:
                    logger.warning("utils.cuda_shm_pinner module not found. Data transfer speed degraded.")
                except Exception as e:
                    logger.opt(exception=e).error("Failed to register FrameServer ShM in CUDA driver.")

        # Setup Concurrent Command Listener
        cmd_thread = t.Thread(target=self._cmd_listener_thread, daemon=True, name="CmdListener")
        cmd_thread.start()

        # Frame counting for statistics (status)
        frame_count_since_last_check = 0
        time_last_check = time.monotonic()

        # Event loop local variables
        last_ticket: Optional[FrameTicket] = None
        ticket:      Optional[FrameTicket] = None

        # Main Event Loop
        try:
            self._worker_ready.set()
            self._status_update({'analyzer_status': 'busy' if self._continuous_mode else 'ready'})
            
            while self._worker_enable.is_set():
                kwargs: dict = {}
                step_id: int = -1
                
                # Check execution mode
                if not self._continuous_mode:
                    with self._step_cond:
                        # Wait for step instruction with timeout to check shutdown flag
                        if not self._step_cond.wait_for(lambda: self._step_pending or (not self._worker_enable.is_set()), timeout=0.1):
                            continue # Timeout, loop again
                        if not self._worker_enable.is_set():
                            break # Exiting
                        kwargs = self._step_kwargs
                        step_id = self._step_id
                    
                    self._status_update({'analyzer_status': 'busy'})
                
                # Fetch Frame
                try:
                    if self._consumer_mode == ConsumerMode.SYNC:
                        ticket = frame_server.get_sync(cid, self._batch_size, timeout=self._fetch_timeout)
                        if last_ticket is not None: frame_server.release_sync(last_ticket)
                    else:
                        ticket = frame_server.get_async(self._batch_size) # TODO: get_async_copy?
                except Exception as e:
                    logger.opt(exception=e).error("Error fetching frame from FrameServer.")
                    if not self._continuous_mode:
                        self._status_subdict_update('error', {step_id: f"Error fetching frame: {e}"})
                        with self._step_cond: self._step_pending = False
                        self._status_update({'analyzer_status': 'ready'})
                    else:
                        self._status_subdict_update('error', {-1: f"Error fetching frame: {e}"})
                    continue

                if ticket is None:
                    if not self._continuous_mode:
                        self._status_subdict_update('error', {step_id: f"Timeout ({self._fetch_timeout}s) fetching frame."})
                        with self._step_cond: self._step_pending = False
                        self._status_update({'analyzer_status': 'ready'})
                    continue # No data available yet
                
                frames_list = frame_server.get_from_ticket(ticket, timeout=0.1)
                if frames_list:
                    extinfo_list = None
                    if self._extinfo_extractor is not None:
                        extinfo_list = []
                        for i in range(len(frames_list)):
                            frames_list[i], extinfo_block = self._extinfo_extractor(frames_list[i])
                            # frames_list now is pure frame without extra_line. 
                            # This step is usually zero-copy, depends on the impl of Extractor.
                            extinfo_list.extend(extinfo_block)
                    
                    # Handle Frame Tensor / Array wrapping logic
                    if len(frames_list) == 1:
                        data = frames_list[0]
                        if self._tensor_type == TensorType.TORCH:
                            data = torch.from_numpy(data)
                            # Tensor Conversion (DMA if pinned, no host mem copy)
                            if self._device == DeviceType.CUDA:
                                data = data.to('cuda', non_blocking=True) # TODO: will it crash when other process write the address?
                    else:
                        block1, block2 = frames_list
                        if self._tensor_type == TensorType.TORCH:
                            t1 = torch.from_numpy(block1)
                            t2 = torch.from_numpy(block2)
                            if self._device == DeviceType.CUDA:
                                # Pre-allocate on GPU and use 2 async copy_ to avoid host concatenation & slow transfer
                                data = torch.empty((self._batch_size, *block1.shape[1:]), dtype=t1.dtype, device='cuda')
                                data[:block1.shape[0]].copy_(t1, non_blocking=True)
                                data[block1.shape[0]:].copy_(t2, non_blocking=True)
                            else:
                                data = torch.cat([t1, t2], dim=0)
                        else:
                            data = np.concatenate(frames_list, axis=0)

                    # Execute Analysis
                    try:
                        self._analyze(data, ext_info=extinfo_list, **kwargs)
                        frame_count_since_last_check += self._batch_size
                        self._analyzed_count.value += self._batch_size
                        
                        # Statistics Interval Update
                        current_time = time.monotonic()
                        elapsed = current_time - time_last_check
                        if elapsed >= self._stat_interval:
                            fps = frame_count_since_last_check / elapsed
                            self._status_update({"frame_count": self._analyzed_count.value, "fps": fps})
                            frame_count_since_last_check = 0
                            time_last_check = current_time
                        
                        if not self._continuous_mode:
                            self._status_subdict_update('error', {step_id: "success"})

                    except AnalyzerException as e:
                        logger.opt(exception=e).error("Subclass error during analysis.")
                        self._status_subdict_update('error', {step_id if not self._continuous_mode else -1: f"Analysis error: {e}"})
                    except Exception as e:
                        logger.opt(exception=e).error("Unexpected error during frame analysis.")
                        self._status_subdict_update('error', {step_id if not self._continuous_mode else -1: f"Unexpected analysis error: {e}"})

                if self._consumer_mode == ConsumerMode.SYNC:
                    last_ticket = ticket # defer release to next iteration
                
                if not self._continuous_mode: 
                    with self._step_cond: self._step_pending = False
                    # Due to pipe, `ready` label here can ensure `error` is received.
                    self._status_update({'analyzer_status': 'ready'}) 
        finally:
            logger.debug("Analyzer worker exited working loop, cleaning up...")
            self._status_update({'analyzer_status': 'idle'})
            self._worker_ready.clear()
            if pinner:
                try:
                    pinner.unpin(frame_server.buffer._data_shm)
                except Exception as e:
                    logger.warning(f"Error unpinning memory: {e}")
            try:
                self._uninitialize_analyzer()
            except Exception as e:
                logger.opt(exception=e).error("Analyzer worker failed during uninitialization.")
            
            if self._consumer_mode == ConsumerMode.SYNC:
                if ticket is not None: frame_server.release_sync(ticket)
                if last_ticket is not None: frame_server.release_sync(last_ticket)
            
            frame_server.close()
            logger.success("Analyzer worker cleanup completed.")
            logger.complete() # Flush logging buffer

    # ======= Control API for Main Process =======

    def start(self):
        """
        Starts the analyzer background worker process.
        It is not safe to call `start` and `stop` concurrently.
        """
        logger = self.__logger
        if self._worker_enable.is_set():
            logger.error("Analyzer already running.")
            return

        logger.info("Starting BaseAnalyzer...")
        
        if self._consumer_mode == ConsumerMode.SYNC and self._fs_cid is None:
            try:
                self._fs_cid = self._frame_server.register_consumer(historical_data=True)
            except RuntimeError as e:
                logger.error(f"Failed to register consumer in frame server: {e}")
                return
        elif self._consumer_mode == ConsumerMode.ASYNC:
            self._fs_cid = -1 # Or any dummy value for async

        try:
            self._worker_enable.set()
            
            self._init_ipc()
            self._start_ipc_threads()
            
            logger.debug("Starting worker process...")
            wp = mp.Process(target=self._worker, args=(self._fs_cid,), name="BaseAnalyzerWorker")
            self._worker_process = wp
            wp.start()

            if not self._worker_ready.wait(timeout=10.0):
                logger.error(f"Timeout waiting for analyzer worker ({wp.pid}) to become ready.")
                self.stop()
                raise TimeoutError("Analyzer worker failed to initialize within timeout.")
                
            logger.success(f"BaseAnalyzer started successfully with worker PID: {wp.pid}")

        except Exception as e:
            logger.opt(exception=e).error("Error starting BaseAnalyzer.")
            self.stop()
            raise

    def stop(self, terminate_timeout=10.0):
        """
        Stops the analyzer and cleans up resources. Will not consume remaining 
        frames in the buffer (equivalent to `eager_stop` in `VideoEncoder`).
        
        It is not safe to call `start` and `stop` concurrently.
        
        :param terminate_timeout: (float) Timeout in seconds to wait for the 
            worker process to force terminate.
        :raises TimeoutError: If the worker process does not terminate within 
            the specified timeout.
        """
        logger = self.__logger
        wp = self._worker_process
        
        if wp is None and self._fs_cid is None:
            logger.info("BaseAnalyzer already stopped.")
            return

        if wp is not None:
            logger.info(f"Stopping BaseAnalyzer worker (PID {wp.pid})...")
            self._worker_enable.clear()

            if wp.is_alive():
                wp.join(timeout=terminate_timeout)
                if wp.is_alive():
                    logger.warning(f"Worker {wp.pid} did not exit within timeout. Terminating...")
                    wp.terminate()
                wp.join(timeout=1.0)
                if wp.is_alive():
                    raise TimeoutError(f"Worker process {wp.pid} did not terminate within timeout.")
                logger.info(f"Worker process {wp.pid} joined.")

            self._worker_process = None
        # All pipe tx finished, safely stop IPC.
        self._stop_ipc()
        
        if self._fs_cid is not None and self._fs_cid >= 0:
            self._frame_server.unregister_consumer(self._fs_cid)
            self._fs_cid = None

        logger.success("BaseAnalyzer stopped.")
