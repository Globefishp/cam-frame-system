import abc
import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.sharedctypes import Synchronized
from typing import Optional, Any, Tuple
import numpy as np
import time # Import time for perf_counter
import ctypes # Import ctypes for memory operations

class NNAnalyzer(abc.ABC):
    '''
    Abstract base class for a neural network analyzer using multiprocessing.
    Handles IPC resource creation, process management, and communication.
    Requires subclasses to implement the analysis logic.
    '''
    # Define appended submission timestamp bytes and data type
    _submission_timestamp_dtype: np.dtype = np.dtype("uint64") # Use uint64 for microsecond timestamp

    # 1. 修改 __init__ 参数，移除 IPC 资源
    def __init__(self,
                 model_path: str,
                 frame_shape: tuple[int, int, int],
                 frame_dtype: np.dtype = np.dtype("uint8")):
        '''
        Initialize the base analyzer. IPC resources will be created in start().
        Args:
            model_path: (str), path to the model file (used by subclasses).
            frame_shape: (tuple[int, int, int]), expected frame size (height, width, channel).
            frame_dtype: (np.dtype), expected frame data type.
        '''
        self.model_path = model_path
        self.frame_shape = frame_shape
        self.frame_dtype = np.dtype(frame_dtype) # Store frame dtype
        # 2. 在内部创建 Queue
        self.result_queue = mp.Queue()
        # 3. 初始化 IPC 资源为 None，将在 start() 中创建
        self.shm: Optional[SharedMemory] = None
        self.shm_cond: Optional[mp_sync.Condition] = None
        # worker processes
        self.worker_processes = []
        # flags
        self.working = mp.Event()
        # 将 initialized 事件改为 worker_waiting, 表示 worker 正在等待帧
        self.worker_ready = mp.Event()
        # Shared flag to indicate if shared memory is filled with a frame
        self._shm_is_filled: Optional[Synchronized[bool]] = None
        # Model placeholder - will be loaded by subclass in worker process via _initialize_worker
        self.model = None

        # Calculate total bytes including appended submission timestamp
        self._frame_data_bytes = int(np.prod(self.frame_shape) * self.frame_dtype.itemsize)
        self._total_frame_bytes = int(self._frame_data_bytes + self._submission_timestamp_dtype.itemsize)

    @abc.abstractmethod
    def _initialize_analyzer(self):
        """
        Initialize the worker process. This method MUST be implemented by
        subclasses and is called once when the worker process starts.
        It should handle model loading, warmup, and setting self.model.
        It should be a blocking function, return when is fully ready.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _analyze(self, frame: np.ndarray) -> Any:
        """
        Analyze the frame using the specific NN model.
        This method MUST be implemented by subclasses.
        Args:
            frame: (np.ndarray), the input frame to analyze.
        Returns:
            Analysis result.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def _uninitialize_analyzer(self):
        """
        Uninitialize the analyzer. This method MUST be implemented by
        subclasses and is called once when the worker process ends. 
        It should handle any cleanup tasks, and be a blocking function.
        """
        raise NotImplementedError


    def submit_frame(self, frame: np.ndarray, timeout: Optional[float] = None) -> Optional[int]:
        """
        Submits a frame for analysis by copying it to shared memory.
        This method will block and wait for the shared memory buffer to be empty
        before writing the new frame. Appends the submission timestamp
        in microseconds to the frame data in shared memory.
        Args:
            frame: (np.ndarray), a frame to submit (original image data).
            timeout: (Optional[float]), seconds to wait for the shared memory buffer to be empty.
                     If None, waits indefinitely.
                     If 0, returns immediately if the buffer is not empty.
        Returns:
            The submission timestamp in microseconds if successful, None if in other case (e.g., timeout).
        """
        # Get submission timestamp in microseconds
        submission_timestamp_us = time.perf_counter_ns() // 1000
        print(f"Submitting frame with submission timestamp (us): {submission_timestamp_us}") # Debug print

        # 4. 添加检查确保 IPC 资源已初始化
        if not self.working.is_set() or self.shm is None or self.shm_cond is None:
            print("Warning: NNAnalyzer is not running or IPC not initialized, cannot submit frame.")
            print("Please call start() before submit_frame().")
            return None

        # 检查 worker 是否正在等待帧
        if not self.worker_ready.is_set():
            print("Warning: NNAnalyzer worker is not waiting for frames yet, Waiting...")
            if timeout is not None:
                if not self.worker_ready.wait(timeout=timeout):
                    print("Warning: Timeout waiting for worker to be ready.")
                    return None
            else:
                self.worker_ready.wait()


        try:
            # Use condition variable to wait for shared memory to be empty
            with self.shm_cond:
                # Wait for _shm_is_filled to be False (shared memory is empty)
                # Use wait_for with timeout, which waits until predicate becomes True.
                # Will not affect much by Condition.notify(). Will also wait for the lock
                # with timeout after predicate becomes True.
                notified = self.shm_cond.wait_for(lambda: not self._shm_is_filled.value, timeout=timeout)

                if not notified:
                    # Timeout occurred while waiting for shared memory to be empty
                    print(f"Warning: submit_frame timed out after {timeout} seconds while waiting for shared memory to be empty.")
                    return None

                # Shared memory will be filled, set _shm_is_filled to True
                self._shm_is_filled.value = True

                # Copy frame data to shared memory
                # 1. Create a NumPy array view for the image data part
                dest = np.ndarray(self.frame_shape,
                                dtype=self.frame_dtype, # Use frame_dtype
                                buffer=self.shm.buf)
                # Ensure frame dimensions and dtype match
                if frame.shape != self.frame_shape or frame.dtype != self.frame_dtype:
                    print(f"Error in submitting a frame to NNAnalyzer: Frame shape/dtype mismatch. Expected {self.frame_shape} {self.frame_dtype}, got {frame.shape} {frame.dtype}")
                    # Need to reset _shm_is_filled if we don't write
                    self._shm_is_filled.value = False
                    self.shm_cond.notify() # Notify worker in case it was waiting
                    return None

                np.copyto(dest, frame)

                # 2. Create a NumPy array view for the timestamp part
                #    The buffer is self.shm.buf, offset is self._frame_data_bytes,
                #    shape is (1,) because it's a single timestamp, dtype is self._submission_timestamp_dtype.
                shm_timestamp_array = np.ndarray(
                    (1,), # Shape for a single scalar value
                    dtype=self._submission_timestamp_dtype,
                    buffer=self.shm.buf,
                    offset=self._frame_data_bytes
                )
                # 3. Write the timestamp directly
                shm_timestamp_array[0] = submission_timestamp_us

                # Notify the worker process that a new frame is available
                self.shm_cond.notify()

        except Exception as e:
            print(f"Error submitting frame: {e}")
            # In case of other errors, try to reset the flag and notify
            # We need to acquire the lock here before accessing _shm_is_filled and notifying
            if self.shm_cond: # Check if shm_cond was initialized
                 with self.shm_cond:
                      if self._shm_is_filled and self._shm_is_filled.value: # Check if _shm_is_filled was initialized and is True
                           self._shm_is_filled.value = False
                           self.shm_cond.notify()
            return None

        return submission_timestamp_us


    def get_result(self, timeout: Optional[float] = None) -> Optional[Tuple[Any, int, int]]:
        """
        Retrieves the next analysis result from the result queue.
        Args:
            timeout: (Optional[float]), seconds to wait for a result.
                     If None, waits indefinitely.
                     If 0, returns immediately (may raise Empty exception).
        Returns:
            A tuple containing (analysis result, submission timestamp in us, total analysis duration in us),
            or potentially raises queue.Empty, or returns None on timeout (if implemented).
        """
        try:
            # get() will raise queue.Empty if timeout is 0 and queue is empty
            # If timeout is None, it waits indefinitely
            # If timeout is > 0, it waits for that many seconds
            result = self.result_queue.get(timeout=timeout)
            print(f"Received result with timestamp (us): {result[1]}, analysis duration: {result[2]} us.") # Debug print
            return result
        except mp.queues.Empty:
            # Handle timeout specifically if needed, though get() with timeout=0 handles it
            # For timeout > 0, get() returns None on timeout
            return None
        except Exception as e:
            print(f"Error getting result: {e}")
            return None


    def _worker(self):
        """
        The main function executed by this background worker process.
        This method first calls the subclass's `_initialize_worker` method
        to load the model and perform any necessary setup. Then, it enters
        a loop where it waits for the shared memory to be filled with a new frame
        (indicated by the `_shm_is_filled` flag), reads the frame data from 
        shared memory (`shm`), calls the subclass's `analyze` method, and 
        puts the result into the shared result queue (`result_queue`). After
        processing, it signals that the shared memory is empty.

        The loop continues until the `working` event is cleared by the main process.
        Includes error handling for initialization and per-frame processing.
        """
        self.worker_ready.clear() # 确保最开始没有在waiting
        try:
            # 1. 初始化 (只执行一次)
            print(f"NNAnalyzer worker process ({mp.current_process().pid}) initializing...")
            self._initialize_analyzer()
            print(f"NNAnalyzer worker process ({mp.current_process().pid}) initialized successfully.")
        except Exception as e:
            print(f"FATAL: NNAnalyzer worker process ({mp.current_process().pid}) failed to initialize: {e}")
            # 初始化失败，工作进程无法继续
            return # 退出 worker 函数

        # 2. 主处理循环
        self.worker_ready.set() # 将worker_ready信号在此处置True，后面的流将由shm_cond和_shm_is_filled控制
        try:
            while self.working.is_set():
                try:
                    frame_to_analyze = None
                    submission_timestamp_us = None
                    with self.shm_cond:  # Acquire condition lock
                        # Wait for _shm_is_filled to be True (shared memory is filled)
                        # Add a timeout to wait_for to prevent infinite blocking if working is cleared
                        notified = self.shm_cond.wait_for(lambda: self._shm_is_filled.value, timeout=1.0)

                        if not notified:
                            # Timeout occurred while waiting for shared memory to be filled
                            continue # Continue loop to check working.is_set()

                        if not self.working.is_set():
                            # Working event was cleared while waiting
                            break # Exit the main processing loop

                        # TODO: 按照目前的逻辑，这里的共享缓存似乎可以被一个queue代替。
                        # 但也许实际上不能，使用一片内存可控性更好。
                        # Shared memory is filled, read data
                        shm_image_array = np.ndarray(
                            self.frame_shape,
                            dtype=self.frame_dtype, # Use frame_dtype
                            buffer=self.shm.buf
                        )
                        frame_to_analyze = shm_image_array.copy() # 拷贝是重要的，避免在分析时共享内存被覆盖

                        # Read timestamp
                        shm_timestamp_array = np.ndarray(
                            (1,),
                            dtype=self._submission_timestamp_dtype,
                            buffer=self.shm.buf,
                            offset=self._frame_data_bytes
                        )
                        submission_timestamp_us = shm_timestamp_array[0] # Will return a copy of _submission_timestamp_dtype
                        print(f"NNAnalyzer worker ({mp.current_process().pid}): Got data with submission timestamp (us): {submission_timestamp_us}")

                        # Reset _shm_is_filled to False, indicating shared memory is now empty
                        self._shm_is_filled.value = False

                        # Notify submitter that shared memory is empty
                        self.shm_cond.notify()

                    # Analyze the frame outside the lock
                    if frame_to_analyze is not None:
                        # Record analysis start time (optional, for more granular timing)
                        # analysis_start_time_us = time.perf_counter_ns() // 1000

                        results = self._analyze(frame_to_analyze) # Call subclass analyze

                        # Record analysis end time (result production time)
                        result_production_time_us = time.perf_counter_ns() // 1000

                        # Calculate total duration from submission to result production
                        total_duration_us = result_production_time_us - submission_timestamp_us

                        # Check if analyze returned None
                        if results is not None:
                            # Put result, submission timestamp, and duration into the result queue
                            self.result_queue.put((results, submission_timestamp_us, total_duration_us))
                        else:
                            print(f"NNAnalyzer worker ({mp.current_process().pid}): Analysis returned None, skipping result queue.")

                except Exception as e:
                    # Catch errors during single frame processing, print and continue loop
                    print(f"Error during frame processing in analyzer worker ({mp.current_process().pid}): {e}")
                    # Add more complex error handling if needed (e.g., retry, skip)
        finally:
        # 3. Cleanup
            print(f"NNAnalyzer worker process ({mp.current_process().pid}) uninitializing...")
            try:
                self._uninitialize_analyzer()
            except Exception as e:
                print(f"Error uninitializing NNAnalyzer worker process ({mp.current_process().pid}): {e}")
            print(f"NNAnalyzer worker process ({mp.current_process().pid}) uninitialized successfully.")

        print(f"NNAnalyzer worker ({mp.current_process().pid}) exiting loop.")


    def start(self):
        """
        Starts the NNAnalyzer.

        This method creates the necessary Inter-Process Communication (IPC)
        resources (SharedMemory, Condition) and launches the background worker
        process responsible for model initialization and frame analysis.
        Does nothing if the analyzer is already running.
        """
        # Start worker process
        if self.working.is_set():
            print("Analyzer already running.")
            return

        print("Starting NNAnalyzer...")
        try:
            # 5. 创建共享内存和条件变量
            # Use the calculated total frame bytes including appended timestamp
            shared_mem_size = self._total_frame_bytes # A single frame buffer
            print(f"Creating SharedMemory (size: {shared_mem_size} bytes)...")
            self.shm = SharedMemory(create=True, size=shared_mem_size)
            print("Creating Condition...")
            self.shm_cond = mp.Condition()
            print("Creating _shm_is_filled flag...")
            self._shm_is_filled = mp.Value('b', False)

            # 6. 设置运行状态并启动工作进程
            self.working.set()
            print("Starting worker process...")
            process = mp.Process(target=self._worker, name="NNAnalyzerWorker")
            self.worker_processes.append(process)
            process.start()
            # 等待 worker 进入等待状态
            self.worker_ready.wait()
            print("NNAnalyzer started successfully.")

        except Exception as e:
            print(f"Error starting NNAnalyzer: {e}")
            # 如果启动失败，清理可能已创建的资源
            if self.shm:
                self.shm.close()
                self.shm.unlink()
                self.shm = None
            self.working.clear() # 确保状态正确
            # self.initialized.clear() # 移除初始化标志的清除
            self.worker_ready.clear() # 清除 worker_waiting 标志


    def stop(self):
        """
        Stops the NNAnalyzer.

        Signals the worker process to terminate, waits for it to join (with a timeout),
        and cleans up the created IPC resources (SharedMemory).
        Does nothing if the analyzer is already stopped.
        """
        if not self.working.is_set() and not self.worker_processes:
            print("Analyzer already stopped.")
            return

        print("Stopping NNAnalyzer...")
        # Exiting safely
        self.working.clear()

        # Notify worker processes to stop if they are waiting on the condition
        if self.shm_cond:
             try:
                 with self.shm_cond:
                     self.shm_cond.notify_all() # 确保等待的 worker 能退出 wait
             except Exception as e:
                 print(f"Error notifying condition during stop: {e}") # Condition might already be closed

        # Wait for worker processes to finish
        print("Waiting for NNAnalyzer worker to join...")
        for process in self.worker_processes:
            process.join(timeout=5.0) # Add timeout to join
            if process.is_alive():
                print(f"Warning: Worker process {process.pid} did not exit gracefully. Terminating.")
                process.terminate() # Force terminate if join times out
        print("Worker process joined.")
        self.worker_processes = [] # 清空列表

        # 7. 清理共享内存资源
        if self.shm:
            print("Cleaning up SharedMemory...")
            try:
                self.shm.close()
                self.shm.unlink() # Important to prevent memory leaks
                print("SharedMemory cleaned up.")
            except FileNotFoundError:
                 print("SharedMemory already unlinked.")
            except Exception as e:
                print(f"Error cleaning up SharedMemory: {e}")
            finally:
                 self.shm = None # Reset shm reference

        # Reset condition variable reference
        self.shm_cond = None

        print("NNAnalyzer stopped.")


class MySpecificNNAnalyzer(NNAnalyzer):
    """
    A concrete implementation of NNAnalyzer using a placeholder analysis.
    """
    # 8. 更新子类 __init__ 的 super() 调用
    def __init__(self, model_path: str, frame_size: tuple[int, int, int]):
        super().__init__(model_path, frame_size)
        # __init__ 不再负责加载模型或打印信息

    # 实现抽象方法 _initialize_worker
    def _initialize_analyzer(self):
        """
        Load the specific model and perform warmup for MySpecificNNAnalyzer.
        """
        print(f"MySpecificNNAnalyzer ({mp.current_process().pid}) loading model from: {self.model_path}")
        # 实际的模型加载逻辑:
        # try:
        #     self.model = load_some_model_library(self.model_path)
        #     print("Model loaded successfully.")
        #     # 执行预热
        #     print("Performing warmup...")
        #     # warmup_data = np.zeros((1, *self.frame_size), dtype=np.uint8) # Example warmup data
        #     # self.model.predict(warmup_data)
        #     print("Warmup complete.")
        # except Exception as e:
        #     print(f"Error loading model or warming up: {e}")
        #     # 可以选择让进程退出或设置一个错误状态
        #     self.model = None # 确保 model 状态明确
        #     # raise # 重新抛出异常可能导致进程崩溃

        # 使用占位符代替实际加载
        self.model = "LoadedModelPlaceholder"
        print(f"MySpecificNNAnalyzer ({mp.current_process().pid}) model placeholder set.")
        print(f"MySpecificNNAnalyzer ({mp.current_process().pid}) worker initialization complete.")


    # 实现抽象方法 analyze
    def _analyze(self, frame: np.ndarray) -> Any:
        """
        Placeholder analysis: returns the frame shape.
        Replace this with actual model inference.
        """
        if self.model is None:
            print(f"Error: Model not loaded in worker process ({mp.current_process().pid}). Cannot analyze.")
            return None # 或者抛出异常

        # print(f"Analyzing frame with shape: {frame.shape} using model: {self.model}") # 调试信息
        # 实际应用中，这里会调用模型进行推理
        # result = self.model.predict(frame)
        # return result
        return frame.shape # 保持原有逻辑作为示例

    # 移除被注释掉的 warmup 方法块以修复 Pylance 错误
        # result = self.model.predict(frame)
        # return result
        return frame.shape # 保持原有逻辑作为示例

    # 移除被注释掉的 warmup 方法块以修复 Pylance 错误

    def _uninitialize_analyzer(self):
        """
        Clean up resources for MySpecificNNAnalyzer.
        """
        print(f"MySpecificNNAnalyzer ({mp.current_process().pid}) uninitializing...")
