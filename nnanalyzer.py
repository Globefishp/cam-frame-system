import abc # 引入 abc 模块
import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Any # 引入类型提示

import numpy as np

class NNAnalyzer(abc.ABC):
    '''
    Abstract base class for a neural network analyzer using multiprocessing.
    Handles IPC resource creation, process management, and communication.
    Requires subclasses to implement the analysis logic.
    '''
    # 1. 修改 __init__ 参数，移除 IPC 资源
    def __init__(self,
                 model_path: str,
                 frame_size: tuple[int, int, int]):
        '''
        Initialize the base analyzer. IPC resources will be created in start().
        Args:
            model_path: (str), path to the model file (used by subclasses).
            frame_size: (tuple[int, int, int]), expected frame size (height, width, channel).
        '''
        self.model_path = model_path
        self.frame_size = frame_size
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
        # Model placeholder - will be loaded by subclass in worker process via _initialize_worker
        self.model = None

    # 移除基类的 warmup 方法，初始化由子类的 _initialize_worker 负责
    # def warmup(self):
    #     """Optional warmup routine for subclasses."""
    #     pass

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


    def submit_frame(self, frame: np.ndarray):
        """
        Submits a frame for analysis by copying it to shared memory
        and notifying the worker process.
        Args:
            frame: (np.ndarray), a frame to submit.
        """
        # 4. 添加检查确保 IPC 资源已初始化
        if not self.working.is_set() or self.shm is None or self.shm_cond is None:
            print("Warning: NNAnalyzer is not running or IPC not initialized, cannot submit frame.")
            print("Please call start() before submit_frame().")
            return
        # 检查 worker 是否正在等待帧
        if not self.worker_ready.is_set():
            print("Warning: NNAnalyzer worker is not waiting for frames yet, Waiting...")
            self.worker_ready.wait()
            return
        try:
            # 使用 self.shm_cond
            with self.shm_cond: # 加锁
                # 将分析帧写入共享内存 (使用 self.shm)
                dest = np.ndarray(self.frame_size,
                                  dtype=np.uint8,
                                  buffer=self.shm.buf)
                # 确保帧尺寸匹配
                if frame.shape != self.frame_size or frame.dtype != np.uint8:
                     print(f"Error in submitting a frame to NNAnalyzer: Frame shape/dtype mismatch. Expected {self.frame_size} {np.uint8}, got {frame.shape} {frame.dtype}")
                     # 可以选择抛出异常或返回错误
                     return # 或者 raise ValueError("Frame shape/dtype mismatch")

                np.copyto(dest, frame)
                # 通知分析进程
                self.shm_cond.notify() # 通常通知任何一个等待者即可
        except Exception as e:
            print(f"Error submitting frame: {e}")


    def get_result(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Retrieves the next analysis result from the result queue.
        Args:
            timeout: (Optional[float]), seconds to wait for a result.
                     If None, waits indefinitely.
                     If 0, returns immediately (may raise Empty exception).
        Returns:
            The analysis result, or potentially raises queue.Empty.
        """
        return self.result_queue.get(timeout=timeout)


    def _worker(self):
        """
        The main function executed by this background worker process.

        This method first calls the subclass's `_initialize_worker` method
        to load the model and perform any necessary setup. Then, it enters
        a loop where it waits for a notification on the shared memory condition
        variable (`shm_cond`), reads the frame data from shared memory (`shm`),
        calls the subclass's `analyze` method to process the frame, and puts
        the result into the shared result queue (`result_queue`).

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
        try:
            while self.working.is_set():
                try: # 将 try...except 移到循环内部，处理单次迭代的错误
                    frame_to_analyze = None
                    with self.shm_cond:  # Acquire condition lock
                        self.worker_ready.set() # 此时，worker才真正准备好接收帧。
                        # 等待 submit_frame 的通知
                        notified = self.shm_cond.wait(timeout=1.0) # 添加超时避免永久阻塞

                        if not notified or not self.working.is_set():
                            continue # 超时或收到停止信号

                        # Read frame from shared memory only after notification
                        frame_to_analyze = np.ndarray(
                            self.frame_size,
                            dtype=np.uint8,
                            buffer=self.shm.buf
                        ).copy() # 拷贝是重要的，避免在分析时共享内存被覆盖

                    if frame_to_analyze is not None:
                        # received frame
                        results = self._analyze(frame_to_analyze) # 调用子类实现的 analyze
                        # 检查 analyze 是否返回 None (例如模型未加载)
                        if results is not None:
                            self.result_queue.put(results)
                        else:
                            print(f"NNAnalyzer worker ({mp.current_process().pid}): Analysis returned None, skipping result queue.")

                except Exception as e:
                    # 捕获处理单帧时的错误，打印并继续循环
                    print(f"Error during frame processing in analyzer worker ({mp.current_process().pid}): {e}")
                    # 根据需要，可以添加更复杂的错误处理逻辑，比如重试或跳过
        finally:
        # 3. 清理
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
            frame_bytes = np.prod(self.frame_size) * np.dtype(np.uint8).itemsize
            # 考虑为共享内存增加一些buffer，例如存储少量帧
            buffer_factor = 10 # 至少能存一帧，这里设为10帧的空间
            shared_mem_size = int(frame_bytes * buffer_factor)
            print(f"Creating SharedMemory (size: {shared_mem_size} bytes)...")
            self.shm = SharedMemory(create=True, size=shared_mem_size)
            print("Creating Condition...")
            self.shm_cond = mp.Condition()

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
