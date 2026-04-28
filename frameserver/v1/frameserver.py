# frameserver.py
# Author: Haiyun Huang & Google Gemini 3.1 Pro

# V2 TODO: Process safe, do flow control for multiple consumer.
#          It receives `get` request from multiple consumers, get data ticket from the ring buffer,
#          cache the tickets for flow control, and issue its own FrameTicket (for release)
#          (or direct return data view) to consumers. It is FrameServer's responsibility to make sure every call to
#          the ring buffer is valid.
#          Background GC: detect the oldest occupied frame, release released frames 
#          by calling release() in ring buffer.
#          For the basic function, we need to maintain a structure that record the occupied frames by
#          each registered `named` consumer.
#          API: Sync named consumers will get continuous (per consumer) data:
#               register_consumer(name); get_sync(name, size)->Tuple[List[NDArray], FrameTicket]; release_sync(name, FrameTicket);
#               Async anonymous consumer do dirty read:
#               get_async_view(offset, size) -> Optional[List[NDArray]] # None if including trash data. similar to peek_frames() in RB v3a.
#               get_async_copy(offset, size) -> Optional[List[NDArray]]

import threading as t
import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Tuple, Set, Dict
from ringbuffers.shared_ring_buffer_v2a import ProcessSafeSharedRingBuffer

from loguru._logger import Logger # for type hint/checking only

class FrameServer:
    """
    A frame server that retrieves data from a shared ring buffer and Sync it 
    to multiple consumers.

    Consumers have two main types: 
        - Strict ("named") consumer: Should not drop any frame and desire a 
            integral memory view in `get` operation until explicitly release. 
            In this case, use `register_consumer()`, `get_sync()` and 
            `release_sync()`.
        - Anonymous consumer: Tolerant to data drop, may also do dirty read 
            for performance. In this case, use `get_async_` API series.

    All consumers will retrieve the same batch of data for the same time.
    
    This frame server is thread-safe for all public API. 
    In current structure, inject this frameserver to downstream consumer to get 
    data. We wish the consumers to **submit** the data in the same process
    as soon as possible. The submission can be (for example): 
        - Upload to GPU, 
        - Pipe to other process,
        - Do python calculation that makes a copy of data.
    To reduce data copy, it's important to do tradeoff between data integrity
    and performance.
    """
    def __init__(self, ring_buffer: ProcessSafeSharedRingBuffer, 
                 batch_size: int = 1, inject_logger: Optional[Logger] = None):
        """
        Initialize FrameServer.

        Args:
            ring_buffer (ProcessSafeSharedRingBuffer): The shared ring  buffer 
                instance to retrieve data from.
            batch_size (int): The number of frames to retrieve at once in following `get` call.
                For sync multiple consumer, this cannot be changed after server started.
                For the last batch, available frames may be less than specified `batch_size`.
            inject_logger (Logger): The loguru logger instance to use. If not specified,
                Logging will be disabled.
        Raises:
            TypeError: If ring_buffer is not an instance of `ProcessSafeSharedRingBuffer`.
            ValueError: If batch_size is not a positive integer.
        """
        if not isinstance(ring_buffer, ProcessSafeSharedRingBuffer):
            raise TypeError("ring_buffer must be an instance of ProcessSafeSharedRingBuffer")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if inject_logger is not None and not isinstance(inject_logger, Logger):
            raise TypeError("inject_logger must be an loguru Logger")

        self.buffer: ProcessSafeSharedRingBuffer = ring_buffer
        self._batch_size: int = batch_size
        if inject_logger is not None:
            inject_logger = inject_logger.bind(friendly_name="FrameServer")
        self._logger: Logger = inject_logger

        self._mutex: t.Lock = t.Lock()
        
        # Two core condition for producer-consumer sync.
        self._data_available: t.Condition = t.Condition(self._mutex)
        self._data_released: t.Condition = t.Condition(self._mutex)
        
        # === Internal cache and state ===
        self._current_batch: Optional[List[NDArray]] = None
        self._current_batch_count: int = 0 # Will not cleared up on stop(). 

        # === Broker thread resource (protected by _mutex) ===
        # Consider critical zone: enable_broker != broker_running: Init & Uninit Stage.
        self._enable_broker: bool = False # Request to enable broker loop.
        self._broker_running: bool = False # Broker running state.
        self._broker_thread: Optional[t.Thread] = None
        
        # === Consumer registration & data lock ===
        self._strict_consumers: Set[str] = set()
        self._pending_releases: Set[str] = set()

    def register_consumer(self, name: str):
        """
        Register a strict ("named") consumer with name as identifier, broker will wait 
        for its explicit release.

        Args:
            name (str): The identifier for the strict consumer, should not 
                duplicate with other strict consumers.
        """
        with self._mutex:
            self._strict_consumers.add(name)

    def start(self) -> bool:
        """
        Start background broker loop to retrive data from shared ring buffer.

        Blocking method, return until the background broker is ready or 
        immediately stopped.

        Returns: 
            bool: True if the background broker is ready at the time of return,
                False if immediately stopped.
        """
        with self._mutex:
            if self._enable_broker:
                self._data_available.wait_for(lambda: self._broker_running) # Will return immediatedly if running.
                if self._logger: self._logger.info("FrameServer has started.") # May started from elsewhere.
                return self._broker_running
            self._enable_broker = True 
            self._broker_thread = t.Thread(target=self._broker_loop, daemon=True, name="FrameServerBrokerLoop")
            self._broker_thread.start()
            # Exit start() until broker is ready.
            self._data_available.wait_for(lambda: self._broker_running or not self._enable_broker)
        if self._logger: self._logger.success("FrameServer started.")
        return self._broker_running

    def stop(self):
        """
        Stop the frame server.

        Blocking method, return until the remaining data in ring buffer is cleared
        by consumers and the background broker is fully stopped.
        """
        with self._mutex:
            if not self._enable_broker:
                if self._broker_thread and self._broker_thread.is_alive():
                    self._broker_thread.join()
                return
            self._enable_broker = False # request end broker loop.
            self._data_released.notify_all()

        pending_consumers: int = 0
        while self._broker_thread and self._broker_thread.is_alive():
            self._broker_thread.join(timeout=0.05)
            if self._logger and len(self._pending_releases) != 0 and \
            pending_consumers != len(self._pending_releases): 
                pending_consumers = len(self._pending_releases)
                self._logger.info(
                    f"FrameServer Broker is waiting for {pending_consumers} "
                    f"consumers {self._pending_releases} to release...")
        if self._logger: self._logger.success("FrameServer stopped successfully.")
        

    def _broker_loop(self):
        """
        Background broker loop that retrive data from shared ring buffer and 
        update internal buffer. Should be the only thread that interact with 
        shared ring buffer.
        """
        with self._mutex:
            if self._enable_broker: 
                # If _mutex rotation start -> stop -> broker_loop,
                # In this case, the condition is necessary.
                self._broker_running = True 
            self._data_available.notify_all() # Wake up start()
        
        try:
            batch_size = self._batch_size
            while self._broker_running:
                # Retrived a batch data from SharedRingBuffer
                batch: List[NDArray] = self.buffer.get(get_frame_num=batch_size, timeout=0.1)
                
                if batch is None:
                    # RingBuffer underflow (less than 1 batch_size), continue waiting.
                    if self._enable_broker: continue
                    else: # stop signalled
                        # Clear all remaining data if exists.
                        if self.buffer.unread_count > 0:
                            batch: List[NDArray] = self.buffer.get(
                                get_frame_num=self.buffer.unread_count, timeout=0.1)
                            if batch is None:
                                # Should not happen.
                                break
                        else:
                            # Stop condition: remaining data cleared.
                            break

                # Load data and notify consumers.
                with self._mutex:
                    self._current_batch = batch
                    self._current_batch_count += 1
                    # Reset pending release set to all strict consumers
                    self._pending_releases = self._strict_consumers.copy()
                    self._data_available.notify_all()
                
                # Pending all strict consumers to release
                with self._mutex:
                    self._data_released.wait_for(
                        lambda: len(self._pending_releases) == 0 or not self._broker_running
                    )
                # Loop ends, next self.buffer.get() will release the ringbuffer data 
                # per ringbuffer's design.
        finally:
             with self._mutex:
                self._broker_running = False
                self._current_batch = None
                self._data_available.notify_all() # Unblock all waiting consumers.
    
    @property
    def broker_running(self):
        """
        Public method to query the broker state. Check this frequently to achieve 
        thread-safety.

        If broker is not running, all `get` method will return `None` data.
        """
        # Read only, no lock is ok and preferred?
        # Not related with the thread instance is ok? since in current logic the 
        # _broker_running property flanked the broker loop.
        return self._broker_running


    # ====== Consumer APIs ======

    def get_sync(self, last_batch_count: int) -> Tuple[Optional[List[NDArray]], int]:
        """
        Get the "next" batch of data, synced by given last batch count ("ID"). For the
        last batch, frames returned may be less than specified `batch_size`.

        This method is blocking and will wait until the next batch of data is available.
        Should be paired with `release_sync()` to release the data by registered consumer
        explicitly. 
        Anonymous consumer should use `get_async_` function series instead.

        Args:
            last_batch_count (int): The last batch count returned from this method. 
                For the first call, pass in 0.

        Returns:
            Tuple[Optional[List[NDArray]], int]: A tuple of (batch_data, current_batch_count).
                batch_data (Optional[List[NDArray]]): Frame data get from SharedRingBuffer.
                    None if Broker is not running.
                current_batch_count (int): The current batch count ("ID") for 
                    the returned batch_data. Use this for the next FrameServer API call.
                    (release_sync(), get_sync())
        """
        with self._mutex:
            self._data_available.wait_for(
                lambda: self._current_batch_count > last_batch_count or not self._broker_running
            )
            if not self._broker_running:
                # current_batch will not be None, this is guaranteed by broker loop.
                # wait_for() is released by broker signal _running, all data cleared.
                return None, self._current_batch_count
            
            return self._current_batch, self._current_batch_count

    def release_sync(self, name: str, data_batch_count: int):
        """
        Release the data of given "ID" for specific registered consumer.

        The data integrity will be guaranteed until all strict consumers 
        release the data.
        The frameserver will wait until all registered consumers release the data
        before `stop()` is returned.

        Args:
            name (str): The identifier for the strict consumer.
            data_batch_count (int): The batch count ("ID") of the data to be released.
        """
        with self._mutex:
            # Only release if the input data_batch_count is current batch count
            # to avoid mis-release by expired call (unexpected retry?). 
            if data_batch_count == self._current_batch_count:
                self._pending_releases.discard(name)
                # If all strict consumers release the data, notify broker to get next batch
                if len(self._pending_releases) == 0:
                    self._data_released.notify()
            else:
                # Optionally log: a thread released expired data, which is ignored
                pass


    def get_async_copy(self) -> Tuple[Optional[List[np.ndarray]], int]:
        """
        Get a copy of current data batch. Can be called by any anonymous consumer.
        Current data batch could have less frames than specified `batch_size` 
        when closing.

        To reduce data copy, use `get_async_view()` to get a dirty view of data.

        Returns:
            Tuple[Optional[List[NDArray]], int]: A tuple of (batch_data, current_batch_count).
                batch_data (Optional[List[NDArray]]): Frame data get from SharedRingBuffer.
                    None if Broker is not running.
        """
        with self._mutex:
            if not self._broker_running:
                return None, self._current_batch_count
            
            # Full copy of batch data.
            safe_copy =[np.copy(arr) for arr in self._current_batch]
            return safe_copy, self._current_batch_count
    
    def get_async_view(self) -> Tuple[Optional[List[np.ndarray]], int]:
        """
        Get a dirty view of current data batch. Can be called by any anonymous consumer. 
        The data integrity is not guaranteed. Current data batch could have less frames 
        than specified `batch_size` when closing.

        If data integrity is necessary, subscribe your function with `register_consumer()`
        and use `get_sync()` to get data.

        Returns:
            Tuple[Optional[List[NDArray]], int]: A tuple of (batch_data, current_batch_count).
                batch_data (Optional[List[NDArray]]): Frame data get from SharedRingBuffer.
                    None if Broker is not running.
        """
        with self._mutex:
            if not self._broker_running:
                return None, self._current_batch_count
            
            return self._current_batch, self._current_batch_count

if __name__ == "__main__":
    import time
    import threading
    from loguru import logger as file_logger
    # 1. 创建共享内存
    rb = ProcessSafeSharedRingBuffer(
        create=True, buffer_capacity=60, frame_shape=(2048, 2048, 1), dtype=np.uint16
    )

    # 2. 启动 FrameServer
    server = FrameServer(rb, batch_size=3, inject_logger=file_logger)
    server.register_consumer("Encoder")
    server.start()

    # 3. 启动一个生产者线程（模拟相机）
    def producer():
        frame = np.ones((1, 2048, 2048, 1), dtype=np.uint16) * 100
        for i in range(100):
            rb.put(frame)
            time.sleep(0.01) # 模拟相机帧率
        print("Producer finished.")

    t_prod = threading.Thread(target=producer)
    t_prod.start()

    # 4. 启动一个严格消费者（模拟 Encoder）
    def strict_consumer():
        last_id = 0
        while True:
            data, current_id = server.get_sync(last_id)
            if data is None:
                break
            # 模拟处理时间（比如编码）
            len_data = np.sum([arr.shape[0] for arr in data])
            print(f"strict: {current_id}, len: {len_data}")
            time.sleep(0.2)
            server.release_sync("Encoder", current_id)
            last_id = current_id
        print("Strict Consumer finished.")

    t_strict = threading.Thread(target=strict_consumer)
    t_strict.start()

    # 5. 启动一个异步消费者（模拟 AI 推理，可以容忍延迟）
    def async_consumer():
        last_id = 0
        start_time = time.time()
        while time.time()-start_time < 5:
            data, current_id = server.get_async_view()
            if current_id > last_id and data is not None:
                # 模拟 AI 耗时（比编码慢）
                len_data = np.sum([arr.shape[0] for arr in data])
                print(f"async: {current_id}, len: {len_data}")
                time.sleep(0.05)
                last_id = current_id
            else:
                time.sleep(0.001)
        print("Async Consumer finished.")

    t_async = threading.Thread(target=async_consumer)
    t_async.start()

    # 6. 等待所有线程结束
    t_prod.join()
    
    # 停止 FrameServer（这会触发优雅排空）
    server.stop()
    
    t_strict.join()
    t_async.join()

    print("All threads finished.")
    rb.close()
    rb.unlink()