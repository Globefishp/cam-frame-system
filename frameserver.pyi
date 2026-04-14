import numpy as np
from loguru._logger import Logger
from numpy.typing import NDArray as NDArray
from ringbuffers.shared_ring_buffer_v2a import ProcessSafeSharedRingBuffer

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
    buffer: ProcessSafeSharedRingBuffer
    def __init__(self, ring_buffer: ProcessSafeSharedRingBuffer, batch_size: int = 1, inject_logger: Logger | None = None) -> None:
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
    def register_consumer(self, name: str):
        '''
        Register a strict ("named") consumer with name as identifier, broker will wait 
        for its explicit release.

        Args:
            name (str): The identifier for the strict consumer, should not 
                duplicate with other strict consumers.
        '''
    def start(self) -> bool:
        """
        Start background broker loop to retrive data from shared ring buffer.

        Blocking method, return until the background broker is ready or 
        immediately stopped.

        Returns: 
            bool: True if the background broker is ready at the time of return,
                False if immediately stopped.
        """
    def stop(self) -> None:
        """
        Stop the frame server.

        Blocking method, return until the remaining data in ring buffer is cleared
        by consumers and the background broker is fully stopped.
        """
    @property
    def broker_running(self):
        """
        Public method to query the broker state. Check this frequently to achieve 
        thread-safety.

        If broker is not running, all `get` method will return `None` data.
        """
    def get_sync(self, last_batch_count: int) -> tuple[list[NDArray] | None, int]:
        '''
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
        '''
    def release_sync(self, name: str, data_batch_count: int):
        '''
        Release the data of given "ID" for specific registered consumer.

        The data integrity will be guaranteed until all strict consumers 
        release the data.
        The frameserver will wait until all registered consumers release the data
        before `stop()` is returned.

        Args:
            name (str): The identifier for the strict consumer.
            data_batch_count (int): The batch count ("ID") of the data to be released.
        '''
    def get_async_copy(self) -> tuple[list[np.ndarray] | None, int]:
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
    def get_async_view(self) -> tuple[list[np.ndarray] | None, int]:
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
