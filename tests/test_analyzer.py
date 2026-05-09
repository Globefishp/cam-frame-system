import pytest
import time
import numpy as np
import multiprocessing as mp
from typing import Any
import sys
from loguru import logger

from analyzers.analyzer import BaseAnalyzer
from analyzers.analyzer_types import TensorType, DeviceType, ConsumerMode
from frameserver.v3.frameserver_v3 import FrameServer
from ringbuffers.shared_ring_buffer_v4 import ProcessSafeSharedRingBuffer


class MockAnalyzer(BaseAnalyzer):
    def __init__(self, frame_server: FrameServer, control_flags=None, command_history=None, **kwargs):
        # Fallback to standard dict/list if not provided (safe for tests that don't need IPC observation)
        self.control_flags = control_flags if control_flags is not None else {}
        self.command_history = command_history if command_history is not None else []
        
        kwargs.setdefault('inject_logger', logger)
        super().__init__(frame_server, **kwargs)

    def _initialize_analyzer(self):
        if self.control_flags.get('init_fail', False):
            raise RuntimeError("Intentional Init Failure")

    def _uninitialize_analyzer(self):
        pass

    def _analyze(self, frame, **kwargs):
        sleep_time = self.control_flags.get('sleep_time', 0.0)
        if sleep_time > 0:
            time.sleep(sleep_time)
            
        if self.control_flags.get('raise_exception', False):
            raise ValueError("Intentional Analyze Exception")

        # Detect tensor type
        is_torch = False
        is_cuda = False
        if not isinstance(frame, np.ndarray):
            # Assume torch
            import torch
            if isinstance(frame, torch.Tensor):
                is_torch = True
                is_cuda = frame.device.type == 'cuda'
                frame = frame.cpu().numpy()

        result_data = {
            "sum": float(frame.sum()),
            "shape": frame.shape,
            "is_torch": is_torch,
            "is_cuda": is_cuda,
            "kwargs": kwargs
        }

        # Status testing (only driven by specific kwargs during step)
        if "test_status_updates" in kwargs:
            self._status_update({"last_analyzed_sum": result_data["sum"]})
            self._status_subdict_update("meta", {"shape": result_data["shape"]})
            self._status_update({"to_be_deleted": 1})
            self._status_del(["to_be_deleted"])
            
        if "test_overwrite" in kwargs:
            self._status_overwrite({"overwritten": True})

        # Update results
        res_key = kwargs.get('frame_id', time.time_ns())
        self._result_update({res_key: result_data})

    def _handle_command(self, cmd_name: str, payload: Any):
        # We can append to the Manager list safely across processes
        self.command_history.append((cmd_name, payload, time.time_ns()))


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    # 1. 移除默认的控制台处理器（它默认没开启 enqueue）
    logger.remove()
    
    # 2. 重新添加处理器，并开启多进程支持
    logger.add(
        sys.stderr, 
        enqueue=True,  # 核心：确保跨进程日志安全
        colorize=True, 
        format="<green>{time}</green> | <level>{level: <8}</level> | <cyan>{process.id}</cyan>:<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    yield

@pytest.fixture
def ring_buffer():
    buffer = ProcessSafeSharedRingBuffer(
        buffer_capacity=10,
        frame_shape=(10, 10, 3),
        dtype=np.uint8,
    )
    yield buffer
    buffer.close()
    buffer.unlink()

@pytest.fixture
def frame_server(ring_buffer):
    fs = FrameServer(True, ring_buffer=ring_buffer)
    yield fs
    fs.close()
    fs.unlink()

@pytest.fixture
def mock_controls():
    """Provides mp.Manager dict and list for true inter-process control & observation."""
    with mp.Manager() as manager:
        control_flags = manager.dict()
        control_flags['sleep_time'] = 0.0
        control_flags['raise_exception'] = False
        control_flags['init_fail'] = False
        
        command_history = manager.list()
        yield control_flags, command_history


def push_frame(fs: FrameServer, val=1):
    frame = np.full((1, 10, 10, 3), val, dtype=np.uint8)
    fs.buffer.put(frame)


def test_initialization_and_pickle(frame_server):
    analyzer = MockAnalyzer(frame_server, tensor_type=TensorType.NUMPY)
    state = analyzer.__getstate__()
    # Verify unpicklable IPC properties are removed
    assert '_cmd_tx' not in state
    assert '_status_rx' not in state
    assert '_result_rx' not in state
    assert '_status_thread' not in state
    assert '_result_thread' not in state


def test_start_stop_cleanly(frame_server):
    analyzer = MockAnalyzer(frame_server)
    assert not analyzer.is_ready
    analyzer.start()
    assert analyzer.is_ready
    time.sleep(0.1) # This signal is slower
    assert analyzer.status.get('analyzer_status') in ['ready', 'busy']
    
    analyzer.stop()
    assert analyzer._worker_process is None
    assert analyzer.status.get('analyzer_status') == 'idle'


def test_double_start_stop(frame_server):
    analyzer = MockAnalyzer(frame_server)
    analyzer.start()
    analyzer.start() # Should not crash
    assert analyzer.is_ready
    
    analyzer.stop()
    analyzer.stop() # Should be safe
    assert analyzer._worker_process is None


def test_multiple_restarts(frame_server):
    analyzer = MockAnalyzer(frame_server, continuous_mode=False)
    for _ in range(2):
        analyzer.start()
        assert analyzer.is_ready

        push_frame(frame_server)
        step_id = analyzer.step()
        
        # poll error using non-blocking API.
        err = None
        for _ in range(20): # Max retry 20.
            err = analyzer.get_error(step_id)
            if err is not None: break
            time.sleep(0.1)
        assert err == 'success'
        
        analyzer.stop()
        assert not analyzer.is_ready


def test_continuous_mode_execution(frame_server):
    analyzer = MockAnalyzer(frame_server, continuous_mode=True, stat_interval=0.1)
    analyzer.start()
    
    # Push multiple frames to test continuous processing
    for i in range(5):
        push_frame(frame_server, val=i)
        time.sleep(0.05)
        
    time.sleep(0.5) # Wait for processing
    
    status = analyzer.status
    assert status.get('analyzer_status') == 'busy'
    assert 'fps' in status
    assert status.get('frame_count', 0) >= 0
    
    results = analyzer.get_results()
    assert len(results) > 0
    
    analyzer.stop()


def test_step_mode_execution(frame_server):
    analyzer = MockAnalyzer(frame_server, continuous_mode=False)
    analyzer.start()
    
    push_frame(frame_server, val=7)
    
    step_id = analyzer.step(frame_id=123, param="test")
    assert step_id != -1
    
    # Now we test the timeout of get_error
    success = False
    err = analyzer.get_error(step_id, timeout=1.0)
    if err == 'success':
        success = True
    assert success
    
    # Test the result is ready immediately after get_error succeeded
    res = analyzer.get_result(123, timeout=0)
    assert res is not None
    assert res['kwargs']['param'] == "test"
    assert res['sum'] == 10*10*3*7
    
    analyzer.stop()


def test_step_timeout_no_data(frame_server):
    # Shorten timeout to speed up test
    analyzer = MockAnalyzer(frame_server, continuous_mode=False, fetch_timeout=0.2)
    analyzer.start()
    
    # FrameServer is empty, step should timeout
    step_id = analyzer.step()
    
    err = None
    for _ in range(20):
        err = analyzer.get_error(step_id)
        if err is not None:
            break
        time.sleep(0.1)

    assert err is not None
    assert "Timeout" in err
    assert analyzer.status.get('analyzer_status') == 'ready'
    
    analyzer.stop()


def test_analyze_exception_propagation(frame_server, mock_controls):
    control_flags, command_history = mock_controls
    control_flags['raise_exception'] = True
    
    analyzer = MockAnalyzer(frame_server, control_flags, command_history, continuous_mode=False)
    analyzer.start()
    
    push_frame(frame_server)
    step_id = analyzer.step()
    
    err = None
    for _ in range(20):
        err = analyzer.get_error(step_id)
        if err is not None:
            break
        time.sleep(0.1)

    assert err is not None
    assert "error: Intentional Analyze Exception" in err
    
    analyzer.stop()


def test_fast_consecutive_steps_dropped(frame_server, mock_controls):
    control_flags, command_history = mock_controls
    control_flags['sleep_time'] = 0.5 # Block _analyze for 0.5s
    
    analyzer = MockAnalyzer(frame_server, control_flags, command_history, continuous_mode=False)
    analyzer.start()
    
    push_frame(frame_server)
    push_frame(frame_server)
    
    step1 = analyzer.step()
    time.sleep(0.1) # ensure worker picks it up
    step2 = analyzer.step() # should be dropped since worker is busy
    
    # Check step2 is dropped immediately.
    start_time = time.time()
    err2 = analyzer.get_error(step2, timeout=1.0)
    
    assert time.time() - start_time < 0.5
    assert err2 is not None
    assert "Dropped" in err2
    
    # Check step1 eventually succeeds
    err1 = None
    for _ in range(20):
        err1 = analyzer.get_error(step1)
        if err1 is not None:
            break
        time.sleep(0.1)
        
    assert err1 == 'success'
    
    analyzer.stop()


def test_status_dict_updates(frame_server):
    analyzer = MockAnalyzer(frame_server, continuous_mode=False)
    analyzer.start()
    push_frame(frame_server)
    
    step_id = analyzer.step(test_status_updates=True)
    
    while analyzer.get_error(step_id) is None:
        time.sleep(0.1)
        
    status = analyzer.status
    assert "to_be_deleted" not in status
    assert "last_analyzed_sum" in status
    assert "meta" in status
    assert "shape" in status["meta"]
    
    push_frame(frame_server)
    step_id = analyzer.step(test_overwrite=True)
    while analyzer.get_error(step_id) is None:
        time.sleep(0.1)
        
    status2 = analyzer.status
    assert "overwritten" in status2
    assert "last_analyzed_sum" not in status2 # It was cleared!
    
    analyzer.stop()


def test_concurrent_custom_command(frame_server, mock_controls):
    control_flags, command_history = mock_controls
    control_flags['sleep_time'] = 0.5
    
    analyzer = MockAnalyzer(frame_server, control_flags, command_history, continuous_mode=False)
    analyzer.start()
    
    push_frame(frame_server)
    step_id = analyzer.step()
    time.sleep(0.1) # Wait a bit for worker to lock into _analyze
    
    # Send custom command while _analyze is blocking
    analyzer.send_command('custom_cmd', 'test_payload')
    
    # Wait for step to finish
    while analyzer.get_error(step_id) is None:
        time.sleep(0.1)
        
    history = list(command_history)
    assert len(history) == 1
    assert history[0][0] == 'custom_cmd'
    assert history[0][1] == 'test_payload'
    
    analyzer.stop()


def test_numpy_and_torch_tensor_conversion(frame_server):
    # Test Numpy
    analyzer_np = MockAnalyzer(frame_server, continuous_mode=False, tensor_type=TensorType.NUMPY)
    analyzer_np.start()
    push_frame(frame_server)
    step_id = analyzer_np.step()
    err = analyzer_np.get_error(step_id, timeout=None)
    assert err == 'success'
    
    assert not analyzer_np.get_latest_result()['is_torch']
    analyzer_np.stop()
    
    # Test Torch
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")
        
    analyzer_pt = MockAnalyzer(frame_server, continuous_mode=False, tensor_type=TensorType.TORCH)
    analyzer_pt.start()
    push_frame(frame_server)
    step_id = analyzer_pt.step()
    err = analyzer_pt.get_error(step_id, timeout=None)
    assert err == 'success'
    
    assert analyzer_pt.get_latest_result()['is_torch']
    analyzer_pt.stop()


def test_cuda_device_support(frame_server):
    try:
        import torch
        from utils.cuda_shm_pinner import CUDAPinner
        try:
            CUDAPinner() # Init test
        except RuntimeError:
            pytest.skip("CUDA Driver not initialized or GPU not available.")
            
        if not torch.cuda.is_available():
            raise RuntimeError("Torch says CUDA not available")
    except (ImportError, RuntimeError, Exception) as e:
        pytest.skip(f"CUDA or PyTorch not available for test: {e}")
        
    analyzer = MockAnalyzer(frame_server, continuous_mode=False, tensor_type=TensorType.TORCH, device=DeviceType.CUDA)
    analyzer.start()
    push_frame(frame_server)
    step_id = analyzer.step()
    while analyzer.get_error(step_id) is None: time.sleep(0.1)
        
    res = analyzer.get_latest_result()
    assert res['is_torch']
    assert res['is_cuda']
    analyzer.stop()


def test_sync_vs_async_consumer(frame_server):
    # Sync mode should request cid and register consumer
    analyzer_sync = MockAnalyzer(frame_server, continuous_mode=False, consumer_mode=ConsumerMode.SYNC)
    analyzer_sync.start()
    assert analyzer_sync._fs_cid is not None and analyzer_sync._fs_cid >= 0
    analyzer_sync.stop()
    
    # Async mode should not need cid
    analyzer_async = MockAnalyzer(frame_server, continuous_mode=False, consumer_mode=ConsumerMode.ASYNC)
    analyzer_async.start()
    assert analyzer_async._fs_cid == -1
    
    push_frame(frame_server)
    step_id = analyzer_async.step()
    while analyzer_async.get_error(step_id) is None: time.sleep(0.1)
    
    assert analyzer_async.get_latest_result() is not None
    analyzer_async.stop()


# === Batch size 测试 ===
class BatchTestAnalyzer(BaseAnalyzer):
    """
    专用于测试 Batch 功能的 Analyzer 子类。
    提取底层的内存连续性信息和特定像素值用于精确验证回绕拼接逻辑。
    """
    def _initialize_analyzer(self):
        pass

    def _uninitialize_analyzer(self):
        pass

    def _handle_command(self, cmd_name: str, payload: Any):
        pass

    def _analyze(self, frame, **kwargs):
        is_torch = not isinstance(frame, np.ndarray)
        
        if is_torch:
            import torch
            is_cuda = frame.device.type == 'cuda'
            is_contiguous = frame.is_contiguous()
            # 获取每个 Frame 的第一个像素点的值，用于验证时间顺序
            first_pixels = frame[:, 0, 0, 0].cpu().numpy().tolist()
        else:
            is_cuda = False
            is_contiguous = frame.flags['C_CONTIGUOUS']
            first_pixels = frame[:, 0, 0, 0].tolist()

        result_data = {
            "shape": frame.shape,
            "is_torch": is_torch,
            "is_cuda": is_cuda,
            "is_contiguous": is_contiguous,
            "first_pixels": first_pixels
        }
        
        # 使用当前纳秒时间戳作为 key 发送回主进程
        self._result_update({time.time_ns(): result_data})


def test_batch_completeness_basic(frame_server):
    """测试常规场景下，完整拉取指定 batch_size 的数据 (不发生回绕)"""
    analyzer = BatchTestAnalyzer(
        frame_server, 
        batch_size=3, 
        tensor_type=TensorType.NUMPY, 
        continuous_mode=False, 
        consumer_mode=ConsumerMode.ASYNC
    )
    analyzer.start()
    
    # 推入3帧测试数据，值分别为 10, 11, 12
    for i in range(3):
        push_frame(frame_server, val=i + 10)
        
    step_id = analyzer.step()
    err = analyzer.get_error(step_id, timeout=2.0)
    assert err == 'success'
    
    res = analyzer.get_latest_result()
    assert res['shape'] == (3, 10, 10, 3)
    assert res['first_pixels'] == [10, 11, 12]
    assert res['is_contiguous'] is True
    
    analyzer.stop()


def test_batch_wraparound_numpy(frame_server):
    """测试 RingBuffer 发生回绕时 Numpy 拼接的连续性和准确性"""
    push_frame(frame_server, val=0)
    push_frame(frame_server, val=1)
    frame_server._gc(2) # 丢弃两个历史数据, 移动初始读写指针到2

    analyzer = BatchTestAnalyzer(
        frame_server, 
        batch_size=4, 
        tensor_type=TensorType.NUMPY, 
        continuous_mode=False, 
        consumer_mode=ConsumerMode.ASYNC
    )
    analyzer.start()
    
    # buffer_capacity 为 10。推入 10 帧, 从2开始回绕
    # 写指针最终停在索引 2，此时 ASYNC 获取最新的 4 帧，跨越了物理边界 (索引 8, 9, 0, 1)
    for i in range(10):
        push_frame(frame_server, val=i+2)
    assert frame_server.buffer.read_ptr == 2
    assert frame_server.buffer.write_ptr == 2
        
    step_id = analyzer.step()
    err = analyzer.get_error(step_id, timeout=2.0)
    assert err == 'success'
    res = analyzer.get_latest_result()
    assert res['shape'] == (4, 10, 10, 3)
    assert res['first_pixels'] == [8, 9, 10, 11]  # 严格检验数据是否按时间顺序合并
    assert res['is_contiguous'] is True
    
    analyzer.stop()


def test_batch_wraparound_torch_cpu(frame_server):
    """测试 RingBuffer 发生回绕时 PyTorch (CPU) 的 cat() 拼接行为"""
    push_frame(frame_server, val=0)
    push_frame(frame_server, val=1)
    frame_server._gc(2)

    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")

    analyzer = BatchTestAnalyzer(
        frame_server, 
        batch_size=4, 
        tensor_type=TensorType.TORCH, 
        device=DeviceType.CPU,
        continuous_mode=False, 
        consumer_mode=ConsumerMode.ASYNC
    )
    analyzer.start()
    
    for i in range(10):
        push_frame(frame_server, val=i+2)
    assert frame_server.buffer.read_ptr == 2
    assert frame_server.buffer.write_ptr == 2
        
    step_id = analyzer.step()
    err = analyzer.get_error(step_id, timeout=2.0)
    assert err == 'success'
    
    res = analyzer.get_latest_result()
    assert res['shape'] == (4, 10, 10, 3)
    assert res['is_torch'] is True
    assert res['is_cuda'] is False
    assert res['first_pixels'] == [8, 9, 10, 11]
    assert res['is_contiguous'] is True
    
    analyzer.stop()


def test_batch_wraparound_torch_cuda(frame_server):
    """测试核心: GPU DMA 场景下，分两段上传并在 VRAM 内部连续组装的准确性"""
    push_frame(frame_server, val=0)
    push_frame(frame_server, val=1)
    frame_server._gc(2)
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("PyTorch not installed")

    analyzer = BatchTestAnalyzer(
        frame_server, 
        batch_size=4, 
        tensor_type=TensorType.TORCH, 
        device=DeviceType.CUDA,
        continuous_mode=False, 
        consumer_mode=ConsumerMode.ASYNC
    )
    analyzer.start()
    
    for i in range(10):
        push_frame(frame_server, val=i+2)
    assert frame_server.buffer.read_ptr == 2
    assert frame_server.buffer.write_ptr == 2
        
    step_id = analyzer.step()
    err = analyzer.get_error(step_id, timeout=2.0)
    assert err == 'success'
    
    res = analyzer.get_latest_result()
    assert res['shape'] == (4, 10, 10, 3)
    assert res['is_torch'] is True
    assert res['is_cuda'] is True
    # 如果两个非阻塞的 copy_ 发生错位或覆盖，这里的序列就会乱掉
    assert res['first_pixels'] == [8, 9, 10, 11] 
    assert res['is_contiguous'] is True
    
    analyzer.stop()

# TODO: 测试注入Extractor时的行为是否如文档所记载: 能够得到一个batch的数据, 并且仅处理原始数据.