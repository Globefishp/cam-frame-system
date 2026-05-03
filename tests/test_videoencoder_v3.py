# Gemini 3.1 Pro
# test_videoencoder_v3.py
import pytest
import multiprocessing as mp
import numpy as np
import time
import sys
from loguru import logger

# 请根据你的实际路径修改导入
from encoders.videoencoder_v3 import BaseVideoEncoder
from encoders.videoencoder_types import EncoderException
from ringbuffers.shared_ring_buffer_v4 import ProcessSafeSharedRingBuffer
from frameserver.v3 import FrameServer

# ================= 辅助/Dummy 类定义 =================

class DummyValidEncoder(BaseVideoEncoder):
    """
    一个真实的子类，利用 mp.Manager().dict() 将子进程的状态同步回主进程进行断言。
    """
    def __init__(self, frame_server, output_path, batch_size, mp_record_dict, **kwargs):
        super().__init__(frame_server, output_path, batch_size, **kwargs)
        self.mp_record_dict = mp_record_dict

    def _initialize_encoder(self):
        self.mp_record_dict['init_calls'] += 1

    def _encode_frames(self, frames):
        # 提取帧序号 (假设写入时，把序号写在第一行第一列的通道0里)
        # frames 是 List[np.ndarray], 我们将其展开
        for arr in frames:
            for i in range(arr.shape[0]):
                frame_idx = int(arr[i, 0, 0, 0])
                # 注意：跨进程追加 manager.list 需要重新赋值或直接 append 到代理对象
                self.mp_record_dict['encoded_frames'].append(frame_idx)
        # 稍微睡眠模拟编码耗时
        time.sleep(0.05)

    def _uninitialize_encoder(self):
        self.mp_record_dict['uninit_calls'] += 1


class DummyErrorEncoder(DummyValidEncoder):
    """用于测试异常情况的 Encoder"""
    def __init__(self, error_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_type = error_type

    def _initialize_encoder(self):
        if self.error_type == 'init':
            raise Exception("Simulated Init Crash")
        super()._initialize_encoder()

    def _encode_frames(self, frames):
        for arr in frames:
            for i in range(arr.shape[0]):
                frame_idx = int(arr[i, 0, 0, 0])
                if self.error_type == 'encode' and frame_idx == 3:
                    raise EncoderException("Simulated Encode Exception on Frame 3", pid=mp.current_process().pid, name="DummyErrorEncoder")
        super()._encode_frames(frames)


# ================= Fixtures =================

@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    # 1. 移除默认的控制台处理器（它默认没开启 enqueue）
    logger.remove()
    
    # 2. 重新添加处理器，并开启多进程支持
    logger.add(
        sys.stderr, 
        enqueue=True,  # 核心：确保跨进程日志安全
        colorize=True, 
        format="<green>{time}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    yield

@pytest.fixture(scope="function")
def mp_manager():
    with mp.Manager() as manager:
        yield manager

@pytest.fixture(scope="function")
def record_dict(mp_manager):
    d = mp_manager.dict()
    d['init_calls'] = 0
    d['uninit_calls'] = 0
    d['encoded_frames'] = mp_manager.list()
    return d

@pytest.fixture(scope="function")
def ring_buffer():
    # 初始化真实的共享内存环形缓冲区 (容量10，大小 10x10x3)
    buf = ProcessSafeSharedRingBuffer(
        create=True, 
        buffer_capacity=10, 
        frame_shape=(10, 10, 3), 
        dtype=np.uint8
    )
    yield buf
    buf.close()
    buf.unlink()

@pytest.fixture(scope="function")
def frame_server(ring_buffer):
    fs = FrameServer(create=True, ring_buffer=ring_buffer)
    yield fs
    fs.close()
    fs.unlink()

# 辅助生成测试帧的函数
def generate_frames(start_idx, count):
    frames = np.zeros((count, 10, 10, 3), dtype=np.uint8)
    for i in range(count):
        frames[i, 0, 0, 0] = start_idx + i
    return frames


# ================= 测试用例 =================

def process_producer(buf, start_idx, count):
    # 注意：这里需要传入 buf 实例
    frames = generate_frames(start_idx, count)
    buf.put(frames)


def test_normal_lifecycle_multiple_starts(ring_buffer, frame_server, record_dict):
    """
    验证要点 2：多次 start/stop 与多进程生产者情况
    - 启动后、写入前，init 已调用。
    - 跨进程写入非整数倍 batch 的帧，stop() 后数据被完整处理。
    - uninit 被调用。
    - buffer 未销毁，再次 start() 能继续处理新写入的数据。
    """
    encoder = DummyValidEncoder(
        frame_server=frame_server, output_path="dummy.mp4", 
        batch_size=3, mp_record_dict=record_dict
    )
    
    # 1. 启动
    encoder.start()
    time.sleep(0.5) # 等待 worker 准备好
    
    # 验证：启动后写入前，_initialize_encoder 被调用 1 次
    assert record_dict['init_calls'] == 1
    assert len(record_dict['encoded_frames']) == 0

    # 2. 多进程生产者：写入 7 帧 (batch_size=3, 不是整数倍)
    p1 = mp.Process(target=process_producer, args=(ring_buffer, 0, 7))
    p1.start()
    p1.join()

    # 3. 停止 Encoder (这会触发 worker 取完剩余帧)
    encoder.stop()
    
    # 验证：uninit 被调用，且 7 帧全被编码，顺序正确
    assert record_dict['uninit_calls'] == 1
    assert list(record_dict['encoded_frames']) ==[0, 1, 2, 3, 4, 5, 6]

    # 4. Ringbuffer 继续工作：再写入 4 帧
    p2 = mp.Process(target=process_producer, args=(ring_buffer, 7, 4))
    p2.start()
    p2.join()

    # 5. 再次启动并停止 (重用 encoder 实例)
    encoder.start()
    time.sleep(0.5)
    encoder.stop()

    # 验证：生命周期重启成功，所有数据无损
    assert record_dict['init_calls'] == 2
    assert record_dict['uninit_calls'] == 2
    assert list(record_dict['encoded_frames']) ==[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_exception_handling_encode_error(ring_buffer, frame_server, record_dict):
    """
    验证要点 3：编码过程中抛出业务异常 (EncoderException)
    期望：进程捕获异常后触发 Eager Stop (不继续消费)，调用 uninit 然后退出。
    """
    encoder = DummyErrorEncoder(
        error_type='encode', frame_server=frame_server, 
        output_path="dummy.mp4", batch_size=2, mp_record_dict=record_dict
    )
    encoder.start()

    # 写入 6 帧，在第 4 帧(idx=3) 会抛出异常
    p = mp.Process(target=process_producer, args=(ring_buffer, 0, 6))
    p.start()
    p.join()
    
    # 因为内部崩溃会清空 _not_eager_stop，我们稍微等一下让它处理完崩溃逻辑
    time.sleep(1.0) 
    
    # 调用 stop 回收进程（此时内部其实已经退出）
    encoder.stop()

    # 验证：只编码了崩溃前的帧 (0, 1, 2)
    assert 3 not in list(record_dict['encoded_frames'])
    assert len(record_dict['encoded_frames']) <= 3 
    # 验证：崩溃后仍然执行了清理动作
    assert record_dict['uninit_calls'] == 1


def test_exception_handling_init_error(ring_buffer, frame_server, record_dict):
    """
    验证要点 3：初始化失败
    期望：主进程 start() 时等待 ready 超时，主动抛出 TimeoutError，并且不发生死锁。
    """
    encoder = DummyErrorEncoder(
        error_type='init', frame_server=frame_server, 
        output_path="dummy.mp4", batch_size=2, mp_record_dict=record_dict
    )
    
    # 调整超时时间让测试跑快点（原代码写死的是 10s，由于框架不好直接改，我们用 mock 强制缩短或者硬等。
    # 这里为了不改动你的源码结构，只能利用 unittest.mock 对 wait 时间做个 patch 缩短测试时间）
    import unittest.mock as mock
    with mock.patch('multiprocessing.synchronize.Event.wait', return_value=False):
        with pytest.raises(TimeoutError, match="Encoder worker failed to initialize within timeout"):
            encoder.start()
    
    # 验证：未走到 encode 和 uninit 流程
    assert record_dict['uninit_calls'] == 0
    assert len(record_dict['encoded_frames']) == 0


def test_status_update_and_lifecycle(ring_buffer, frame_server, record_dict):
    """
    验证状态更新机制的正确性及生命周期 (IPC Pipe 同步)
    - 启动时赋予较小的 stat_interval
    - 写入部分帧，验证主进程能否通过 encoder.status 读取到 frame_count 和 fps
    - 验证多次 stop/start 之间状态是否能被正确重置和重建
    """
    # 1. 实例化 Encoder，设置 stat_interval=0.2 (触发较快)
    encoder = DummyValidEncoder(
        frame_server=frame_server, output_path="dummy.mp4", 
        batch_size=2, mp_record_dict=record_dict, stat_interval=0.1
    )
    
    # 2. 首次启动并写入 10 帧 (耗时约 10 * 0.05 = 0.5s)
    encoder.start()
    p1 = mp.Process(target=process_producer, args=(ring_buffer, 0, 10))
    p1.start()
    p1.join()
    
    # 等待一小会儿确保 pipe 数据已经被主进程守护线程 poll 读取
    time.sleep(0.5) 
    
    # 3. 验证正确性
    status = encoder.status
    assert "frame_count" in status, "状态字典应包含 frame_count"
    assert status["frame_count"] >= 8, f"在4个interval中至少会获取8帧 {status}"
    assert "fps" in status, "状态字典应包含 fps"
    assert status["fps"] > 0, "fps 应该大于 0"

    # 5. 停止编码器
    encoder.stop()

    # 6. 重启编码器验证生命周期重建
    encoder.start()
    # 刚刚 start 后，状态应该被重置，且没有新状态发过来
    assert encoder.status == {}

    # 写入 7 帧
    p3 = mp.Process(target=process_producer, args=(ring_buffer, 10, 7))
    p3.start()
    p3.join()
    time.sleep(0.6)
    
    assert encoder.status["frame_count"] >= 4
    assert encoder.status["frame_count"] <= 7
    
    encoder.stop()