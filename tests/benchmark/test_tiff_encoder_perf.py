# tests/benchmark/test_tiff_encoder_perf.py

# [Result] 全系统端到端平均吞吐量: 1558.1 MB/s

import os
import sys
import time
import pytest
import numpy as np
import multiprocessing as mp
from ringbuffers.shared_ring_buffer_v2a import ProcessSafeSharedRingBuffer

# 假设你的 TiffEncoder 放在了 encoders.tiffencoder 中
from encoders.tiff_encoder import TiffEncoder 
from loguru import logger

# ----------------- 测试配置 -----------------
FRAME_H = 2048
FRAME_W = 2060
FRAME_C = 1
DTYPE = np.uint16
BYTES_PER_FRAME = FRAME_H * FRAME_W * FRAME_C * np.dtype(DTYPE).itemsize # 约 8.44 MB
TOTAL_TEST_FRAMES = 1000 # 测试 100 帧，约 844 MB 数据，防止测试过长
BATCH_SIZE = 5
FPS_SIMULATION = 0 # 0 表示全速狂飙，测试系统极限
# --------------------------------------------

@pytest.fixture(scope="function")
def real_ring_buffer():
    """创建一个真实的进程安全 RingBuffer"""
    logger.remove()
    logger.add(sys.stderr, enqueue=True)
    print(f"\n[Setup] 创建 SharedRingBuffer, 容量: {BATCH_SIZE * 4} 帧...")
    buffer = ProcessSafeSharedRingBuffer(
        create=True,
        buffer_capacity=BATCH_SIZE * 4, # 缓冲 20 帧，约 168MB 共享内存
        frame_shape=(FRAME_H, FRAME_W, FRAME_C),
        dtype=DTYPE
    )
    yield buffer
    
    print("\n[Teardown] 清理 SharedRingBuffer...")
    buffer.close()
    buffer.unlink()

@pytest.fixture(scope="function")
def temp_output_file(tmp_path):
    """提供一个测试输出文件路径"""
    output_path = tmp_path / "high_speed_test.ome.tif"
    yield str(output_path)
    # 测试结束后可选择是否删除该大文件以节省空间
    if output_path.exists():
        output_path.unlink()

def test_extreme_throughput_encoding(real_ring_buffer, temp_output_file):
    """
    全系统性能与吞吐量测试。
    测试要点：
    1. 多进程 IPC 通信 (RingBuffer) 在极限带宽下不崩溃。
    2. TiffEncoder 进程能持续写入大数据并在结束时完美落盘。
    3. 输出最终的磁盘写入吞吐量。
    """
    
    # 1. 初始化 Encoder
    print(f"\n[Test] 初始化 TiffEncoder，输出路径: {temp_output_file}")
    encoder = TiffEncoder(
        shared_buffer=real_ring_buffer,
        output_path=temp_output_file,
        batch_size=BATCH_SIZE,
        expected_fps=100.0, # 只是用于 logger 警告，不限速
        inject_logger=logger,
        frame_size=(FRAME_H, FRAME_W, FRAME_C),
        z_slices=10,
        ch_colors=["#00FF00"], # 单通道绿色
        pixel_size_um=0.2925
    )
    
    # 2. 启动子进程
    encoder.start()
    assert encoder.is_ready, "Encoder 必须准备就绪"

    # 3. 准备随机数据池 (降低生成随机数的 CPU 开销对吞吐量的影响)
    print("[Test] 预生成随机测试帧...")
    dummy_frame = np.random.randint(0, 65535, (1, FRAME_H, FRAME_W, FRAME_C), dtype=DTYPE)
    
    # 4. 全速推送数据 (主进程扮演相机)
    print(f"[Test] 开始全速推送 {TOTAL_TEST_FRAMES} 帧 ({TOTAL_TEST_FRAMES * BYTES_PER_FRAME / 1024**2:.2f} MB)...")
    
    start_time = time.perf_counter()
    frames_pushed = 0
    
    while frames_pushed < TOTAL_TEST_FRAMES:
        # 修改单个像素作为时间戳/序号，防止编译器优化
        dummy_frame[0, 0, 0, 0] = frames_pushed 
        
        # 将数据推入缓冲，如果缓冲满了则阻塞等待
        success = real_ring_buffer.put(dummy_frame, timeout=2.0)
        assert success, f"RingBuffer put 超时！可能消费端卡死。已推送: {frames_pushed}"
        
        frames_pushed += 1
        
        if FPS_SIMULATION > 0:
            time.sleep(1.0 / FPS_SIMULATION)

    push_end_time = time.perf_counter()
    push_duration = push_end_time - start_time
    push_fps = frames_pushed / push_duration
    push_mbs = (frames_pushed * BYTES_PER_FRAME / 1024**2) / push_duration
    print(f"[Result] 相机推送完毕! 耗时: {push_duration:.3f}s | 推送速率: {push_fps:.1f} FPS | 吞吐量: {push_mbs:.1f} MB/s")
    
    # 5. 停止 Encoder 并等待排空与落盘
    print("[Test] 正在停止 Encoder 并等待缓冲区排空落盘...")
    stop_start_time = time.perf_counter()
    
    encoder.stop(exit_timeout=30.0) # 给予足够的超时时间以写入磁盘
    
    stop_duration = time.perf_counter() - stop_start_time
    total_duration = time.perf_counter() - start_time
    total_mbs = (frames_pushed * BYTES_PER_FRAME / 1024**2) / total_duration
    
    print(f"[Result] Encoder 停止并落盘完毕! 缓冲排空与封装耗时: {stop_duration:.3f}s")
    print(f"[Result] 全系统端到端平均吞吐量: {total_mbs:.1f} MB/s")

    # 6. 验证结果
    # 验证缓冲是否被抽干
    assert real_ring_buffer.unread_count == 0, "停止后缓冲区应当被完全排空"
    
    # 验证物理文件是否存在且大小合理
    assert os.path.exists(temp_output_file), "物理文件不存在"
    file_size_mb = os.path.getsize(temp_output_file) / 1024**2
    print(f"[Verify] 输出文件大小: {file_size_mb:.2f} MB")
    
    # 理论大小：100帧 + (可能因 Z=10 触发黑帧补齐) => 这里 100 刚好能被 10 整除，无补齐
    expected_mb = 100 * BYTES_PER_FRAME / 1024**2
    assert file_size_mb > expected_mb, "由于包含 BigTIFF 头，文件应该略大于纯像素数据"

    # 性能断言 (放宽标准以兼容 CI 服务器，通常 NVMe 应大于 400MB/s)
    assert total_mbs > 50.0, f"端到端吞吐量过低 ({total_mbs:.1f} MB/s)，系统可能存在严重瓶颈"


