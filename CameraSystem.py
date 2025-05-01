import numpy as np
from threading import Thread, Event, Lock, Condition

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

from collections import deque
from typing import List, Optional, Any

from huateng_camera import Camera
from ringbuffer import RingBuffer
from shared_ring_buffer import ProcessSafeSharedRingBuffer
# 不再需要导入具体的 Analyzer 实现，只需要类型提示
# from nnanalyzer import MySpecificNNAnalyzer
from nnanalyzer import NNAnalyzer # 导入基类用于类型提示
from videoencoder import BaseVideoEncoder, X264Encoder # Import video encoder classes

import time

import mvsdk

class CameraSystem:
    """
    Manages camera frame capture, analysis, and video encoding using separate threads.

    This system orchestrates the flow of frames from a camera to an analyzer and a video encoder.
    It uses a ring buffer for temporary frame storage and threading for concurrent operations.
    The lifecycle (start/stop) of the provided analyzer and video encoder instances
    is assumed to be managed externally.
    """
    # 1. 修改 __init__ 签名，接收 analyzer 和 encoder 实例
    def __init__(self, camera: Camera, analyzer: NNAnalyzer, video_encoder: BaseVideoEncoder):
        """
        Initializes the CameraSystem with a Camera instance, an NNAnalyzer instance,
        and a BaseVideoEncoder instance.
        Args:
            camera (Camera): The camera object to grab frames from.
            analyzer (NNAnalyzer): The pre-initialized analyzer object to submit frames to.
                                   Assumes analyzer manages its own IPC resources.
            video_encoder (BaseVideoEncoder): The pre-initialized video encoder object.
                                              Assumes encoder manages its own IPC resources.
        """
        self.ring_buffer = RingBuffer(60 * 10)  # 环形缓冲60帧 * 10s
        self.snapshot_condition = Condition(Lock())  # 快照开始锁
        self.running = Event()
        self.camera = camera
        # 3. 直接使用传入的 analyzer 和 encoder 实例
        self.analyzer = analyzer
        self.video_encoder = video_encoder
        self.thread_pool = []  # 线程池，用于管理线程

    def capture_thread(self):
        while self.running.is_set():
            frame = self.camera.grab()
            # The grab method returns a *view* of ndarray
            if frame is not None:
                self.ring_buffer.put(frame)
    
    def snapshot_thread(self):
        while self.running.is_set():
            with self.snapshot_condition: # 防止多次触发但是只分析了一次的情况
                self.snapshot_condition.wait()  # 等待快照开始信号
                analysis_frame = self.ring_buffer.get_latest()
                if not self.running.is_set() or analysis_frame is None:
                    # print("Snapshot thread: No frame in buffer.") # 调试信息
                    continue

                # 3. 调用 analyzer 的 submit_frame 接口
                # print(f"Snapshot thread: Submitting frame {analysis_frame.shape}") # 调试信息
                self.analyzer.submit_frame(analysis_frame)
                # 移除直接操作共享内存和条件变量的代码
        
    def encode_thread(self):
        """
        Thread that periodically retrieves frames from the ring buffer and submits them to the video encoder.
        """
        while self.running.is_set():
            # Get all available frames from the ring buffer up to the latest
            frames = self.ring_buffer.popleft_till_latest()
            # The implementation here need to submit each frame timely (within 1/fps interval)
            # this method will also omit the first frame in the final video.
            # Actually, the all frames in RingBuffer is the same, due to RingBuffer store references only.
            # TODO: Change RingBuffer (which store references of frames captured by Camera) to
            #       ProcessSafeSharedRingBuffer (which store copies of frames) to avoid the problem.
            
            if frames:
                print(f"Encode thread: Retrieved {len(frames)} frames from buffer. Submitting for encoding...")
                # Submit each retrieved frame to the video encoder
                for frame in frames:
                    # The video_encoder.submit_frame method handles the cross-process communication
                    self.video_encoder.submit_frame(frame)
                print(f"Encode thread: Finished submitting {len(frames)} frames.")
            # The TODO comment about encoding processing is now handled by the video_encoder.submit_frame call.

    def snapshot_and_analyze(self):
        # Trigger snapshot_thread to perform analysis
        with self.snapshot_condition:
            self.snapshot_condition.notify_all()  # 发送快照开始信号

        # 4. 调用 analyzer 的 get_result 接口
        try:
            # print("Waiting for analysis result...") # 调试信息
            # 可以设置超时，避免永久阻塞
            result = self.analyzer.get_result(timeout=5.0)
            print(f"Analysis Result: {result}")
        except mp.queues.Empty:
            print("Error: Did not receive analysis result within timeout.")
        except Exception as e:
            print(f"Error getting analysis result: {e}")


    def start(self):
        """Starts the internal threads of the CameraSystem."""
        if self.running.is_set():
            print("CameraSystem already running.")
            return
        print("Starting CameraSystem threads...")
        self.running.set()
        # 4. 移除 analyzer.start() 的调用，由外部管理
        # self.analyzer.start()
        self.thread_pool.append(Thread(target=self.capture_thread, name="CaptureThread"))
        self.thread_pool.append(Thread(target=self.snapshot_thread, name="SnapshotThread"))
        self.thread_pool.append(Thread(target=self.encode_thread, name="EncodeThread"))
        for thread in self.thread_pool:
            thread.start()
    
    def stop(self):
        """Stops the internal threads of the CameraSystem."""
        if not self.running.is_set():
            print("CameraSystem already stopped.")
            return
        print("Stopping CameraSystem threads...")
        # 5. 移除 analyzer.stop() 的调用
        # self.analyzer.stop()
        self.running.clear() # 发送终止循环信号
        with self.snapshot_condition: # 挂snapshot锁
            self.snapshot_condition.notify_all() # 终止snapshot_thread的等待
        print("Waiting for CameraSystem threads to join...")
        for thread in self.thread_pool: # 等待线程池结束
            print(f"Joining {thread.name}...")
            thread.join()
            print(f"{thread.name} joined.")
        # 6. 移除共享内存的清理 (由 Analyzer 管理)
        # self.shared_mem.close()
        # self.shared_mem.unlink()
        print("CameraSystem stopped.")
        self.thread_pool = [] # 清空列表


if __name__ == "__main__":
    # 导入具体实现用于测试
    from nnanalyzer import MySpecificNNAnalyzer
    # 导入 numpy 用于计算 frame_bytes (虽然现在不需要了，但保留以防万一)
    import numpy as np

    '''
    简单的测试
    '''
    # 枚举相机并选择
    DevList = mvsdk.CameraEnumerateDevice()
    camera = Camera(DevList[0])
    camera.open()
    # 7. 在外部创建 Analyzer 和 VideoEncoder 实例
    analyzer = None
    video_encoder = None
    camera_system = None
    try:
        print("Creating Analyzer instance...")
        # TODO: 从相机或配置获取正确的尺寸
        frame_shape = (1024, 1280, 3)
        analyzer = MySpecificNNAnalyzer(
            model_path='example_path', # TODO: 使用配置
            frame_size=frame_shape
        )
        print("Analyzer instance created.")

        print("Creating VideoEncoder instance...")
        # TODO: 从相机或配置获取正确的尺寸和输出路径
        output_video_path = 'output.mp4' # Example output path
        video_encoder = X264Encoder(
            output_path=output_video_path,
            frame_size=frame_shape,
            fps=25, # Example FPS
            bitrate='800k', # Example bitrate
            buffer_capacity=600,
            batch_size=10,
        )
        print("VideoEncoder instance created.")


        # 8. 在外部创建 CameraSystem 实例并注入 Analyzer 和 VideoEncoder
        print("Creating CameraSystem instance...")
        camera_system = CameraSystem(camera, analyzer, video_encoder)
        print("CameraSystem instance created.")

        # 9. 分别启动 Analyzer, VideoEncoder, 和 CameraSystem
        print("Starting Analyzer...")
        analyzer.start() # Analyzer 管理自己的启动和 IPC 创建
        print("Starting VideoEncoder...")
        video_encoder.start() # VideoEncoder 管理自己的启动和 IPC 创建
        print("Starting CameraSystem...")
        camera_system.start() # CameraSystem 只启动自己的线程

        # 等待用户输入以停止系统
        print("System running. Press Enter to trigger snapshot and analyze.")
        time.sleep(1) # 等待线程启动
        input("Press Enter for first snapshot...")
        camera_system.snapshot_and_analyze()
        input("Press Enter for second snapshot...")
        camera_system.snapshot_and_analyze()
        input("Press Enter to stop the system...\n")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 10. 分别停止 CameraSystem, VideoEncoder, 和 Analyzer
        if camera_system:
            print("Stopping CameraSystem...")
            camera_system.stop()
        if video_encoder:
            print("Stopping VideoEncoder...")
            video_encoder.stop() # VideoEncoder 管理自己的停止和 IPC 清理
        if analyzer:
            print("Stopping Analyzer...")
            analyzer.stop() # Analyzer 管理自己的停止和 IPC 清理

        # 关闭相机
        print("Closing camera...")
        camera.close()

        print("System shutdown complete.")
