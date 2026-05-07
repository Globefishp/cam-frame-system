# utils/cuda_shm_pinner.py
# tested in CUDA 13.1, but most CUDA API is not that new, may allow lower version.
# refactor to cuda-python to get better API version management(which API should use v2?)

import ctypes
import sys
from contextlib import contextmanager
import threading
import multiprocessing.shared_memory as mp_shm
from typing import Optional

# CUDA Driver API Constants
CUDA_SUCCESS = 0
CU_MEMHOSTREGISTER_PORTABLE = 0x01 # The memory returned by this call will be considered as pinned memory by all CUDA contexts, not just the one that performed the allocation.
CU_MEMHOSTREGISTER_DEVICEMAP = 0x02 # Maps the allocation into the CUDA address space. The device pointer to the memory may be obtained by calling cuMemHostGetDevicePointer().

class CUDAPinner:
    """
    CUDAPinner is a helper class for marking a SharedMemory as page-locked memory IN **CUDA** driver.

    This will not affect OpenGL and external CUDA context.
    Concurrent CUDA execution while `pin`/`unpin` should be avoided,
    since it will switch the thread-local CUDA context.
    """
    def __init__(self, device_id: int = 0):
        self.cuda: Optional[ctypes.cdll] = None
        self.device_id: int = device_id
        self.device = ctypes.c_int()
        self.context = ctypes.c_void_p()
        self._lock = threading.Lock()
        self._init_driver()
    
    def _check_err_code(self, err_code: int) -> None:
        if err_code == CUDA_SUCCESS: return
        err_name_p = ctypes.c_char_p()
        err_str_p = ctypes.c_char_p()
        ret1 = self.cuda.cuGetErrorName(err_code, ctypes.byref(err_name_p))
        ret2 = self.cuda.cuGetErrorString(err_code, ctypes.byref(err_str_p))
        if ret1 != CUDA_SUCCESS or ret2 != CUDA_SUCCESS:
            raise RuntimeError(f"Invalid error code {err_code}.")
        raise RuntimeError(f"CUDA Error [{err_code}] {err_name_p.value.decode('utf-8')}: {err_str_p.value.decode('utf-8')}")

    @contextmanager
    def _use_primary_context(self):
        """Context manager for push and pop CUDA context for DLL calls."""
        with self._lock:
            current_ctx = ctypes.c_void_p()
            self._check_err_code(self.cuda.cuCtxGetCurrent(ctypes.byref(current_ctx)))
            if current_ctx.value == self.context.value: 
                # Avoid unnecessary push
                yield
            else:
                print(f"PUSH Ctx {self.context.value}")
                self._check_err_code(self.cuda.cuCtxPushCurrent_v2(self.context))
                # print(f"Set Ctx {self.context.value}")
                # self._check_err_code(self.cuda.cuCtxSetCurrent(self.context))
                try:
                    yield
                finally:
                    popped_ctx = ctypes.c_void_p()
                    self._check_err_code(self.cuda.cuCtxPopCurrent_v2(ctypes.byref(popped_ctx)))
                    print(f"POPed Ctx {popped_ctx.value}")
                    # print(f"Restore Ctx {current_ctx.value}")
                    # self._check_err_code(self.cuda.cuCtxSetCurrent(current_ctx))

    def _init_driver(self):
        try:
            if sys.platform == 'win32':
                self.cuda = ctypes.windll.nvcuda
            else:
                self.cuda = ctypes.cdll.LoadLibrary('libcuda.so.1')
        except OSError as e:
            raise RuntimeError("CUDA Driver not found.") from e
            
        # Initialize CUDA Driver
        self._check_err_code(self.cuda.cuInit(0))
        # Get default GPU handle
        self._check_err_code(self.cuda.cuDeviceGet(ctypes.byref(self.device), 0))

        # # Create a new cuda context.
        # self._check_err_code(self.cuda.cuCtxCreate_v2(ctypes.byref(self.context), 0x00, self.device))
        # Get Primary CUDA Context
        self._check_err_code(self.cuda.cuDevicePrimaryCtxRetain(ctypes.byref(self.context), self.device))
            
    # Cleanup CUDA context.
    def __del__(self):
        if self.cuda and self.context.value is not None:
            # self._check_err_code(self.cuda.cuCtxDestroy_v2(self.context))
            self._check_err_code(self.cuda.cuDevicePrimaryCtxRelease(self.device))

    def pin(self, shm: mp_shm.SharedMemory) -> None:
        """
        Pin a SaredMemory in CUDA driver. This will get *process-local* virtual 
        address of *current instance* of SharedMemory, which is different from 
        any other instances with the same name (including inherit from/to other processes). 
        
        :param shm: SharedMemory object to pin.
        :type shm: multiprocessing.shared_memory.SharedMemory
        :raises: RuntimeError: If any driver error happens.
        """
        buffer_ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
        buffer_size = shm.size
        
        # flags: PORTABLE 让所有 CUDA Context 都能访问，DEVICEMAP 允许映射到设备地址空间
        flags = CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP
            
        with self._use_primary_context():
        # CUresult cuMemHostRegister ( void* p, size_t bytesize, unsigned int  Flags )
            self._check_err_code(self.cuda.cuMemHostRegister(ctypes.c_void_p(buffer_ptr), ctypes.c_size_t(buffer_size), ctypes.c_uint(flags)))

    def unpin(self, shm: mp_shm.SharedMemory):
        """
        Unpin a SharedMemory in CUDA driver. Should be exactly the same SharedMemory
        object as the one passed to pin().

        :param shm: SharedMemory object to unpin.
        :type shm: multiprocessing.shared_memory.SharedMemory
        :raises: RuntimeError: If any driver error happens.
        """
        buffer_ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
        with self._use_primary_context():
            self._check_err_code(self.cuda.cuMemHostUnregister(ctypes.c_void_p(buffer_ptr)))
    
    # cuMemHostGetDevicePointer_v2 is valid, remember to check API version!
    def get_device_pointer(self, shm: mp_shm.SharedMemory):
        buffer_ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
        d_ptr = ctypes.c_void_p()
        with self._use_primary_context():
            self._check_err_code(self.cuda.cuMemHostGetDevicePointer_v2(ctypes.byref(d_ptr), ctypes.c_void_p(buffer_ptr), 0))
        return d_ptr

# ==== 使用示例 ====
if __name__ == "__main__":
    # 假设这是你的共享内存
    shm = mp_shm.SharedMemory(create=True, size=1024 * 1024 * 10) # 10MB

    pinner = CUDAPinner()

    # 注册为 Pinned Memory
    pinner.pin(shm)

    print(f"Device Pointer: {pinner.get_device_pointer(shm)}, Host Pointer: {ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(shm.buf)))}")
    
    # 清理
    pinner.unpin(shm)
    shm.close()
    shm.unlink()

# === Pytorch Benchmark ===
if __name__ == "__main__":
    def pytorch_benchmark():
        import torch
        import numpy as np
        import time
        # 1. 准备数据 (200MB)
        size_mb = 200
        byte_size = size_mb * 1024 * 1024
        shm = mp_shm.SharedMemory(create=True, size=byte_size)
        
        # 包装成 numpy array 方便 PyTorch 读取
        shm_np = np.frombuffer(shm.buf, dtype=np.uint8)
        
        # 初始化 CUDA Pinner
        pinner = CUDAPinner()
        
        # 创建 GPU 目标 Tensor
        target_gpu = torch.empty(byte_size, dtype=torch.uint8, device='cuda')

        # CUDA Event 计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        def run_test(name, iterations=50):
            # 预热
            for _ in range(5):
                torch.from_numpy(shm_np).to('cuda', non_blocking=True)
            
            torch.cuda.synchronize()
            
            times = []
            for _ in range(iterations):
                # 将内存转为 Tensor
                # 注意：torch.from_numpy 不拷贝数据，只是共享内存
                src_tensor = torch.from_numpy(shm_np)
                
                start_event.record()
                # 执行 HtoD 拷贝
                target_gpu.copy_(src_tensor, non_blocking=True)
                end_event.record()
                
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event)) # 单位毫秒
                
            avg_ms = sum(times) / iterations
            bandwidth = (byte_size / (1024**3)) / (avg_ms / 1000.0) # GB/s
            print(f"[{name}] 平均耗时: {avg_ms:.3f} ms | 有效带宽: {bandwidth:.2f} GB/s")

        try:
            # --- 测试 2: 已 Pin 的内存 (Pinned) ---
            print("\n正在执行 Pinner.pin()...")
            pinner.pin(shm)
            
            # 验证 PyTorch 是否识别到了这块内存是 Pinned
            # (虽然 pin 是在驱动层做的，但 PyTorch 内部调用的底层 API 会返回其状态)
            is_pinned_in_torch = torch.from_numpy(shm_np).is_pinned()
            print(f"PyTorch 检测到内存状态: {'Pinned' if is_pinned_in_torch else 'Pageable'}")
            
            print("正在测试 Pinned Memory (已 Pin)...")
            run_test("Pinned")

            # --- 测试 1: 未 Pin 的内存 (Pageable) ---
            pinner.unpin(shm)
            print("正在测试 Pageable Memory (未 Pin)...")
            run_test("Pageable")


        finally:
            del shm_np
            shm.close()
            shm.unlink()

    # === GL Benchmark ===
    def gl_benchmark():
        import moderngl
        import numpy as np
        import time
        ctx = moderngl.create_standalone_context()
        
        # 2. 准备大数据块 (例如 4K 分辨率 RGBA 字节流，约 32MB)
        # 32MB通常在三缓内, 为了让差异明显，我们用 200MB
        size_mb = 200
        byte_size = size_mb * 1024 * 1024
        shm = mp_shm.SharedMemory(create=True, size=byte_size)
        shm_np = np.frombuffer(shm.buf, dtype=np.uint8)
        shm_np[:] = np.random.randint(0, 255, size=byte_size, dtype=np.uint8)

        # 3. 创建纹理 (假设为巨大的 1D 纹理或适配尺寸的 2D)
        # ModernGL texture.write 内部调用 glTexSubImage2D
        tex = ctx.texture((4096, byte_size // (4096 * 4)), 4, dtype='f1')

        pinner = CUDAPinner()

        def benchmark(name, iterations=100):
            # 预热
            for _ in range(10):
                tex.write(shm_np)
            ctx.finish()

            start_time = time.perf_counter()
            for _ in range(iterations):
                tex.write(shm_np)
                # 注意：在 OpenGL 中，如果不调用 finish，write 可能是异步入队的
                # 为了测量真实的上传耗时，我们需要 finish
                ctx.finish() 
            end_time = time.perf_counter()

            avg_duration = (end_time - start_time) / iterations
            bandwidth = (byte_size / (1024**3)) / avg_duration
            print(f"[{name}] 平均耗时: {avg_duration*1000:.3f} ms | 有效带宽: {bandwidth:.2f} GB/s")

        try:
            # --- 测试 1: 普通内存 ---
            benchmark("Pageable (Normal)")

            # --- 测试 2: CUDA Pinned 内存 ---
            print("\n执行 CUDA Pinning...")
            pinner.pin(shm)
            benchmark("CUDA Pinned")
            pinner.unpin(shm)

        finally:
            del shm_np
            tex.release()
            shm.close()
            shm.unlink()

    gl_benchmark()
    pytorch_benchmark()
    
    
