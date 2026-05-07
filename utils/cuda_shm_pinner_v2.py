# utils/cuda_shm_pinner.py
# require cuda-python, tested in CUDA driver API 13.1. No need to manually manage API version.

import threading
import multiprocessing.shared_memory as mp_shm
from contextlib import contextmanager
from typing import Optional

try:
    from cuda.bindings import driver as cuda
except ImportError:
    # Older version should use cuda directly
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        from cuda import cuda

class CUDAPinner:
    """
    CUDAPinner is a helper class for marking a SharedMemory as page-locked memory 
    IN **CUDA** driver.

    This will not affect OpenGL and external CUDA context.
    Concurrent CUDA execution while `pin`/`unpin` should be avoided,
    since it will switch the thread-local CUDA context.
    """
    def __init__(self, device_id: int = 0):
        self.device_id: int = device_id
        self._lock = threading.Lock()
        
        self.device = None
        self.context = None
        
        self._init_driver()

    def _check_err_code(self, err: cuda.CUresult) -> None:
        if err != cuda.CUresult.CUDA_SUCCESS:
            _, err_name = cuda.cuGetErrorName(err)
            _, err_str = cuda.cuGetErrorString(err)
            raise RuntimeError(f"CUDA Error [{err.name}]: {err_str}")

    @contextmanager
    def _use_primary_context(self):
        """Context manager for push and pop CUDA context."""
        with self._lock:
            # Pythonic API: Tuple[ret_code, Out_params, ...]
            err, current_ctx = cuda.cuCtxGetCurrent()
            self._check_err_code(err)

            if current_ctx == self.context:
                # Avoid unnecessary push
                yield
            else:
                err, = cuda.cuCtxPushCurrent(self.context)
                self._check_err_code(err)
                try:
                    yield
                finally:
                    err, popped_ctx = cuda.cuCtxPopCurrent()
                    self._check_err_code(err)

    def _init_driver(self):
        try:
            err, = cuda.cuInit(0)
            self._check_err_code(err)
        except Exception as e:
            raise RuntimeError("CUDA Driver initialization failed. Is CUDA available?") from e

        # Get default GPU handle
        err, self.device = cuda.cuDeviceGet(self.device_id)
        self._check_err_code(err)

        # Get Primary CUDA Context
        err, self.context = cuda.cuDevicePrimaryCtxRetain(self.device)
        self._check_err_code(err)

    def __del__(self):
        if self.device is not None:
            cuda.cuDevicePrimaryCtxRelease(self.device)

    def pin(self, shm: mp_shm.SharedMemory) -> None:
        """
        Pin a SaredMemory in CUDA driver. This will get *process-local* virtual 
        address of *current instance* of SharedMemory, which is different from 
        any other instances with the same name (including inherit from/to other processes). 
        
        :param shm: SharedMemory object to pin.
        :type shm: multiprocessing.shared_memory.SharedMemory
        :raises: RuntimeError: If any driver error happens.
        """

        # CU_MEMHOSTREGISTER_PORTABLE: The memory returned by this call will be 
        #   considered as pinned memory by all CUDA contexts, not just the one that 
        #   performed the allocation.
        # CU_MEMHOSTREGISTER_DEVICEMAP: Maps the allocation into the CUDA address 
        #   space. The device pointer to the memory may be obtained by calling 
        #   cuMemHostGetDevicePointer().
        flags = cuda.CU_MEMHOSTREGISTER_PORTABLE | cuda.CU_MEMHOSTREGISTER_DEVICEMAP
            
        with self._use_primary_context():
            # cuda-python use Cython that support buffer protocol (equivalent to a ptr)
            err, = cuda.cuMemHostRegister(shm.buf, shm.size, flags)
            self._check_err_code(err)

    def unpin(self, shm: mp_shm.SharedMemory):
        """
        Unpin a SharedMemory in CUDA driver. Should be exactly the same SharedMemory
        object as the one passed to pin().

        :param shm: SharedMemory object to unpin.
        :type shm: multiprocessing.shared_memory.SharedMemory
        :raises: RuntimeError: If any driver error happens.
        """
        with self._use_primary_context():
            err, = cuda.cuMemHostUnregister(shm.buf)
            self._check_err_code(err)

    def get_device_pointer(self, shm: mp_shm.SharedMemory) -> int:
        """
        Get the CUDA device pointer (address) mapped to this SharedMemory.
        """
        with self._use_primary_context():
            err, d_ptr = cuda.cuMemHostGetDevicePointer(shm.buf, 0)
            self._check_err_code(err)
            
        # ctypedef unsigned long long CUdeviceptr_v2; ctypedef CUdeviceptr_v2 CUdeviceptr
        return d_ptr

if __name__ == "__main__":
    import ctypes
    shm = mp_shm.SharedMemory(create=True, size=1024 * 1024 * 10) # 10MB

    pinner = CUDAPinner()
    # 注册为 Pinned Memory
    pinner.pin(shm)

    print(f"Device Pointer: {pinner.get_device_pointer(shm)}, Host Pointer: {ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(shm.buf)))}")
    pinner.unpin(shm)
    shm.close()
    shm.unlink()

if __name__ == "__main__":
    def pytorch_benchmark():
        import torch
        import numpy as np
        import time

        size_mb = 200
        byte_size = size_mb * 1024 * 1024
        shm = mp_shm.SharedMemory(create=True, size=byte_size)
        
        shm_np = np.frombuffer(shm.buf, dtype=np.uint8)
        
        pinner = CUDAPinner()
        
        target_gpu = torch.empty(byte_size, dtype=torch.uint8, device='cuda')

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        def run_test(name, iterations=50):
            # 预热
            for _ in range(5):
                torch.from_numpy(shm_np).to('cuda', non_blocking=True)
            
            torch.cuda.synchronize()
            
            times = []
            for _ in range(iterations):
                src_tensor = torch.from_numpy(shm_np)
                
                start_event.record()
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
    
    pytorch_benchmark()
