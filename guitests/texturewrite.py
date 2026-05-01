import moderngl
import numpy as np
import time

# 假设已经创建了 context，例如: ctx = moderngl.create_standalone_context()
# 如果你没有单独创建，请忽略这行，使用你自己的 ctx
ctx = moderngl.create_standalone_context()

sizes = [
    (1024, 1024, "1K"),
    (4096, 4096, "4K"),
    (8192, 8192, "8K")
]

# 通道数改为 4 (RGBA)，这对 GPU DMA 传输最友好
CHANNELS = 4

# --- 预热 ---
print("Warming up GPU...")
warmup_data = np.random.randint(0, 255, (1024, 1024, CHANNELS), dtype='u1').tobytes()
warmup_tex = ctx.texture((1024, 1024), CHANNELS)
for _ in range(10):
    warmup_tex.write(warmup_data)
    ctx.finish()
warmup_tex.release()

iterations = 5

for w, h, label in sizes:
    # 提前准备数据
    data_list = [np.random.randint(0, 255, (h, w, CHANNELS), dtype='u1').tobytes() 
                 for _ in range(iterations * 2)]
    
    # 提前创建纹理，避免把显存分配时间算进去
    tex = ctx.texture((w, h), CHANNELS)

    # --- 测试 2: write() + finish() ---
    times_finish = []
    for i in range(iterations):
        d = data_list[i + iterations]
        
        start = time.perf_counter()
        tex.write(d)
        ctx.finish() # 强制 CPU 等待 GPU 完成数据搬运
        end = time.perf_counter()
        
        times_finish.append(end - start)

    avg_finish = np.mean(times_finish) * 1000
    
    # --- 测试 1: 仅调用 write() ---
    times_write = []
    for i in range(iterations):
        d = data_list[i]
        
        start = time.perf_counter()
        tex.write(d)
        # 不调用 finish，仅仅是把指令塞进驱动的命令队列
        end = time.perf_counter()
        
        times_write.append(end - start)

    avg_write = np.mean(times_write) * 1000

    # 释放纹理
    tex.release()
    
    size_mb = (w * h * CHANNELS) / (1024 * 1024)
    print(f"[{label}] {w}x{h} ({size_mb:.1f} MB, RGBA)")
    print(f"  Avg Only write():     {avg_write:8.4f} ms")
    print(f"  Avg Write + finish(): {avg_finish:8.4f} ms")
    
    if avg_finish > 0:
        speed = size_mb / (avg_finish / 1000)
        print(f"  Real Transfer Speed:  {speed:8.2f} MB/s")
    print("-" * 40)