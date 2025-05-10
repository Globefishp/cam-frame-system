import subprocess
import numpy as np
import time
import os
import sys
import select
from threading import Thread

# 视频参数
width = 640
height = 480
fps = 30
total_frames = 100

def generate_test_frame():
    """生成随机BGR24测试帧"""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

def create_tcfile(filename, num_frames, fps):
    """创建时间戳文件"""
    interval = 1.0 / fps
    with open(filename, 'w') as f:
        f.write("# timecode format v2\n")
        # f.write("0.00\n")
        for i in range(num_frames):
            f.write(f"{i * interval*1000:.2f}\n")

def read_output(process, output_type):
    """从进程读取输出(stdout/stderr)的线程函数"""
    pipe = process.stdout if output_type == 'stdout' else process.stderr
    while True:
        line = pipe.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(f"[x264 {output_type}] {line.decode().strip()}")

def main():
    tcfile_path = 'timestamps.txt'
    output_path = 'output.h264'
    create_tcfile(tcfile_path, total_frames, fps)

    x264_cmd = [
        'x264',
        '--input-res', f'{width}x{height}',
        # '--fps', str(fps),
        '--input-csp', 'bgr',
        '--input-depth', '8',
        '--tcfile-in', tcfile_path,
        '-o', output_path,
        '-'
    ]

    # 启动x264进程，捕获stdout和stderr
    process = subprocess.Popen(
        x264_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0  # 无缓冲
    )

    # 启动线程来读取stdout和stderr
    stdout_thread = Thread(target=read_output, args=(process, 'stdout'))
    stderr_thread = Thread(target=read_output, args=(process, 'stderr'))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()

    try:
        for i in range(total_frames):
            frame = generate_test_frame()
            # with open(tcfile_path, 'a') as f:
            #     f.write(f"{i * 1.0 / fps *1000:.2f}\n")
            try:
                process.stdin.write(frame.tobytes())
                process.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                print(f"Error writing frame {i}: {e}")
                break
            
            time.sleep(1/fps)
            
    except KeyboardInterrupt:
        print("\nEncoding interrupted by user")
    finally:
        # 关闭管道
        if process.stdin:
            process.stdin.close()
        
        # 等待线程结束
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        
        # 等待进程结束
        return_code = process.wait()
        # os.remove(tcfile_path)
        
        if return_code == 0:
            print(f"\nEncoding completed! Output saved to {output_path}")
        else:
            print(f"\nEncoding failed with return code {return_code}")

if __name__ == "__main__":
    main()