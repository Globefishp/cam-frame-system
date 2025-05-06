import av
import numpy as np
import random

# 参数设置
width, height = 640, 480
fps = 5  # 仅为占位值，实际播放速度由时间戳决定
duration = 10  # 视频总时长（秒）
base_interval = 1.0 / fps  # 基础间隔（秒）

# 生成时间序列：基础间隔 + 小随机扰动（保持单调递增）
timestamps = []
current_time = 0.0
while current_time < duration:
    timestamps.append(current_time)
    # 在基础间隔上添加±10%的随机扰动
    current_time += base_interval * (1 + random.uniform(-0.5*base_interval, 30*base_interval))

# 初始化容器（MKV格式支持VFR）
output = av.open('output_vfr.mkv', 'w')
stream = output.add_stream('libx264')  # 不指定fps！
stream.width = width
stream.height = height
stream.pix_fmt = 'yuv420p'

# 关键设置：禁用固定帧率假设
stream.codec_context.time_base = av.time_base  # time_base=Fraction(1000000, 1), 需要取反
stream.codec_context.time_base = 1 / stream.codec_context.time_base # 使用默认微秒级时间基（1/1000000）
stream.codec_context.framerate = 0  # 直接赋值为0表示可变帧率

print("编码开始（时间戳驱动的运动）...")
for i, ts in enumerate(timestamps):
    # 生成黑色帧
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 白色方块位置由时间戳决定（匀速移动）
    square_size = 50
    speed = 100  # 像素/秒
    x = int(speed * ts) % (width - square_size)
    y = height // 2 - square_size // 2  # 垂直居中
    frame[y:y+square_size, x:x+square_size] = 255  # 白色方块
    
    # 转换为VideoFrame并设置PTS
    av_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
    av_frame.pts = int(ts * stream.codec_context.time_base.denominator)  # 外部时间戳→PTS
    
    # 编码
    packets = list(stream.encode(av_frame))
    print(f"Frame {i:2d} | Time: {ts:.3f}s | PTS: {av_frame.pts} | Packets: {len(packets)}")
    for packet in packets:
        output.mux(packet)

# 刷新编码器
flush_packets = list(stream.encode())
print(f"Flush | Packets: {len(flush_packets)}")
for packet in flush_packets:
    output.mux(packet)

output.close()
print("编码完成！")

# 验证命令
print("\n验证时间戳和播放速度：")
print("1. 查看实际时间戳：")
print("   ffprobe -show_frames -select_streams v output_vfr.mkv | grep 'pkt_pts_time'")
print("2. 播放观察方块运动是否匀速：")
print("   ffplay output_vfr.mkv")