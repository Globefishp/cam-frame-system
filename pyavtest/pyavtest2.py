import av
import numpy as np

# 视频参数设置
frame_width = 640
frame_height = 480
frame_channels = 3  # RGB
total_frames = 120   # 生成30帧
fps = 30

# 创建输出容器（MKV格式支持B帧和VFR）
output = av.open('output_lookahead.mkv', 'w')
stream = output.add_stream('libx264', rate=fps)
stream.width = frame_width
stream.height = frame_height
stream.pix_fmt = 'yuv420p'  # 常用像素格式

# 正确设置编码器参数（关键修改！）
stream.codec_context.options = {
    'preset': 'fast',
    'crf': '23',
    'x264-params': f"rc_lookahead=10:bframes=3",  # 通过 x264-params 传递
}
# 关键设置：禁用固定帧率假设
stream.codec_context.time_base = av.time_base  # 默认微秒级（1/1000000）
stream.avg_frame_rate = av.Rational(0, 1)     # 明确标记为可变帧率

print("开始编码（观察Lookahead导致的延迟输出）...")
for i in range(total_frames):
    # 生成黑色帧
    frame = np.zeros((frame_height, frame_width, frame_channels), dtype=np.uint8)
    
    # 绘制移动的白色方块
    square_size = 50
    x = (i * 5) % (frame_width - square_size)
    y = (i * 3) % (frame_height - square_size)
    frame[y:y+square_size, x:x+square_size, :] = 255  # 白色方块
    
    # 将numpy数组转换为VideoFrame
    av_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
    av_frame.pts = i  # 设置显示时间戳
    
    # 编码当前帧
    packets = list(stream.encode(av_frame))  # 将迭代器转为列表以便观察
    
    # 打印输出信息
    print(f"输入帧 {i:2d}, 输出Packet数量: {len(packets)}")
    for packet in packets:
        output.mux(packet)

# 刷新编码器缓冲区（重要！）
print("\n刷新编码器缓冲区...")
flush_packets = list(stream.encode())
print(f"刷新后输出Packet数量: {len(flush_packets)}")
for packet in flush_packets:
    output.mux(packet)

output.close()
print("编码完成！")

# 验证命令提示
print("\n验证命令：")
print(f"ffplay output_lookahead.mkv")
print(f"ffprobe -show_frames output_lookahead.mkv | grep \"pkt_pts\|pict_type\"")