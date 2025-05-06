import av
import numpy as np

# 创建输出容器（MKV 格式支持 B 帧和 VFR）
output = av.open('output_lookahead.mkv', 'w')
stream = output.add_stream('libx264', rate=30)  # 占位帧率（实际由时间戳控制）

# 正确设置编码器参数（关键修改！）
stream.codec_context.options = {
    'preset': 'fast',
    'crf': '23',
    'x264-params': f"rc_lookahead=10:bframes=3",  # 通过 x264-params 传递lookahead
}

# 生成测试帧（10 帧绿色画面）
frames = []
for i in range(50):
    frame = av.VideoFrame.from_ndarray(
        np.zeros((480, 640, 3), dtype=np.uint8) + 
        np.array([0, 255, 0], dtype=np.uint8),  # 绿色
        format='rgb24'
    )
    frame.pts = i  # 简单 PTS（单位：帧号）
    frames.append(frame)

# 编码并观察 Packet 输出
print("开始编码（注意前几次 encode() 可能无输出）...")
for i, frame in enumerate(frames):
    packets = list(stream.encode(frame))  # 将迭代器转为列表以便观察
    print(f"输入帧 {i}, 输出 Packet 数量: {len(packets)}")
    for packet in packets:
        output.mux(packet)

# 刷新编码器缓冲区（必须调用！）
print("刷新编码器缓冲区...")
flush_packets = list(stream.encode())  # 传入 None 或空参数
print(f"刷新后输出 Packet 数量: {len(flush_packets)}")
for packet in flush_packets:
    output.mux(packet)

output.close()
print("编码完成！")

