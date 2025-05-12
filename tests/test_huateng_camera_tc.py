# 测试文件开头添加这段代码，使其既能直接运行又能被pytest运行
import os
import sys
import pytest

# 只有当直接运行测试文件时才需要添加路径
if __name__ == "__main__":
    # 获取当前文件所在目录的父目录（即项目根目录）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 将项目根目录添加到Python路径
    sys.path.insert(0, project_root)

import mvsdk
from huateng_camera_tc import extract_tc_from_frames, Camera, TIMECODE_BYTES, TIMECODE_DTYPE, APPENDED_ROWS_FOR_TIMECODE
import numpy as np
import pytest

def test_extract_tc_from_frames_basic_multi_frame():
    """
    测试 extract_tc_from_frames 函数处理多帧数据的功能。
    """
    # 定义模拟帧的参数
    original_height = 100
    original_width = 200
    channels = 3
    num_frames = 5
    appended_rows = APPENDED_ROWS_FOR_TIMECODE
    combined_height = original_height + appended_rows
    timecode_bytes = TIMECODE_BYTES

    # 创建模拟的多帧数据
    # 使用 uint8 数据类型模拟图像像素和时间码字节
    combined_frames = np.zeros((num_frames, combined_height, original_width, channels), dtype=np.uint8)

    # 定义预期的时间码值
    expected_timecodes = np.array([1001, 1002, 1003, 1004, 1005], dtype=TIMECODE_DTYPE)

    # 手动将时间码嵌入到每帧的附加行中
    # 时间码存储在附加行的开头，小端字节序
    timecode_write_offset_in_frame = original_height * original_width * channels

    for i in range(num_frames):
        timecode_value = expected_timecodes[i]
        timecode_as_bytes = timecode_value.tobytes() # tobytes() defaults to little-endian

        # 将字节写入模拟帧的正确位置
        # combined_frames[i] 是一个 (combined_height, original_width, channels) 数组
        # 将其展平以便计算字节偏移量
        frame_flat = combined_frames[i].ravel()
        frame_flat[timecode_write_offset_in_frame : timecode_write_offset_in_frame + timecode_bytes] = np.frombuffer(timecode_as_bytes, dtype=np.uint8)

    # 调用 extract_tc_from_frames 函数
    extracted_images, extracted_timecodes = extract_tc_from_frames(
        combined_frames,
        original_height,
        original_width,
        channels,
        timecode_dtype=TIMECODE_DTYPE
    )

    # 断言结果
    # 检查提取的时间码是否与预期一致
    np.testing.assert_array_equal(extracted_timecodes, expected_timecodes)

    # 检查提取的图像数据形状是否正确
    assert extracted_images.shape == (num_frames, original_height, original_width, channels)

# 测试append timecode是否工作。
if __name__ == '__main__':
    import cv2 # 用于 __main__ 示例
    import time # 用于 __main__ 示例

    DevList = mvsdk.CameraEnumerateDevice()
    if len(DevList) < 1:
        print("No camera found!")
        exit()

    print("Available cameras:")
    for i, DevInfo in enumerate(DevList):
        print(f"{i}: {DevInfo.GetFriendlyName()} {DevInfo.GetPortType()}")
    
    try:
        cam_idx = int(input(f"Select camera index (0 to {len(DevList)-1}): "))
        if not (0 <= cam_idx < len(DevList)):
            print("Invalid index.")
            exit()
    except ValueError:
        print("Invalid input.")
        exit()

    # 初始化相机 (来自 huateng_camera_tc.py，因此是修改后的)
    # 此文件中的 Camera 类现在是处理时间码嵌入的类。
    # 默认不启用时间码融合
    # cam = Camera(DevList[cam_idx], exposure_time_ms=10) # exposure_time_ms 用于 ~100fps

    # 示例：启用时间码融合
    cam = Camera(DevList[cam_idx], exposure_time_ms=10, tc=True)

    if not cam.open(): # Default is tc=False
        print("Failed to open camera.")
        exit()

    print(f"Camera opened: {cam.DevInfo.GetFriendlyName()}")
    print(f"Image dimensions (HxWxC): {cam.height}x{cam.width}x{cam.channels}")
    if cam.timecode_enabled:
        print(f"Output buffer dimensions (H'xWxC): {cam.output_frame_height}x{cam.width}x{cam.channels}")
        print(f"Timecode will be stored in the first {TIMECODE_BYTES} bytes of the appended row(s).")
    else:
        print("Timecode fusion is disabled.")


    cv2.namedWindow("Image View", cv2.WINDOW_NORMAL)
    if cam.timecode_enabled:
        cv2.namedWindow("Appended Row (Raw)", cv2.WINDOW_NORMAL) # To visualize the raw appended row

    frame_count = 0
    max_frames_to_show = 200 # 限制测试的帧数

    try:
        while frame_count < max_frames_to_show:
            start_time = time.time()

            # 抓取帧
            frame_data = cam.grab()

            if frame_data is not None:
                frame_count += 1
                if cam.timecode_enabled:
                    # 如果启用了时间码，frame_data 是包含时间码的 combined_frame
                    combined_frame = frame_data
                    # 1. 提取并显示实际图像部分
                    image_view = combined_frame[:cam.height, :, :] # 切片到原始高度

                    # 2. 提取附加行
                    # 附加数据从索引 cam.height (原始高度) 开始
                    appended_data = combined_frame[cam.height:, :, :] # 这将是 (附加行, W, C)

                    # 3.a 从附加数据的开头手工提取时间码
                    # 展平第一个附加行，以便轻松获取前 TIMECODE_BYTES
                    first_appended_row_flat = appended_data[0].ravel()
                    timecode_bytes_from_buffer = first_appended_row_flat[:TIMECODE_BYTES]
                    extracted_timecode = int.from_bytes(timecode_bytes_from_buffer, byteorder='little')

                    # 3.b 使用函数提取时间码
                    # extract_tc_from_frames 期望一个帧列表，即使只有一个帧
                    _, extracted_timecodes_func = extract_tc_from_frames(
                        np.array([combined_frame]), # 将单帧放入数组
                        cam.height,
                        cam.width,
                        cam.channels,
                        timecode_dtype=TIMECODE_DTYPE
                    )
                    extracted_timecode_func = extracted_timecodes_func[0] # 获取单帧的时间码

                    # 断言手动提取的时间码与函数提取的时间码一致
                    assert extracted_timecode == extracted_timecode_func, f"Timecode mismatch, manually extract: {extracted_timecode}, got: {extracted_timecode_func} by extract_tc_from_frames"

                    # --- 可视化和检查 ---
                    # 显示图像
                    cv2.imshow("Image View", image_view)

                    # 显示原始附加行 (如果有多个，则显示第一个)
                    # 为了使其有点可见，我们可以将其重塑
                    # 这里为了让图像不要太宽，
                    # 重塑为 (64, -1)
                    display_appended_row = appended_data[0].reshape(64, -1) # (64, W*C)
                    # 如果需要，进行显示归一化，或者按原样显示
                    cv2.imshow("Appended Row (Raw)", display_appended_row)

                    # Print info
                    elapsed_ms = (time.time() - start_time) * 1000
                    print(f"Frame {frame_count}: Grab time {elapsed_ms:.2f}ms, Extracted TC: {extracted_timecode}")

                    # 逻辑检查：验证填充模式 (如果未被 TC 覆盖)
                    # fill_pattern 是 0xAA (170)
                    # 检查时间码后附加行中的字节
                    if cam.appended_rows > 0 and cam.width * cam.channels > TIMECODE_BYTES:
                        test_byte_offset_in_appended_row = TIMECODE_BYTES
                        # 检查一个应该仍然是填充模式的字节
                        # 假设 TIMECODE_BYTES < appended_row_size_in_bytes
                        bytes_per_appended_row = cam.width * cam.channels
                        if test_byte_offset_in_appended_row < bytes_per_appended_row:
                            pattern_check_byte = first_appended_row_flat[test_byte_offset_in_appended_row]
                            if pattern_check_byte == 0xAA:
                                print(f"  Pattern check: Byte at offset {test_byte_offset_in_appended_row} in appended row is {hex(pattern_check_byte)} (Correct).")
                            else:
                                print(f"  Pattern check: Byte at offset {test_byte_offset_in_appended_row} in appended row is {hex(pattern_check_byte)} (Expected {hex(0xAA)} if not overwritten by SDK beyond image).")
                        else:
                             print(f"  Pattern check: Not enough space in appended row beyond timecode for pattern check (row size: {bytes_per_appended_row}).")

                else:
                    # 如果禁用了时间码，frame_data 就是原始图像
                    image_view = frame_data
                    cv2.imshow("Image View", image_view)
                    elapsed_ms = (time.time() - start_time) * 1000
                    print(f"Frame {frame_count}: Grab time {elapsed_ms:.2f}ms (Timecode disabled)")


                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Exiting...")
                    break
                elif key == ord('s'): # 测试快照/暂停
                    print("Paused. Press any key to continue.")
                    cv2.waitKey(0)

            else:
                print("Failed to grab frame.")
                time.sleep(0.01) # 防止忙循环

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Closing camera...")
        cam.close()
        cv2.destroyAllWindows()
        print("Done.")
