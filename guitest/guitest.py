import cv2
import threading
import queue
import time
import numpy as np

# 假设有三个队列
camera_q = queue.Queue(maxsize=2)
processed_q = queue.Queue(maxsize=2)
network_q = queue.Queue(maxsize=1) # 网络内容可能更新不频繁
command_queue = queue.Queue() # 主线程 -> 工作线程

stop_event = threading.Event()

# --- 工作线程 (用于main_loop2) ---
def complex_preview_logic_worker(input_queue, output_queue, cmd_queue, stop_event_ref):
    print("Preview worker thread started.")
    frame_count = 0
    current_resolution = (640, 480) # 默认分辨率
    # camera = cv2.VideoCapture(0) # 假设相机在这里初始化
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, current_resolution[0])
    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, current_resolution[1])

    while not stop_event_ref.is_set():
        # 1. 处理来自主线程的命令
        try:
            command_data = cmd_queue.get_nowait() # 非阻塞获取命令
            command_type = command_data.get("type")
            payload = command_data.get("payload")

            if command_type == "SET_RESOLUTION":
                new_res = payload
                if isinstance(new_res, tuple) and len(new_res) == 2:
                    print(f"Worker: Received command to set resolution to {new_res}")
                    current_resolution = new_res
                    # 实际应用中，这里需要重新配置相机或处理逻辑
                    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, current_resolution[0])
                    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, current_resolution[1])
                    # 可能需要重新打开相机或重置某些状态
                else:
                    print(f"Worker: Invalid resolution payload: {payload}")
            # elif command_type == "OTHER_COMMAND":
            #     # ... 处理其他命令
            #     pass
            cmd_queue.task_done() # 如果使用 JoinableQueue
        except queue.Empty:
            pass # 没有命令
        except Exception as e:
            print(f"Worker: Error processing command: {e}")


        # 2. --- 模拟复杂的图像获取和处理 (使用当前分辨率) ---
        time.sleep(0.05) # 模拟处理耗时
        try:
            frame = input_queue.get_nowait() # 尝试获取帧
            if frame is None: # 上游结束
                output_queue.put(None) # 通知下游
                break
            frame = cv2.resize(frame, current_resolution, interpolation=cv2.INTER_AREA) # INTER_AREA 适合缩小
        except queue.Empty:
            frame = None # 没有新帧
            frame = np.zeros((current_resolution[1], current_resolution[0], 3), dtype=np.uint8) # 使用当前分辨率创建dummy frame
        cv2.putText(frame, f"Frame: {frame_count} Res: {current_resolution[0]}x{current_resolution[1]}",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        frame_count += 1
        # --- 复杂逻辑结束 ---

        # 3. 将处理好的帧放入队列
        try:
            if output_queue.full(): output_queue.get_nowait()
            output_queue.put_nowait(frame)
        except queue.Full:
            pass
        except Exception as e:
            print(f"Worker error putting frame: {e}")
            break
    # camera.release()
    print("Preview worker thread finishing.")
    try:
        output_queue.put_nowait(None)
    except queue.Full:
        pass

# --- 示例工作线程 ---
def camera_worker(q_out, stop_ev):
    cap = cv2.VideoCapture(0) # 尝试打开摄像头
    if not cap.isOpened():
        print("Camera worker: Cannot open camera")
        q_out.put(None) # 发送结束信号
        return
    count = 0
    while not stop_ev.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Camera worker: Failed to grab frame")
            time.sleep(0.1) # 等待一下再试
            continue
        # 模拟一些基本信息
        cv2.putText(frame, f"Cam: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        count += 1
        try:
            if q_out.full(): q_out.get_nowait() # 清理旧帧
            q_out.put_nowait(frame)
        except queue.Full:
            pass
        time.sleep(0.03) # 模拟摄像头帧率
    cap.release()
    q_out.put(None) # 发送结束信号
    print("Camera worker finished.")

def processing_worker(q_in, q_out, stop_ev):
    count = 0
    while not stop_ev.is_set():
        try:
            frame = q_in.get(timeout=0.1) # 从摄像头队列获取
            if frame is None: # 上游结束
                q_out.put(None)
                break
            
            # 模拟复杂处理
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR) # 转回BGR方便显示
            cv2.putText(processed_frame, f"Processed: {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            count += 1
            
            try:
                if q_out.full(): q_out.get_nowait()
                q_out.put_nowait(processed_frame)
            except queue.Full:
                pass
            # time.sleep(0.05) # 模拟处理耗时
        except queue.Empty:
            continue # 没有新帧
    q_out.put(None)
    print("Processing worker finished.")

def network_worker(q_out, stop_ev):
    status_messages = ["Status: OK", "Status: Loading...", "Status: Connected", "Status: Error"]
    idx = 0
    while not stop_ev.is_set():
        # 模拟网络更新
        message_frame = np.zeros((100, 400, 3), dtype=np.uint8)
        cv2.putText(message_frame, status_messages[idx % len(status_messages)], (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        idx += 1
        try:
            if q_out.full(): q_out.get_nowait()
            q_out.put_nowait(message_frame)
        except queue.Full:
            pass
        
        # 模拟网络请求/更新间隔，可能比其他线程慢很多
        for _ in range(50): # 等待 5 秒 (50 * 0.1s)
            if stop_ev.is_set(): break
            time.sleep(0.1)
    q_out.put(None)
    print("Network worker finished.")

def main_loop1():
    # --- 主线程 ---
    cv2.namedWindow("Camera Feed")
    cv2.namedWindow("Processed Feed")
    cv2.namedWindow("Network Status")

    # 启动工作线程
    cam_thread = threading.Thread(target=camera_worker, args=(camera_q, stop_event), daemon=True)
    proc_thread = threading.Thread(target=processing_worker, args=(camera_q, processed_q, stop_event), daemon=True) # 注意proc_thread的输入是camera_q
    net_thread = threading.Thread(target=network_worker, args=(network_q, stop_event), daemon=True)

    cam_thread.start()
    proc_thread.start()
    net_thread.start()

    # 用于保存上一帧，以便在队列为空时继续显示
    last_cam_frame = None
    last_proc_frame = None
    last_net_frame = None

    all_workers_done = False

    while not all_workers_done:
        cam_frame_received = False
        proc_frame_received = False
        net_frame_received = False

        # 尝试从队列获取摄像头帧
        try:
            cam_frame = camera_q.get_nowait()
            if cam_frame is None: # 收到结束信号
                print("Main: Camera worker signaled stop.")
                # cam_thread.join() # 可以选择在这里join，或者最后统一join
                last_cam_frame = None # 不再显示
            else:
                last_cam_frame = cam_frame
            cam_frame_received = True
        except queue.Empty:
            pass # 队列为空，不更新

        # 尝试从队列获取处理后的帧
        try:
            proc_frame = processed_q.get_nowait()
            if proc_frame is None:
                print("Main: Processing worker signaled stop.")
                last_proc_frame = None
            else:
                last_proc_frame = proc_frame
            proc_frame_received = True
        except queue.Empty:
            pass

        # 尝试从队列获取网络帧
        try:
            net_frame = network_q.get_nowait()
            if net_frame is None:
                print("Main: Network worker signaled stop.")
                last_net_frame = None
            else:
                last_net_frame = net_frame
            net_frame_received = True
        except queue.Empty:
            pass

        # 显示帧（如果获取到新的或有上一帧）
        if last_cam_frame is not None:
            cv2.imshow("Camera Feed", last_cam_frame)
        # else: # 可以显示一个"waiting"或"stopped"的图像

        if last_proc_frame is not None:
            cv2.imshow("Processed Feed", last_proc_frame)

        if last_net_frame is not None:
            cv2.imshow("Network Status", last_net_frame)

        key = cv2.waitKey(20) & 0xFF # 20ms等待，约50FPS的UI刷新率
        if key == ord('q'):
            print("Main: 'q' pressed, initiating shutdown.")
            stop_event.set() # 通知所有子线程停止
            break
        
        # 检查所有worker是否都已结束 (更健壮的方式是检查线程is_alive() 和队列是否都收到了None)
        # 一个简化的检查：如果所有last_xxx_frame都变成None了（因为worker发送了None信号）
        # 且对应的线程已经join（或者我们等待它们join）
        # 在这里，我们主要依赖 stop_event 和最后的 join

    print("Main: Exiting main loop. Waiting for threads to join...")
    stop_event.set() # 确保设置了

    # 等待所有线程结束
    threads = [cam_thread, proc_thread, net_thread]
    for t in threads:
        if t.is_alive():
            print(f"Main: Joining {t.name}...")
            t.join(timeout=5) # 给一个超时
            if t.is_alive():
                print(f"Main: Thread {t.name} did not terminate gracefully.")

    cv2.destroyAllWindows()
    print("Main: Application finished.")

# --- 主线程 (main_loop) 修改 ---
def main_loop2():
    # ... (之前的初始化) ...
    print("Main thread started.")
    cv2.namedWindow("Preview")
    cv2.namedWindow("Controls") # 一个简单的控制窗口示例

    # 启动子线程，传入命令队列
    cam_thread = threading.Thread(target=camera_worker, args=(camera_q, stop_event), daemon=True)
    preview_thread = threading.Thread(target=complex_preview_logic_worker,
                                      args=(camera_q, processed_q, command_queue, stop_event),
                                      daemon=True)
    cam_thread.start()
    preview_thread.start()

    last_displayed_frame = None
    supported_resolutions = [(640, 480), (800, 600), (320, 240)]
    current_res_index = 0

    def on_mouse_click(event, x, y, flags, param):
        nonlocal current_res_index
        if event == cv2.EVENT_LBUTTONDOWN:
            # 简单示例：点击控制窗口切换分辨率
            if 0 <= y < 50: # 假设按钮在顶部
                current_res_index = (current_res_index + 1) % len(supported_resolutions)
                new_res = supported_resolutions[current_res_index]
                print(f"Main: Sending SET_RESOLUTION command: {new_res}")
                command_queue.put({"type": "SET_RESOLUTION", "payload": new_res})

    cv2.setMouseCallback("Controls", on_mouse_click)


    while True:
        display_frame = None
        # ... (从 frame_queue 获取帧的逻辑不变) ...
        try:
            new_frame = processed_q.get(timeout=0.01)
            if new_frame is None: break
            last_displayed_frame = new_frame
            display_frame = new_frame
        except queue.Empty:
            if last_displayed_frame is not None:
                display_frame = last_displayed_frame
        except Exception as e:
            print(f"Main loop error getting frame: {e}")
            break

        if display_frame is not None:
            cv2.imshow("Preview", display_frame)
        else:
            placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder_frame, "Waiting for frames...", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Preview", placeholder_frame)

        # 创建一个简单的控制面板
        controls_img = np.zeros((100, 300, 3), dtype=np.uint8)
        cv2.putText(controls_img, "Click to change res", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        cv2.putText(controls_img, f"Next: {supported_resolutions[(current_res_index + 1) % len(supported_resolutions)]}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
        cv2.imshow("Controls", controls_img)


        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            # ... (退出逻辑) ...
            print("Main loop: 'q' pressed, exiting.")
            break
        elif key == ord('r'): # 示例：按 'r' 键发送切换分辨率指令
            current_res_index = (current_res_index + 1) % len(supported_resolutions)
            new_res = supported_resolutions[current_res_index]
            print(f"Main: Sending SET_RESOLUTION command: {new_res}")
            command_queue.put({"type": "SET_RESOLUTION", "payload": new_res})

    print("Main loop: Setting stop event for worker.")
    stop_event.set()
    # 如果命令队列是 JoinableQueue，可以等待命令处理完毕
    # command_queue.join()
    print("Main loop: Joining worker thread...")
    preview_thread.join(timeout=2)
    cam_thread.join(timeout=2)
    if preview_thread.is_alive() or cam_thread.is_alive():
        print("Main loop: Worker thread did not terminate gracefully.")
    cv2.destroyAllWindows()
    print("Main loop finished.")

if __name__ == "__main__":
    main_loop2()