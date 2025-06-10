import requests
import numpy as np
import cv2
import json
import base64
import os

def create_square_image(width, height, square_size, top_left_x, top_left_y):
    """
    创建一个黑色背景，带有白色方块的图像。
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # 绘制白色方块
    image[top_left_y:top_left_y + square_size, top_left_x:top_left_x + square_size] = [255, 255, 255]
    return image

def send_image_and_prompt(url, image, prompt_data=None):
    """
    发送图像和可选的 prompt 数据到指定的 URL。
    """
    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = img_encoded.tobytes()

    files = {'image': ('image.png', img_bytes, 'image/png')}
    data = {}
    if prompt_data:
        data['prompt'] = json.dumps(prompt_data)

    try:
        response = requests.post(url, files=files, data=data, proxies={'http': None})
        response.raise_for_status() # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应内容: {e.response.text}")
        return {"status": "failure", "message": str(e)}

def save_image(image, filename):
    """
    保存图像到文件。
    """
    cv2.imwrite(filename, image)
    print(f"图像已保存到: {filename}")

if __name__ == "__main__":
    service_url = "http://222.29.33.185:5000"
    
    # 1. 生成第一个图像 (黑底白方块在中心)
    img_width, img_height = 500, 500
    square_size = 100
    # 中心位置
    center_x = (img_width - square_size) // 2
    center_y = (img_height - square_size) // 2
    image1 = create_square_image(img_width, img_height, square_size, center_x, center_y)
    save_image(image1, "image1.png")

    # 2. 调用 prompt_frame
    print("\n--- 测试 prompt_frame ---")
    # prompt 点在方块中心
    prompt_points = [[center_x + square_size // 2, center_y + square_size // 2]]
    prompt_labels = [1] # 1 for foreground
    obj_id = 123 # 任意对象 ID

    prompt_data = {
        "obj_id": obj_id,
        "points": prompt_points,
        "labels": prompt_labels
    }
    prompt_frame_response = send_image_and_prompt(f"{service_url}/prompt_frame", image1, prompt_data)
    print("prompt_frame 响应:", json.dumps(prompt_frame_response, indent=2))

    if prompt_frame_response.get("status") == "success":
        # 尝试解码并保存返回的 mask 图像
        tracked_objects = prompt_frame_response.get("tracked_objects", {})
        if str(obj_id) in tracked_objects:
            mask_b64 = tracked_objects[str(obj_id)].get("mask_image_png_b64")
            if mask_b64:
                try:
                    mask_bytes = base64.b64decode(mask_b64)
                    mask_np = np.frombuffer(mask_bytes, np.uint8)
                    mask_image = cv2.imdecode(mask_np, cv2.IMREAD_UNCHANGED)
                    if mask_image is not None:
                        cv2.imwrite("mask_prompt_frame.png", mask_image)
                        print("prompt_frame 返回的 mask 已保存到: mask_prompt_frame.png")
                    else:
                        print("无法解码 prompt_frame 返回的 mask 图像。")
                except Exception as e:
                    print(f"保存 prompt_frame mask 图像时出错: {e}")
    else:
        print("prompt_frame 调用失败，无法继续测试 track_frame。")
        exit()

    # 3. 生成第二个图像 (黑底白方块位置改变)
    # 向右下方移动
    new_center_x = center_x + 50
    new_center_y = center_y + 50
    image2 = create_square_image(img_width, img_height, square_size, new_center_x, new_center_y)
    save_image(image2, "image2.png")

    # 4. 调用 track_frame
    print("\n--- 测试 track_frame ---")
    track_frame_response = send_image_and_prompt(f"{service_url}/track_frame", image2)
    print("track_frame 响应:", json.dumps(track_frame_response, indent=2))

    if track_frame_response.get("status") == "success":
        # 尝试解码并保存返回的 mask 图像
        tracked_objects = track_frame_response.get("tracked_objects", {})
        if str(obj_id) in tracked_objects: # 仍然使用 prompt_frame 时的 obj_id
            mask_b64 = tracked_objects[str(obj_id)].get("mask_image_png_b64")
            if mask_b64:
                try:
                    mask_bytes = base64.b64decode(mask_b64)
                    mask_np = np.frombuffer(mask_bytes, np.uint8)
                    mask_image = cv2.imdecode(mask_np, cv2.IMREAD_UNCHANGED)
                    if mask_image is not None:
                        cv2.imwrite("mask_track_frame.png", mask_image)
                        print("track_frame 返回的 mask 已保存到: mask_track_frame.png")
                    else:
                        print("无法解码 track_frame 返回的 mask 图像。")
                except Exception as e:
                    print(f"保存 track_frame mask 图像时出错: {e}")
    else:
        print("track_frame 调用失败。")
