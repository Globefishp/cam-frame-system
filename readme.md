mvsdk.py:	相机SDK接口库（参考文档 WindowsSDK安装目录\Document\MVSDK_API_CHS.chm）

grab.py: 使用SDK采集图片，并保存到硬盘文件
cv_grab.py: 使用SDK采集图片，转换为opencv的图像格式

CameraSystem.py: 主模块，包含示例逻辑。包含初始化子类和Shared ring buffer管理。
videoencoder.py: 视频编码器基类，管理IPC资源。
nnanalyzer.py: 神经网络分析器基类，管理IPC资源。
x264_encoder_x264.py: 使用x264.exe实现x264编码器，带时间码记录，使用mp4fpsmod来mux时间码。
sleap_analyzer.py: 使用一个SLEAP(TopDown)策略训练的Centroid模型进行目标识别和中心点分析，底层应该是UNet。

架构图：![Architecture](Architecture(v1.3beta).png)