# Camera System For Real-time Detection and more

## 安装依赖:
```bash
conda create -n framesysrt -c conda-forge python=3.13 numpy moderngl pyopengl pyside6 loguru 
```
对于含有分析器(目前基于YOLO)的应用, 额外安装
```bash
conda activate framesysrt
conda install cuda-python==13.1.0 ultralytics pytorch-gpu torchvision pyqtgraph
```
有时, conda-forge自动解析的pytorch和torchvision版本可能会不匹配, 在当前时点(260510), 尝试使用
```bash
conda install cuda-python==13.1.0 ultralytics pytorch-gpu=2.10 torchvision==0.25 pyqtgraph
```
对于将来的LSFM应用:
```bash
conda install numba tqdm
```
其中numba负责PCOEdge4.2黑平衡加速, tqdm负责与Piezo设备通信的进度条显示.

## V2.0-pre2
- 改进了录制部分的逻辑, 支持自动分卷和手动触发分卷
- 改进了预览窗口, 现已支持鼠标拖拽和缩放,
- 增加了烧录时间戳功能(编码时时间(不准确, 但容易实现)和硬件秒级时间戳(准确的采集时间, 但为相对值)), 以及相应的UI选项.
- 修正了Analyzer UI多Tile显示数据错乱的问题.
- 改进总体UI.
- 修正了部分逻辑和错误处理, 
- 修正了部分接口的类型注解.

已知问题:
- 烧录时间戳勾选后, 由于底层零拷贝, 内存数据共享, UI也会预览到烧录的简易时间戳. 会与GPU渲染的时间戳重叠. Probably won't fix.

## V2.0-pre1
依赖倒置结构, 所有组件构建在数据总线ring_buffer_v4和多消费者分发器frameserver_v3上.
亮点:
- 完全多进程设计. 核心总线和分发支持全局单例的各线程/进程注入. 分发器支持可配置的锁读/脏读模式. 最多支持32个消费者同时读写各自32个帧区域.
- 最大限度地零拷贝设计. 数据进数据总线拷贝一次, 从总线读取分发永远返回视图而非拷贝. Analyzer基类支持数据从数据总线直接DMA上传显存(需要CUDA驱动).
- 规范化的日志和错误抛出. 遵循合适的Let it crash原则. 遵循分级raise原则, raise后就不记录error原则, 分层错误缓解原则.

- 相机模块现已完全抽象成规范接口, 支持时间码融合到帧内容中传递, 支持传出可跨进程序列化的元数据提取器, 以从一批融合数据中提取出多帧和一组元数据. 
- 包含参考相机`HuatengCamera`, 现在支持 8/12bit 软件ISP流水线, 包含可配置的颜色校正. 软件ISP由C实现, 需要SSE4.1, AVX2指令集.

- 可扩展的基类. Analyzer和VideoEncoder基类现在处理了时间码提取和所有IPC细节, 子类只需继承接口关注具体实现.
- Aanlyzer支持数据DMA上传显存, 目前支持Numpy和PyTorch两个接口. 含有丰富可配置的运行时选项, 支持连续运行和触发运行, 支持非同步读(适用于分析速度小于生产速度)和同步读(分析速度远大于生产速度). CPU密集部分完全在子进程中完成, 暴露给主进程可控的IPC接口: 支持可超时等待的,主进程状态查询和结果返回接口. 包含参考实现的`YOLOBaseAnalyzer`和`YOLOPosColorAnalyzer`.
- VideoEncoder与Analyzer类似, 但是完全自主运行, 同样可以处理时间码. 包含参考实现的`X264Encoder`.

- 前后端分离, 后端支持无头运行. 
- 前端采用PySide6 + ModernGL (OpenGL) 技术栈, 支持垂直同步和完全与生产解耦的渲染. 
- GL渲染部分, 分离数据上传和实际渲染, 使用双缓冲避免撕裂. 支持叠加时间码显示和渲染叠加几何(线/点, 在渲染检测框时很有用)
- 分析器部分, 支持`YOLOPosColorAnalyzer`中实时数据显示.(可能还需要打磨)
- 尽管前端组件仍需打磨, 但现在基本上采用插件式注入(TODO: AnalyzerWidget 需要注入Analyzer而非整个Backend)
