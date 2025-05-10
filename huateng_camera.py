import numpy as np
import mvsdk
import platform
from typing import Optional, Tuple, Union

FRAME_TIME = 5

def enumerate_cameras():
    # 枚举相机
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")
        return

    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))

    cams = []
    for i in map(lambda x: int(x), input("Select cameras: ").split()):
        cam = Camera(DevList[i])
        if cam.open():
            cams.append(cam)

class Camera(object):
    def __init__(self, DevInfo, exposure_time_ms: float = FRAME_TIME, **kwargs): # Added **kwargs
        super(Camera, self).__init__()
        self.DevInfo = DevInfo
        self.hCamera = 0
        self.cap = None
        self.pFrameBuffer = 0
        self.target_exposure_time_ms = exposure_time_ms # Store target exposure time
        self.acutal_exposure_time_ms = None # Store actual exposure time
        # Store other kwargs if needed, e.g., self.other_params = kwargs

    @property
    def width(self):
        return self.cap.sResolutionRange.iWidthMax

    @property
    def height(self):
        return self.cap.sResolutionRange.iHeightMax

    @property
    def target_fps(self) -> float:
        """Returns the target FPS based on exposure time."""
        if self.target_exposure_time_ms > 0:
            return 1000.0 / self.target_exposure_time_ms
        else:
            # Return a default or raise an error if exposure time is invalid
            print("Warning: Invalid exposure_time_ms (<= 0), cannot calculate target_fps.")
            return 0.0 # Or raise ValueError("Exposure time must be positive")
    @property
    def actual_fps(self) -> float:
        """Returns the actual FPS based on exposure time."""
        if self.acutal_exposure_time_ms is not None and self.acutal_exposure_time_ms > 0:
            return 1000.0 / self.acutal_exposure_time_ms
        elif self.acutal_exposure_time_ms is None:
            # Return 0.0 if actual exposure time is not yet available
            raise ValueError("Unexpected error in actual_fps: acutal_exposure_time_ms is "
                             "not initialized yet. Call open() before get actual_fps.")
        else:
            # Keep the ValueError for invalid positive exposure time
            raise ValueError("Unexpected error in actual_fps: Actual exposure time must be positive")


    def open(self):
        if self.hCamera > 0:
            return True

        # 打开相机
        hCamera = 0
        try:
            hCamera = mvsdk.CameraInit(self.DevInfo, -1, -1)
        except mvsdk.CameraException as e:
            print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
            return False

        # 获取相机特性描述
        cap = mvsdk.CameraGetCapability(hCamera)

        # 判断是黑白相机还是彩色相机
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

        # 分配RGB buffer，用来存放ISP输出的图像
        # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
        pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

        # 相机模式切换成连续采集
        mvsdk.CameraSetTriggerMode(hCamera, 0)

        # 手动曝光
        mvsdk.CameraSetAeState(hCamera, 0)
        # 设置曝光时间
        mvsdk.CameraSetExposureTime(hCamera, self.target_exposure_time_ms * 1000)
        # 读取实际的曝光时间
        self.acutal_exposure_time_ms = mvsdk.CameraGetExposureTime(hCamera) / 1000.0  # Convert to ms

        try:
            # 切换为手动白平衡
            mvsdk.CameraSetWbMode(hCamera, 0)
            # 设置一次白平衡
            mvsdk.CameraSetOnceWB(hCamera)
        except mvsdk.CameraException as e:
            print("CameraSetOnceWB Failed({}): {}".format(e.error_code, e.message) )

        # 让SDK内部取图线程开始工作
        mvsdk.CameraPlay(hCamera)

        self.hCamera = hCamera
        self.pFrameBuffer = pFrameBuffer
        self.cap = cap
        return True

    def close(self):
        if self.hCamera > 0:
            mvsdk.CameraUnInit(self.hCamera)
            self.hCamera = 0

        mvsdk.CameraAlignFree(self.pFrameBuffer)
        self.pFrameBuffer = 0

    def grab(self, tc: bool=False) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[int]]]:
        '''
        从相机取一帧图片
        Args:
            tc: 是否输出时间戳，以0.1ms为单位。
        
        Return: 图片数据，numpy数组，如果有时间戳，返回的是一个tuple，第一个元素是图片数据，第二个元素是时间戳
        '''
        # 从相机取一帧图片
        hCamera = self.hCamera
        pFrameBuffer = self.pFrameBuffer
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            # windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
            # linux下直接输出正的，不需要上下翻转
            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)
            
            # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
            # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )
            if tc:
                return frame, FrameHead.uiTimeStamp
            else:
                return frame
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )
            if tc:
                return None, None
            else:
                return None
