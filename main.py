import sys
import multiprocessing as mp
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtCore import Qt

from backend.system_backend import HeadlessBackend
from frontend.main_window import MainWindow

# Important to use the specific classes requested by the user
from cameras import BitDepth
from cameras.huatengcam.huateng_camera_v4 import HuatengCamera
from encoders.x264_encoder_x264 import X264Encoder

from loguru import logger
from loguru._logger import Logger # for type hint only.

def main():
    # 1. Enforce 'spawn' start method for mp_obj_proxy
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Set logger to enqueue=True enabling multiprocess logging.
    logger.remove()
    logger.add(
        sys.stderr, 
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{process.id}</cyan>:"
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        enqueue=True
    )

    # 2. Configure PySide6 and OpenGL globally
    # Critical: This allows sharing contexts globally
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    
    # Request Core Profile context (ModernGL friendly)
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setSwapInterval(1) # Enable VSync
    QSurfaceFormat.setDefaultFormat(fmt)

    # 3. Setup Camera and Encoder configs
    # Emulate Camera Enum
    try:
        import cameras.huatengcam.mvsdk_mod as mvsdk_mod
        DevList = mvsdk_mod.CameraEnumerateDevice()
        logger.info(f"Found {len(DevList)} Huateng cameras.")
        dev_info = DevList[0] if DevList else None
    except Exception as e:
        logger.warning(f"Could not enumerate camera: {e}")
        dev_info = None

    if not dev_info:
        logger.error("No Huateng cameras found. Exiting.")
        sys.exit(1)

    camera_kwargs = {
        'dev_info': dev_info,
        'fps': 10,
        'bitdepth': BitDepth._12,
        'exposure_time_ms': 20,
        'gain': 2.0,
        'inject_logger': logger
    }
    
    encoder_kwargs = {
        # output_path is injected dynamically in start_recording
        'preset': 'fast',
        'crf': 23,
        'threads': 2,
        'batch_size': 30
    }

    # 4. Initialize HeadlessBackend
    logger.info("Initializing Headless Backend...")
    backend = HeadlessBackend(
        camera_class=HuatengCamera,
        camera_kwargs=camera_kwargs,
        encoder_class=X264Encoder,
        encoder_kwargs=encoder_kwargs,
        buffer_capacity=120,
        inject_logger=logger
    )
    
    # Start Backend Camera Process
    # backend.start_capture() # Let the UI start the capture
    logger.success("Backend Initialized.")

    # 5. Create UI
    window = MainWindow(backend)
    
    # 6. Show GUI and enter event loop
    window.show()
    ret = app.exec()

    # 7. Cleanup
    logger.info("Application exiting. Cleaning up backend...")
    backend.shutdown()
    logger.success("Cleanup complete.")
    sys.exit(ret)

if __name__ == '__main__':
    main()
