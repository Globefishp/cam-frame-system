from backend.system_backend import HeadlessBackend
from typing import List, Optional
from numpy.typing import NDArray

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QOpenGLContext, QOffscreenSurface, QImage, QPainter, QColor, QFont, QSurfaceFormat
from PySide6.QtCore import Qt
import moderngl
from OpenGL.GL import glFlush
import numpy as np
import threading

from utils.text_overlay import TextOverlayGen

class GLTextureUploadThread(QThread):
    """
    Background QThread that continuously reads frames from the backend, 
    uploads them to a ModernGL texture, and generates a timestamp texture.
    """
    # Emits: tex_glo, ts_glo, width, height, channels
    frame_ready = Signal(int, int, int, int, int)

    def __init__(self, backend: HeadlessBackend, share_context: QOpenGLContext, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.share_context = share_context
        self.running = True
        self.vsync_event = threading.Event()
        self.ctx: Optional[moderngl.Context] = None
        self.ts_gen = TextOverlayGen()
        
        # Double buffering for textures to avoid collision with rendering thread
        self.textures: List[Optional[moderngl.Texture]] = [None, None] # TODO: Do we need double buffering?
        self.write_idx = 0
        self.last_head_id = -1
        
        self.ts_texture: Optional[moderngl.Texture] = None # timestamp texture
        
        # 1. Setup Offscreen Surface & Context (Must be created in GUI thread)
        self.offscreen_surface = QOffscreenSurface()
        self.offscreen_surface.setFormat(self.share_context.format())
        self.offscreen_surface.create()

        self.gl_context = QOpenGLContext()
        self.gl_context.setFormat(self.share_context.format())
        self.gl_context.setShareContext(self.share_context)
        
        if not self.gl_context.create():
            print("Failed to create shared OpenGL context in render thread.")
        
        # Move the context to this qthread so it can be made current in run()
        self.gl_context.moveToThread(self)

    def run(self):
        if not self.gl_context.makeCurrent(self.offscreen_surface):
            print("Failed to make context current in render thread.")
            return

        # 2. Create ModernGL context bound to this QOpenGLContext
        self.ctx = moderngl.create_context()
        
        # 3. Processing loop
        while self.running:
            # Wait for vsync signal.
            # By current impl. it is frameSwapped signal, so read won't parallel with write.
            # but current double buffering is useful for preventing tearing when get/upload 
            # exceed refresh interval (next read will get the last complete texture glo)
            self.vsync_event.wait(timeout=0.033) # 30fps if no vsync received.
            self.vsync_event.clear()
            
            ticket, data = self.backend.get(size=1)
            if ticket is not None and data is not None:
                # Dirty read check: ignore if frame hasn't updated
                if ticket.head_id == self.last_head_id:
                    continue
                self.last_head_id = ticket.head_id
                
                frame: NDArray = data[0][0] # data is List[NDArray] per batch
                h, w = frame.shape[0], frame.shape[1]
                channels = frame.shape[2] if frame.ndim == 3 else 1
                dtype = "f1" if frame.dtype == np.uint8 else "nu2"
                
                # --- Main Video Texture ---
                tex = self.textures[self.write_idx]
                if (tex is None or tex.size != (w, h) or 
                   tex.components != channels or tex.dtype != dtype):
                   # (Re)create texture
                    if tex: tex.release()

                    tex = self.ctx.texture((w, h), channels, dtype=dtype)
                    self.textures[self.write_idx] = tex
                
                tex.write(frame) # blocking, TODO: will copy to driver MEM. CUDA Driver Pinned & DMA to PBO?

                # Block here ensure texture writing finished before emitting signal to UI.
                self.ctx.finish() 
                # glFlush() # flush context unblockingly. But for double buffering, no tearing is intended.
                
                # --- Create GPU Timestamp ---
                if self.ts_texture is None:
                    self.ts_texture = self.ctx.texture((self.ts_gen.width, self.ts_gen.height), 4, dtype='f1')

                self.ts_texture.write(self.ts_gen.generate())
                ts_glo = self.ts_texture.glo
                
                # Notify UI to paint
                self.frame_ready.emit(tex.glo, ts_glo, w, h, channels)
                
                # Swap buffer
                self.write_idx = 1 - self.write_idx

        # 4. Cleanup
        for tex in self.textures:
            if tex:
                tex.release()
        if self.ts_texture:
            self.ts_texture.release()

        self.gl_context.doneCurrent()

    def stop(self):
        self.running = False
        self.vsync_event.set() # wake the run loop.
        self.wait()
        self.offscreen_surface.destroy() # Offscreen should be cleaned by its creator thread.
