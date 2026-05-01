import numpy as np
from numpy.typing import NDArray
from typing import Optional, Dict
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QOpenGLContext
import moderngl

class CameraDisplayWidget(QOpenGLWidget):
    """
    Custom QOpenGLWidget that uses ModernGL to draw textures provided by a background thread.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ctx: Optional[moderngl.Context] = None
        
        # Shader programs and VAOs
        self.prog = None
        self.vao = None
        
        self.ts_prog = None
        self.ts_vao = None
        
        # State from background thread
        # effective glo starts from 1.
        self.current_tex_glo = 0
        self.current_ts_glo = 0
        self.tex_w = 0
        self.tex_h = 0
        self.tex_channels = 0
        self._texture_cache: Dict[int, moderngl.Texture] = {}
        self._ts_texture_cache: Dict[int, moderngl.Texture] = {}

        # ModernGL texture wrapper cache
        self._tex_wrapper: Optional[moderngl.Texture] = None
        self._ts_tex_wrapper: Optional[moderngl.Texture] = None

        # Transform matrix
        self._transform_mtx: NDArray = np.eye(4, dtype='f4')

        # FBO and its cache
        self.fbo: Optional[moderngl.Framebuffer] = None
        self.last_fbo_id = None
        self.last_size = None
        
        self.interpolation = moderngl.LINEAR

    def initializeGL(self):
        # PySide6 has already created the context and made it current
        self.ctx = moderngl.create_context()
        
        # Shader for displaying standard texture
        vertex_shader = """
        #version 330
        uniform mat4 transform;
        in vec2 in_vert;
        in vec2 in_uv;
        out vec2 v_uv;
        void main() {
            gl_Position = transform * vec4(in_vert, 0.0, 1.0);
            v_uv = in_uv;
        }
        """
        fragment_shader = """
        #version 330
        uniform sampler2D Texture;
        uniform int channels;
        in vec2 v_uv;
        out vec4 f_color;
        void main() {
            vec4 col = texture(Texture, v_uv);
            if (channels == 1) {
                // Grayscale
                f_color = vec4(col.r, col.r, col.r, 1.0);
            } else {
                // RGB/BGR (assume RGB for now, or shader can handle swizzling if needed)
                f_color = vec4(col.bgr, 1.0);
            }
        }
        """
        self.prog = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        
        # Full screen quad (ModernGL textures are 0,0 at bottom left usually, but images are top left. 
        # We might need to flip V coordinate. Camera frames are top-left origin)
        vertices = np.array([
            # x, y, u, v
            -1.0, -1.0,  0.0, 1.0, # Bottom left
             1.0, -1.0,  1.0, 1.0, # Bottom right
            -1.0,  1.0,  0.0, 0.0, # Top left
             1.0,  1.0,  1.0, 0.0, # Top right
        ], dtype='f4')
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '2f 2f', 'in_vert', 'in_uv')])
        
        
        # --- Timestamp overlay quad ---
        ts_fragment_shader = """
        #version 330
        uniform sampler2D Texture;
        in vec2 v_uv;
        out vec4 f_color;
        void main() {
            f_color = texture(Texture, v_uv); // RGBA direct
        }
        """
        self.ts_prog = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=ts_fragment_shader)
        
        # Position: Top right, small size
        ts_vertices = np.array([
            -1.0,  0.925,  0.0, 1.0,
            -0.5,  0.925,  1.0, 1.0,
            -1.0,  1.0,    0.0, 0.0,
            -0.5,  1.0,    1.0, 0.0,
        ], dtype='f4')
        self.ts_vbo = self.ctx.buffer(ts_vertices.tobytes())
        self.ts_vao = self.ctx.vertex_array(self.ts_prog, [(self.ts_vbo, '2f 2f', 'in_vert', 'in_uv')])

        self.fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())

    @Slot(int, int, int, int, int)
    def on_frame_ready(self, tex_glo, ts_glo, w, h, channels):
        """Slot to receive updated texture metadata from the background thread."""
        if (w,h) != (self.tex_w, self.tex_h):
            self._transform_mtx = self._compute_transform_mtx(self.width(), self.height(), w, h)

        self.current_tex_glo = tex_glo
        self.current_ts_glo = ts_glo
        self.tex_w = w
        self.tex_h = h
        self.tex_channels = channels
        
        self.update()

    def resizeGL(self, w: int, h: int):
        self._transform_mtx = self._compute_transform_mtx(self.width(), self.height(), self.tex_w, self.tex_h)

    def paintGL(self):
        # 1. Ensure ModernGL renders to Qt's FBO, not the default window Framebuffer 0
        current_fbo_id = self.defaultFramebufferObject()
        if self.last_fbo_id != current_fbo_id or self.last_size != (self.width(), self.height()):
            self.fbo: moderngl.Framebuffer = self.ctx.detect_framebuffer(current_fbo_id)
            self.last_fbo_id = current_fbo_id
            self.last_size = (self.width(), self.height())

        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        
        # Enable Alpha Blending for the timestamp
        self.ctx.enable(moderngl.BLEND)
        
        if self.current_tex_glo > 0:
            tex = self._texture_cache.get(self.current_tex_glo)
            if tex is None or tex.size != (self.tex_w, self.tex_h):
                # Cache miss, update.
                # Wrap the external OpenGL texture ID into a ModernGL texture object
                tex: moderngl.Texture = self.ctx.external_texture(
                    glo=self.current_tex_glo, 
                    size=(self.tex_w, self.tex_h), 
                    components=self.tex_channels,
                    samples=0, dtype='f1')
                tex.filter = (self.interpolation, self.interpolation)
                if len(self._texture_cache) > 4:
                    # Too many cach, flush old textures. TODO: Auto GC? or NOT? 研究一下modernGL的生命周期管理.
                    self._texture_cache = {self.current_tex_glo: tex}
                else:
                    self._texture_cache[self.current_tex_glo] = tex
            
            tex.use(location=0)
            
            self.prog['transform'].write(self._transform_mtx)
            
            self.prog['Texture'].value = 0
            self.prog['channels'].value = self.tex_channels
            self.vao.render(moderngl.TRIANGLE_STRIP)
            
        if self.current_ts_glo > 0:
            # Draw Timestamp
            ts_tex = self._ts_texture_cache.get(self.current_ts_glo)
            if ts_tex is None:
                ts_tex: moderngl.Texture = self.ctx.external_texture(glo=self.current_ts_glo, 
                                                                     size=(200, 30), 
                                                                     components=4,
                                                                     samples=0, dtype='f1')
                ts_tex.filter = (self.interpolation, self.interpolation)
                if len(self._ts_texture_cache) > 4:
                    self._ts_texture_cache = {self.current_ts_glo: ts_tex}
                else:
                    self._ts_texture_cache[self.current_ts_glo] = ts_tex
            ts_tex.use(location=0)
            self.ts_prog['transform'].write(self._transform_mtx)
            self.ts_prog['Texture'].value = 0
            self.ts_vao.render(moderngl.TRIANGLE_STRIP)

    @staticmethod
    def _compute_transform_mtx(w: int, h: int, tex_w: int, tex_h: int) -> NDArray:
        """Calculate aspect ratio preserving matrix"""
        # Note that GLSL interpret matrix as F-continuous. 
        # PyGLM returns F-continuous, and `np.array` can wrap glm.mat4x4 correctly.
        # modernGL.context.write() require NDArray to be "C", no need for using order='F'.
        if tex_w == 0 or tex_h == 0 or w == 0 or h == 0:
            return np.eye(4, dtype='f4')
            
        widget_aspect = w / h
        image_aspect = tex_w / tex_h
        
        scale_x, scale_y = 1.0, 1.0
        if widget_aspect > image_aspect:
            scale_x = image_aspect / widget_aspect
        else:
            scale_y = widget_aspect / image_aspect
            
        mat = np.eye(4, dtype='f4')
        mat[0, 0] = scale_x
        mat[1, 1] = scale_y
        return mat