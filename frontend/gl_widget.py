from OpenGL.GL.ARB import compressed_texture_pixel_storage
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Dict
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QOpenGLContext
import moderngl
from . import gl_shaders


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
        self._ts_transform_mtx: NDArray = np.eye(4, dtype='f4')

        # FBO and its cache
        self.fbo: Optional[moderngl.Framebuffer] = None
        self.last_fbo_id = None
        self.last_size = None
        
        self.interpolation = moderngl.LINEAR

        # Zoom and pan for mouse control
        self.zoom_factor = 1.0    # [1.0, 16.0]
        self.pan_x = 0.0          # in NDC coordinates [-1, 1]
        self.pan_y = 0.0          # in NDC coordinates [-1, 1]
        self.last_mouse_pos = None # Used for recording drag state.

    def initializeGL(self):
        # PySide6 has already created the context and made it current
        self.ctx = moderngl.create_context()
        
        # === Displaying main texture ===
        self.prog = self.ctx.program(
            vertex_shader=gl_shaders.UV_2D_VERTEX_SHADER, 
            fragment_shader=gl_shaders.FRAME_FRAGMENT_SHADER
        )
        
        # Full screen quad (ModernGL textures are 0,0 at bottom left usually, but images are top left. 
        # We might need to flip V coordinate. Camera frames are top-left origin)
        # Let input 2D x y be in [-1, 1] (similar to NDC)
        vertices = np.array([
            # x, y, u, v
            -1.0, -1.0,  0.0, 1.0, # Bottom left
             1.0, -1.0,  1.0, 1.0, # Bottom right
            -1.0,  1.0,  0.0, 0.0, # Top left
             1.0,  1.0,  1.0, 0.0, # Top right
        ], dtype='f4')
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '2f 2f', 'in_vert', 'in_uv')])
        
        
        # === Timestamp overlay quad ===
        self.ts_prog = self.ctx.program(
            vertex_shader=gl_shaders.UV_2D_VERTEX_SHADER, 
            fragment_shader=gl_shaders.OVERLAY_2D_FRAGMENT_SHADER
        )
        
        # Texture space [-1,1] -> ts_transform_mtx -> upper-left
        ts_vertices = np.array([
            -1.0, -1.0,  0.0, 1.0,
             1.0, -1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 0.0,
             1.0,  1.0,  1.0, 0.0,
        ], dtype='f4')
        self.ts_vbo = self.ctx.buffer(ts_vertices)
        self.ts_vao = self.ctx.vertex_array(self.ts_prog, [(self.ts_vbo, '2f 2f', 'in_vert', 'in_uv')])

        # === Geometry Primitive Overlay ===
        self.geo_prog = self.ctx.program(
            vertex_shader=gl_shaders.COLOR_2D_VERTEX_SHADER, 
            fragment_shader=gl_shaders.COLOR_FRAGMENT_SHADER
        )
        # --- Line type object ---
        # Dynamic VBO for max 2000 vertices * 6 floats (x, y, r, g, b, a) = 48000 bytes
        self.lines_vbo = self.ctx.buffer(reserve=48000, dynamic=True)
        self.lines_vao = self.ctx.vertex_array(self.geo_prog, [(self.lines_vbo, '2f 4f', 'in_vert', 'in_color')])

        self.lines_v_n = 0 # record for the vertex number in lines_vbo.

        # --- Point type object ---
        self.points_vbo = self.ctx.buffer(reserve=48000, dynamic=True)
        self.points_vao = self.ctx.vertex_array(self.geo_prog, [(self.points_vbo, '2f 4f', 'in_vert', 'in_color')])
        self.ctx.point_size = 5.0
        self.points_v_n = 0

        # === GL FBO ===
        self.fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())

    @Slot(object)
    def update_overlay_lines(self, lines_data: Optional[NDArray]):
        """
        Receive lines data for overlay for arbitary threads.

        :param lines_data: numpy array of shape (N, 6) containing (x, y, r, g, b, a).
            x and y should be in original image pixel coordinates (origin top-left).
            lines will be connect in points order.
        """
        if lines_data is None or len(lines_data) == 0:
            self.lines_v_n = 0
            self.update()
            return

        # Normalize coordinates to frame [-1, 1] (apply global transform_mtx)
        if self.tex_w > 0 and self.tex_h > 0:
            lines_data = lines_data.copy()
            lines_data[:, 0] = (lines_data[:, 0] / self.tex_w) * 2.0 - 1.0
            lines_data[:, 1] = 1.0 - (lines_data[:, 1] / self.tex_h) * 2.0
            
        byte_data = lines_data.astype('f4').tobytes()
        if len(byte_data) <= self.lines_vbo.size:
            self.lines_vbo.write(byte_data)
            self.lines_v_n = len(lines_data)
            self.update()

    @Slot(object)
    def update_overlay_points(self, points_data: Optional[NDArray]):
        """
        Receive points data for overlay for arbitary threads.

        :param points_data: numpy array of shape (N, 6) containing (x, y, r, g, b, a).
        """
        if points_data is None or len(points_data) == 0:
            self.points_v_n = 0
            self.update()
            return

        # Normalize coordinates to frame [-1, 1]
        if self.tex_w > 0 and self.tex_h > 0:
            points_data = points_data.copy()
            points_data[:, 0] = (points_data[:, 0] / self.tex_w) * 2.0 - 1.0
            points_data[:, 1] = 1.0 - (points_data[:, 1] / self.tex_h) * 2.0
            
        byte_data = points_data.astype('f4').tobytes()
        if len(byte_data) <= self.points_vbo.size:
            self.points_vbo.write(byte_data)
            self.points_v_n = len(points_data)
            self.update()

    @Slot(int, int, int, int, int)
    def on_frame_ready(self, tex_glo, ts_glo, w, h, channels):
        """Slot to receive updated texture metadata from the background thread."""
        if (w,h) != (self.tex_w, self.tex_h):
            self._update_transform_mtx(w, h)

        self.current_tex_glo = tex_glo
        self.current_ts_glo = ts_glo
        self.tex_w = w
        self.tex_h = h
        self.tex_channels = channels
        
        self.update()

    def resizeGL(self, w: int, h: int):
        self._update_transform_mtx(self.tex_w, self.tex_h)
    
    def _update_transform_mtx(self, tex_w, tex_h):
        frame_mtx = self._frame_transform_mtx(self.width(), self.height(), tex_w, tex_h, 
                                              self.zoom_factor, self.pan_x, self.pan_y)
        ts_extra_mtx = self._timestamp_transform_mtx(200, 30)
        # modernGL.context.write() require C-continuous memory, but GLSL interprete as F-continuous.
        self._transform_mtx = frame_mtx.T.copy()
        self._ts_transform_mtx = (frame_mtx @ ts_extra_mtx).T.copy()

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
            self.ts_prog['transform'].write(self._ts_transform_mtx)
            self.ts_prog['Texture'].value = 0
            self.ts_vao.render(moderngl.TRIANGLE_STRIP)

        # Draw line type overlay
        if self.lines_v_n > 0:
            self._paint_lines_overlay()
        # Draw point type overlay
        if self.points_v_n > 0:
            self._paint_points_overlay()

    def _paint_lines_overlay(self):
        """Paint dynamic color primitives overlaid on the frame."""
        self.geo_prog['transform'].write(self._transform_mtx)
        self.lines_vao.render(moderngl.LINES, vertices=self.lines_v_n)

    def _paint_points_overlay(self):
        """Paint dynamic color primitives overlaid on the frame."""
        self.geo_prog['transform'].write(self._transform_mtx)
        self.points_vao.render(moderngl.POINTS, vertices=self.points_v_n)

    @staticmethod
    def _frame_transform_mtx(w: int, h: int, tex_w: int, tex_h: int,
                             zoom: float = 1.0, pan_x: float = 0.0, pan_y: float = 0.0) -> NDArray:
        """Calculate aspect ratio preserving matrix"""
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
        # Zoom (z_x, z_y) * Scale_fit
        mat[0, 0] = scale_x * zoom
        mat[1, 1] = scale_y * zoom
        # Pan (t_x, t_y)
        mat[0, 3] = pan_x
        mat[1, 3] = pan_y
        return mat

    @staticmethod
    def _timestamp_transform_mtx(ts_w, ts_h):
        """Calculate the timestamp transform matrix to snap it to upper-left 
        inside the frame, targeting **frame** normalized coordinate [-1,1]."""
        target_h = 0.04 * 2
        target_w = target_h * (ts_w / ts_h)

        mat_ts = np.eye(4, dtype='f4')
        mat_ts[0, 0] = target_w / 2
        mat_ts[1, 1] = target_h / 2
        mat_ts[0, 3] = -1.0 + target_w / 2 
        mat_ts[1, 3] =  1.0 - target_h / 2
        return mat_ts

    def _clamp_pan(self):
        """Clamp pan to prevent unnecessary black borders when zoomed in."""
        if self.tex_w == 0 or self.tex_h == 0 or self.width() == 0 or self.height() == 0:
            return

        widget_aspect = self.width() / self.height()
        image_aspect = self.tex_w / self.tex_h
        
        scale_x = image_aspect / widget_aspect if widget_aspect > image_aspect else 1.0
        scale_y = widget_aspect / image_aspect if widget_aspect <= image_aspect else 1.0
        
        # z*s_x = half width, max pan < content exceeds the window.
        # if half width < 1 (in NDC), no pan.
        # |t_x| <= max(0, z * s_x - 1.0)
        max_pan_x = max(0.0, self.zoom_factor * scale_x - 1.0)
        max_pan_y = max(0.0, self.zoom_factor * scale_y - 1.0)

        # Apply clamping
        self.pan_x = max(-max_pan_x, min(max_pan_x, self.pan_x))
        self.pan_y = max(-max_pan_y, min(max_pan_y, self.pan_y))

    def wheelEvent(self, event):
        """Zoom the image using the mouse position as the center of zoom."""
        if self.tex_w == 0 or self.tex_h == 0:
            return

        # Calculate mouse position in NDC [-1, 1]
        mx = (event.position().x() / self.width()) * 2.0 - 1.0
        my = 1.0 - (event.position().y() / self.height()) * 2.0 # Y-axis need flip in NDC

        # Calculate new zoom factor (step 1.2x)
        zoom_step = 1.2
        old_zoom = self.zoom_factor
        if event.angleDelta().y() > 0:
            self.zoom_factor *= zoom_step
        else:
            self.zoom_factor /= zoom_step
            
        # Limit the zoom factor [1.0 (Fit window), 16.0 (Max 1600%)]
        self.zoom_factor = max(1.0, min(16.0, self.zoom_factor))

        # Compensate pan to keep the pixel under the mouse (s_x * x_pixel) unchange
        # z * s * x + t = NDC, (NDC-t)/z is invariant before and after zoom.
        ratio = self.zoom_factor / old_zoom
        self.pan_x = mx - ratio * (mx - self.pan_x)
        self.pan_y = my - ratio * (my - self.pan_y)

        # Apply clamping and update
        self._clamp_pan()
        self._update_transform_mtx(self.tex_w, self.tex_h)
        self.update()

    def mousePressEvent(self, event):
        """Start dragging"""
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.position()

    def mouseMoveEvent(self, event):
        """Dragging the image"""
        if self.last_mouse_pos is not None:
            # Convert mouse movement to NDC offset
            dx = (event.position().x() - self.last_mouse_pos.x()) / self.width() * 2.0
            dy = -(event.position().y() - self.last_mouse_pos.y()) / self.height() * 2.0 # Qt Y downwards is positive, NDC Y upwards is positive

            self.pan_x += dx
            self.pan_y += dy
            
            self.last_mouse_pos = event.position()

            # Apply clamping and update
            self._clamp_pan()
            self._update_transform_mtx(self.tex_w, self.tex_h)
            self.update()

    def mouseReleaseEvent(self, event):
        """End dragging"""
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = None