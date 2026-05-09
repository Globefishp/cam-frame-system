"""
Centralized storage for OpenGL Shaders used in the gl_widget.
"""

# Common Vertex Shader for Frame and Timestamp displays
UV_2D_VERTEX_SHADER = """
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

# Fragment Shader for standard video frames (supports 1 or 3 channels)
FRAME_FRAGMENT_SHADER = """
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
        // RGB/BGR (assume BGR if input is from OpenCV/FFmpeg, 
        // swizzling handled here if needed)
        f_color = vec4(col.bgr, 1.0);
    }
}
"""

# Fragment Shader for timestamp overlay (RGBA)
OVERLAY_2D_FRAGMENT_SHADER = """
#version 330
uniform sampler2D Texture;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = texture(Texture, v_uv); // RGBA direct
}
"""

# Vertex Shader for color primitives (lines, bboxes)
COLOR_2D_VERTEX_SHADER = """
#version 330
uniform mat4 transform;
in vec2 in_vert;
in vec4 in_color;
out vec4 v_color;
void main() {
    gl_Position = transform * vec4(in_vert, 0.0, 1.0);
    v_color = in_color;
}
"""

# Fragment Shader for color primitives
COLOR_FRAGMENT_SHADER = """
#version 330
in vec4 v_color;
out vec4 f_color;
void main() {
    f_color = v_color;
}
"""
