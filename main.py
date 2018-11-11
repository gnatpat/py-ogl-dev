from collections import *
import sys, time
from math import *
from ctypes import *

from OpenGL.GL import *
from OpenGL.GL import shaders as ogl_shaders

import glfw

from PIL import Image

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

def make_struct(name, fields):

    class Struct:
        __slots__ = list(fields.keys())

        def __init__(self, **kwargs):
            extra_fields = kwargs.keys() - fields.keys()
            assert len(extra_fields) == 0, "Passed fields not in %s - %s" % (name, ', '.join(extra_fields))

            for key, value in ChainMap(kwargs, fields).items():
                setattr(self, key, value)

        def __repr__(self):
            fields = ', '.join(f'{key}={getattr(self, fields)}' for field in fields)
            return "{name} {{{fields}}}" 

    return Struct
                

PerspectiveProjection = namedtuple('PerspectiveProjection',
                                   ['fov', 'width', 'height', 'z_near', 'z_far'])


Vertex = namedtuple('Vertex', ['position', 'uv', 'normal'])

def chunk(iterable, chunk_size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def calc_normals(positions, triangle_indices):
    normals = [Vector3f() for _ in range(len(positions))]
    for triangle in triangle_indices:
        v1 = positions[triangle[1]] - positions[triangle[0]]
        v2 = positions[triangle[2]] - positions[triangle[0]]

        normal = v1.cross(v2)
        normal.normalize()

        normals[triangle[0]] += normal
        normals[triangle[1]] += normal
        normals[triangle[2]] += normal

    for normal in normals:
        normal.normalize()

    print(positions)
    print(normals)
    return normals


def verticies_to_ctype(vertices):
    raw_v = ((pos.x, pos.y, pos.z, uv[0], uv[1], normal.x, normal.y, normal.z) for pos, uv, normal in vertices)
    return (c_float * (len(vertices) * 8))(*[f for fs in raw_v for f in fs])


def get_window_width():
    return glutGet(GLUT_WINDOW_WIDTH)


def get_window_height():
    return glutGet(GLUT_WINDOW_HEIGHT)


class Texture:

    def __init__(self, filename, has_alpha=True, flip=True):
        components = 4 if has_alpha else 3
        mode = GL_RGBA if has_alpha else GL_RGB

        im = Image.open(filename)
        if flip:
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
        width, height = im.size
        pydata = [i for rgba in im.getdata() for i in rgba]
        data = (c_ubyte * (components * width * height))(*pydata)

        self.texture_object = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_object)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glTexImage2D(GL_TEXTURE_2D, 0, mode, width, height, 0,
                     mode, GL_UNSIGNED_BYTE, data)
        #glGenerateMipmap(GL_TEXTURE_2D)

    def bind(self, texture_unit):
        glActiveTexture(texture_unit)
        glBindTexture(GL_TEXTURE_2D, self.texture_object)


class Camera:

    def __init__(self, pos=None, target=None, sensitivity=0.1):
        self.pos = pos if pos else Vector3f(0.0, 0.0, 0.0)
        self.sensitivity = sensitivity

        htarget = Vector3f(target.x, 0, target.z)
        htarget.normalize()
        self.yaw = degrees(atan2(htarget.z, -htarget.x)) + 180
        self.pitch = degrees(asin(target.y))


    def update(self, dt):
        target = self.get_target_vector()
        speed = 6
        if INPUT.is_key_down(glfw.KEY_W):
            self.pos += target * speed * dt
        if INPUT.is_key_down(glfw.KEY_S):
            self.pos += target * -speed * dt
        if INPUT.is_key_down(glfw.KEY_A):
            left = target.cross(Vector3f.UP)
            left.normalize()
            self.pos += left * speed * dt
        if INPUT.is_key_down(glfw.KEY_D):
            right = Vector3f.UP.cross(target)
            right.normalize()
            self.pos += right * speed * dt

        dx, dy = INPUT.dmouse

        self.yaw += dx * self.sensitivity
        self.pitch += dy * self.sensitivity

    def to_camera_transform_matrix(self):
        n = self.get_target_vector()
        u = Vector3f.UP.cross(n)
        u.normalize()
        v = n.cross(u)
        
        m = Matrix4f()
        m[0] = [u.x, u.y, u.z, 0]
        m[1] = [v.x, v.y, v.z, 0]
        m[2] = [n.x, n.y, n.z, 0]
        m[3][3] = 1.0

        return m * to_translation_matrix(-1.0 * self.pos)

    def get_target_vector(self):
        target = Vector3f(1.0, 0.0, 0.0)

        h_rot = Quaternion.from_vector_and_angle(Vector3f.UP, self.yaw)
        target = h_rot.rotate(target)

        h_axis = Vector3f.UP.cross(target)
        v_rot = Quaternion.from_vector_and_angle(h_axis, self.pitch)
        target = v_rot.rotate(target)
        return target



class Quaternion:

    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def conj(self):
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def rotate(self, v):
        return ((self * v) * self.conj())._to_vec()

    @classmethod
    def from_vector_and_angle(cls, v, theta_deg):
        theta = radians(theta_deg)
        q = Quaternion(v.x * sin(theta/2),
                       v.y * sin(theta/2),
                       v.z * sin(theta/2),
                       cos(theta/2))
        return q

    def _to_vec(self):
        return Vector3f(self.x, self.y, self.z)

    def __mul__(self, other):
        if isinstance(other, Vector3f):
            return self * Quaternion(other.x, other.y, other.z, 0)
        if not isinstance(other, Quaternion):
            return NotImplemented
        q0, q1, q2, q3 = self.w, self.x, self.y, self.z
        r0, r1, r2, r3 = other.w, other.x, other.y, other.z
        return Quaternion(x=r0*q1 + r1*q0 - r2*q3 + r3*q2,
                          y=r0*q2 + r1*q3 + r2*q0 - r3*q1,
                          z=r0*q3 - r1*q2 + r2*q1 + r3*q0,
                          w=r0*q0 - r1*q1 - r2*q2 - r3*q3)

    def __repr__(self):
        return f"Quaternion(x={self.x:2.2f}, y={self.y:2.2f}, z={self.z:2.2f}, w={self.w:2.2f})"


class Matrix4f:

    def __init__(self, values=None):
        self.values = values if values else [0.0] * 16

    def __getitem__(self, index):
        class ListView:
            def __setitem__(inner_self, inner_index, value):
                self.values[index * 4 + inner_index] = value
            def __getitem__(inner_self, inner_index):
                return self.values[index * 4 + inner_index]
        # Grim. Context dependant object creation.
        return ListView()

    def __setitem__(self, index, value):
        self.values[index*4:index*4+4] = value

    def __mul__(self, other):
        if isinstance(other, Matrix4f):
            result = Matrix4f()
            for i in range(4):
                for j in range(4):
                    result.values[4*i+j] = sum(self.values[4 * i + k] * other.values[4 * k + j] for k in range(4))
            return result
        if isinstance(other, Vector4f):
            values = [0.0] * 4
            for i in range(4):
                values[i] = (self.values[4 * i + 0] * other.x +
                             self.values[4 * i + 1] * other.y +
                             self.values[4 * i + 2] * other.z +
                             self.values[4 * i + 3] * other.w)
            return Vector4f(*values)
        return NotImplemented

    def __repr__(self):
        return "[%3.2f %3.2f %3.2f %3.2f\n %3.2f %3.2f %3.2f %3.2f\n %3.2f %3.2f %3.2f %3.2f\n %3.2f %3.2f %3.2f %3.2f]" % tuple(self.values)

    def ctypes(self):
        return (c_float * 16)(*self.values)


Matrix4f.IDENTITY = Matrix4f([1, 0, 0, 0,
                              0, 1, 0, 0,
                              0, 0, 1, 0,
                              0, 0, 0, 1])


class Vector2f:

    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    def normalize(self):
        length = self.length()
        self.x /= length
        self.y /= length

    def length(self):
        return sqrt(self.x * self.x + self.y * self.y)

    def copy(self):
        return Vector2f(self.x, self.y)

    def __add__(self, other):
        if isinstance(other, Vector2f):
            return Vector2f(self.x+other.x, self.y+other.y)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector2f):
            return Vector2f(self.x-other.x, self.y-other.y)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Vector2f(other * self.x, other * self.y)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"Vector2f(x={self.x:2.2f}, y={self.y:2.2f})"

    def ctypes(self):
        return (c_float*2)(self.x, self.y)


class Vector3f:

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def cross(self, other):
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return Vector3f(x, y, z)

    def normalize(self):
        length = self.length()
        self.x /= length
        self.y /= length
        self.z /= length

    def length(self):
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def copy(self):
        return Vector3f(self.x, self.y, self.z)

    def __add__(self, other):
        if isinstance(other, Vector3f):
            return Vector3f(self.x+other.x, self.y+other.y, self.z+other.z)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector3f):
            return Vector3f(self.x-other.x, self.y-other.y, self.z-other.z)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Vector3f(other * self.x, other * self.y, other * self.z)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"Vector3f(x={self.x:2.2f}, y={self.y:2.2f}, z={self.z:2.2f})"

    def ctypes(self):
        return (c_float*3)(self.x, self.y, self.z)


class Vector4f:

    def __init__(self, x=0.0, y=0.0, z=0.0, w=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    def normalize(self):
        length = self.length()
        self.x /= length
        self.y /= length
        self.z /= length
        self.w /= length

    def length(self):
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)

    def copy(self):
        return Vector4f(self.x, self.y, self.z, self.w)

    def __add__(self, other):
        if isinstance(other, Vector4f):
            return Vector4f(self.x+other.x, self.y+other.y, self.z+other.z, self.w+other.w)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector4f):
            return Vector4f(self.x-other.x, self.y-other.y, self.z-other.z, self.w-other.w)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Vector4f(other * self.x, other * self.y, other * self.z, other * self.w)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"Vector4f(x={self.x:2.2f}, y={self.y:2.2f}, z={self.z:2.2f}, w={self.w:2.2f})"

    def ctypes(self):
        return (c_float*4)(self.x, self.y, self.z, self.w)


Vector3f.UP = Vector3f(0.0, 1.0, 0.0)
Vector3f.FORWARD = Vector3f(0.0, 0.0, 1.0)
Vector3f.ORIGIN = Vector3f(0.0, 0.0, 0.0)


def to_scale_matrix(v):
    m = Matrix4f()
    m[0][0] = v.x
    m[1][1] = v.y
    m[2][2] = v.z
    m[3][3] = 1.0
    return m


def to_rotation_matrix(v):
    x = radians(v.x)
    y = radians(v.y)
    z = radians(v.z)

    mx, my, mz = Matrix4f(), Matrix4f(), Matrix4f()
    
    mx[0][0] = 1.0
    mx[1][1] = cos(x)
    mx[1][2] = -sin(x)
    mx[2][1] = sin(x)
    mx[2][2] = cos(x)
    mx[3][3] = 1.0
    
    my[0][0] = cos(y)
    my[0][2] = -sin(y)
    my[1][1] = 1.0
    my[2][0] = sin(y)
    my[2][2] = cos(y)
    my[3][3] = 1.0
    
    mz[0][0] = cos(z)
    mz[0][1] = -sin(z)
    mz[1][0] = sin(z)
    mz[1][1] = cos(z)
    mz[2][2] = 1.0
    mz[3][3] = 1.0

    return mz * my * mx


def to_translation_matrix(v):
    m = Matrix4f()

    m[0][0] = 1.0
    m[1][1] = 1.0
    m[2][2] = 1.0
    m[0][3] = v.x
    m[1][3] = v.y
    m[2][3] = v.z
    m[3][3] = 1.0

    return m


def to_perspective_projection_matrix(projection):
    fov, width, height, z_near, z_far = projection

    ar = width / height
    z_range = z_near - z_far
    tan_half_fov = tan(radians(fov / 2.0))

    m = Matrix4f()
    m[0][0] = 1.0 / (tan_half_fov * ar)
    m[1][1] = 1.0 / tan_half_fov
    m[2][2] = (-z_near - z_far) / z_range
    m[2][3] = 2.0 * z_far * z_near / z_range
    m[3][2] = 1.0
    return m


def to_orthographic_projection(width, height):
    near = -1.0
    far = 1.0
    l = 0.0
    r = float(width)
    t = 0.0
    b = float(height)

    m = Matrix4f()
    m[0][0] = 2.0 / (r - l)
    m[1][1] = 2.0 / (t - b)
    m[2][2] = -2 / (far - near)

    m[0][3] = - (r + l) / (r - l)
    m[1][3] = - (t + b) / (t - b)
    m[2][3] = - (far + near) / (far - near)
    m[3][3] = 1
    return m


class Pipeline:

    def __init__(self, camera):
        self.scale = Vector3f(1.0, 1.0, 1.0)
        self.world_pos = Vector3f()
        self.rotate = Vector3f()
        self.projection = PerspectiveProjection(fov=30.0, width=1024, height=768, z_near=1.0, z_far=1000.0)
        self.camera = camera

    def set_scale(self, x, y, z):
        self.scale.x = x
        self.scale.y = y
        self.scale.z = z

    def set_pos(self, x, y, z):
        self.world_pos.x = x
        self.world_pos.y = y
        self.world_pos.z = z

    def set_rotation(self, x, y, z):
        self.rotate.x = x
        self.rotate.y = y
        self.rotate.z = z

    def set_perspective_projection(self, projection):
        self.projection = projection

    def set_camera(self, camera):
        self.camera = camera

    def bake(self):
        camera_translation_matrix = to_translation_matrix(-1.0 * self.camera.pos)
        camera_rotation_matrix = self.camera.to_camera_transform_matrix()

        projection_matrix = to_perspective_projection_matrix(self.projection)
        return (projection_matrix * 
                camera_rotation_matrix * camera_translation_matrix * 
                self.world_transformation())

    def world_transformation(self):
        scale_matrix = to_scale_matrix(self.scale)
        rotation_matrix = to_rotation_matrix(self.rotate)
        translation_matrix = to_translation_matrix(self.world_pos)

        return translation_matrix * rotation_matrix * scale_matrix


class FPSCounter:

    def __init__(self, reading_freq=1.0):
        self.reading_freq = reading_freq
        self.last_frame = time.time()
        self.duration_since_last_reading = 0.0
        self.frames_since_last_reading = 0.0
        self.last_reading = 0

    def frame(self):
        now = time.time()
        self.duration_since_last_reading += now - self.last_frame
        self.frames_since_last_reading += 1
        if self.duration_since_last_reading >= self.reading_freq:
            self.last_reading = self.frames_since_last_reading
            self.duration_since_last_reading -= self.reading_freq
            self.frames_since_last_reading = 0
            print(self.last_reading)
        self.last_frame = now

    def fps(self):
        return self.last_reading


Sprite = make_struct('Sprite', 
                     {'texture': None,
                      'position': Vector2f(0, 0),
                      'size': Vector2f(32, 32),
                      'rotation': 0.0,
                      'color': Vector3f(1.0, 1.0, 1.0)})


class GameManager:

    def __init__(self):
        self._sprite_vao = gen_sprite_buffer()

        self._shader = ShaderProgram('shader.vs', 'shader.fs')
        self._shader.use()
        self._shader.set("projection", to_orthographic_projection(SCREEN_WIDTH, SCREEN_HEIGHT))
        print(to_orthographic_projection(SCREEN_WIDTH, SCREEN_HEIGHT) * Vector4f(0, 300, 0, 1))
        self._shader.set("image", 0)

        self.textures = {
                'AWESOMEFACE': Texture('awesomeface.png', True, flip=False)
                        }

    def render(self):
        self._shader.use()

        glBindVertexArray(self._sprite_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        sprites = [Sprite(texture="AWESOMEFACE", position=Vector2f(200, 200), size=Vector2f(300, 400), rotation=45.0,
                          color=Vector3f(0.0, 1.0, 0.0))]
        glBindVertexArray(self._sprite_vao)
        glActiveTexture(GL_TEXTURE0)
        for sprite in sprites:
            model = to_translation_matrix(Vector3f(sprite.position.x, sprite.position.y, 0))
            centre_point_translation = Vector3f(sprite.size.x * 0.5, sprite.size.y * 0.5, 0.0) 
            model *= to_translation_matrix(centre_point_translation)
            model *= to_rotation_matrix(Vector3f(0.0, 0.0, sprite.rotation))
            model *= to_translation_matrix(-1 * centre_point_translation)

            model *= to_scale_matrix(Vector3f(sprite.size.x, sprite.size.y))

            self._shader.set("model", model)
            self._shader.set("spriteColor", sprite.color)

            self.textures[sprite.texture].bind(GL_TEXTURE0)

            glDrawArrays(GL_TRIANGLES, 0, 6)

    def update(self, dt):
        pass


class Input:

    def __init__(self, window):
        self.window = window
        self.mouse_pos = None
        self.dmouse = (0.0, 0.0)
        self.dscroll = 0.0
        self.keys_pressed = set()
        self.keys_released = set()
        self.keys_down = set()


    def is_key_down(self, key):
        return key in self.keys_down

    def key_pressed(self, key):
        return key in self.keys_pressed

    def key_released(self, key):
        return key in self.keys_released

    def update_end(self):
        self.dmouse = (0.0, 0.0)
        self.dscroll = 0.0

        self.keys_pressed.clear()
        self.keys_released.clear()

    def mouse_callback(self, window, xpos, ypos):
        old_mouse_pos = self.mouse_pos
        self.mouse_pos = (xpos, ypos)
        if not old_mouse_pos:
            return 
        self.dmouse = (self.mouse_pos[0] - old_mouse_pos[0], self.mouse_pos[1] - old_mouse_pos[1])

    def scroll_callback(self, window, xoffset, yoffset):
        self.dscroll = yoffset

    def key_callback(self, window, key, scancode, action, mode):
        if action == glfw.PRESS and key not in self.keys_down:
            self.keys_pressed.add(key)
            self.keys_down.add(key)
        if action == glfw.RELEASE and key in self.keys_down:
            self.keys_released.add(key)
            self.keys_down.remove(key)


class ShaderProgram:

    def __init__(self, vertex_path, fragment_path):
        vertex_shader = _compile_shader(vertex_path, GL_VERTEX_SHADER)
        fragment_shader = _compile_shader(fragment_path, GL_FRAGMENT_SHADER)

        shader_program = glCreateProgram()
        glAttachShader(shader_program, vertex_shader)
        glAttachShader(shader_program, fragment_shader)
        glLinkProgram(shader_program)
        if not glGetProgramiv(shader_program, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(shader_program))
        self.id = shader_program

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

    def use(self):
        glUseProgram(self.id)

    def set(self, name, value):
        location = glGetUniformLocation(self.id, name)
        if location == -1:
            raise ValueError(f'{name} does not exist in shader.')
        if isinstance(value, (bool, int)):
            glUniform1i(location, value)
            return
        if isinstance(value, float):
            glUniform1f(location, value)
            return
        if isinstance(value, Matrix4f):
            glUniformMatrix4fv(location, 1, GL_TRUE, value.ctypes())
            return
        if isinstance(value, Vector3f):
            glUniform3fv(location, 1, value.ctypes())
            return
        raise ValueError('tried to set uniform location in shader of bad type')


Shader = namedtuple('Shader', ['file', 'type'])


def _compile_shader(shader_file, shader_type):
    with open(shader_file) as f:
        shader_text = f.read()
    return ogl_shaders.compileShader(shader_text, shader_type)


def glfw_init():
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.RESIZABLE, GL_FALSE)


def glfw_create_window():
    window = glfw.create_window(SCREEN_WIDTH, SCREEN_HEIGHT, "LearnOpenGL", None, None)
    glfw.make_context_current(window)
    return window


def set_gl_options():
    glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
    #glEnable(GL_CULL_FACE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


def framebuffer_size_callback(window, width, height):
    global SCREEN_WIDTH, SCREEN_HEIGHT
    SCREEN_WIDTH = width
    SCREEN_HEIGHT = height
    glViewport(0, 0, width, height)


def gen_sprite_buffer():
    vertices = [
            0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 0.0, 1.0, 0.0 ]

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, (c_float * len(vertices))(*vertices), GL_STATIC_DRAW)

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(c_float), c_void_p(0))
    glEnableVertexAttribArray(0)

    return vao


def gen_triangle_buffer():
    vertices = [
            -1, -1, 0.0,
             0.0, -1, 0.0,
             -0.5,  0.0, 0.0
               ]
    indices = [
            0, 1, 2
              ]
    vao = glGenVertexArrays(1)
    vbo, ebo = glGenBuffers(2)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, (c_float * 9)(*vertices), GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (c_uint * 6)(*indices), GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(c_float), c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindVertexArray(0)
    return vao


def gen_square_buffer():
    vertices = [
             0.5,  0.5, 0.0,
             0.5, -0.5, 0.0,
            -0.5, -0.5, 0.0,
            -0.5,  0.5, 0.0
                ]

    indices = [
            3, 1, 0,
            3, 2, 1
              ]

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, (c_float * 12)(*vertices), GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (c_uint * 6)(*indices), GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(c_float), c_void_p(0))
    glEnableVertexAttribArray(0)

    return vao



def gen_textured_square_buffer():
    vertices = [
         0.5,  0.5, 0.0,  1.0, 1.0,
         0.5, -0.5, 0.0,  1.0, 0.0,
        -0.5, -0.5, 0.0,  0.0, 0.0,
        -0.5,  0.5, 0.0,  0.0, 1.0
                ]

    indices = [
            3, 1, 0,
            3, 2, 1
              ]

    vao = glGenVertexArrays(1)
    vbo, ebo = glGenBuffers(2)

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, (c_float * len(vertices))(*vertices), GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (c_uint * len(indices))(*indices), GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(c_float), c_void_p(0))
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(c_float), c_void_p(12))
    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)

    return vao


def gen_textured_cube_buffer():
    vertices = [
            -0.5, -0.5, -0.5,  0.0, 0.0,
             0.5, -0.5, -0.5,  1.0, 0.0,
             0.5,  0.5, -0.5,  1.0, 1.0,
             0.5,  0.5, -0.5,  1.0, 1.0,
            -0.5,  0.5, -0.5,  0.0, 1.0,
            -0.5, -0.5, -0.5,  0.0, 0.0,

            -0.5, -0.5,  0.5,  0.0, 0.0,
             0.5, -0.5,  0.5,  1.0, 0.0,
             0.5,  0.5,  0.5,  1.0, 1.0,
             0.5,  0.5,  0.5,  1.0, 1.0,
            -0.5,  0.5,  0.5,  0.0, 1.0,
            -0.5, -0.5,  0.5,  0.0, 0.0,

            -0.5,  0.5,  0.5,  1.0, 0.0,
            -0.5,  0.5, -0.5,  1.0, 1.0,
            -0.5, -0.5, -0.5,  0.0, 1.0,
            -0.5, -0.5, -0.5,  0.0, 1.0,
            -0.5, -0.5,  0.5,  0.0, 0.0,
            -0.5,  0.5,  0.5,  1.0, 0.0,

             0.5,  0.5,  0.5,  1.0, 0.0,
             0.5,  0.5, -0.5,  1.0, 1.0,
             0.5, -0.5, -0.5,  0.0, 1.0,
             0.5, -0.5, -0.5,  0.0, 1.0,
             0.5, -0.5,  0.5,  0.0, 0.0,
             0.5,  0.5,  0.5,  1.0, 0.0,

            -0.5, -0.5, -0.5,  0.0, 1.0,
             0.5, -0.5, -0.5,  1.0, 1.0,
             0.5, -0.5,  0.5,  1.0, 0.0,
             0.5, -0.5,  0.5,  1.0, 0.0,
            -0.5, -0.5,  0.5,  0.0, 0.0,
            -0.5, -0.5, -0.5,  0.0, 1.0,

            -0.5,  0.5, -0.5,  0.0, 1.0,
             0.5,  0.5, -0.5,  1.0, 1.0,
             0.5,  0.5,  0.5,  1.0, 0.0,
             0.5,  0.5,  0.5,  1.0, 0.0,
            -0.5,  0.5,  0.5,  0.0, 0.0,
            -0.5,  0.5, -0.5,  0.0, 1.0]

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, (c_float * len(vertices))(*vertices), GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(c_float), c_void_p(0))
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(c_float), c_void_p(12))
    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)

    return vao


def gen_cube_buffer():
    vertices = [
            -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 0.0,
             0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 0.0,
             0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 1.0,
             0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 1.0,
            -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 1.0,
            -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 0.0,
                                                          
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 0.0,
             0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 0.0,
             0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 1.0,
             0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 1.0,
            -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 1.0,
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 0.0,
                                                          
            -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  1.0, 0.0,
            -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,  1.0, 1.0,
            -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 1.0,
            -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 1.0,
            -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,  0.0, 0.0,
            -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  1.0, 0.0,
                                                          
             0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  1.0, 0.0,
             0.5,  0.5, -0.5,  1.0,  0.0,  0.0,  1.0, 1.0,
             0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  0.0, 1.0,
             0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  0.0, 1.0,
             0.5, -0.5,  0.5,  1.0,  0.0,  0.0,  0.0, 0.0,
             0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  1.0, 0.0,
                                                          
            -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0, 1.0,
             0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  1.0, 1.0,
             0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0, 0.0,
             0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0, 0.0,
            -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  0.0, 0.0,
            -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0, 1.0,
                                                          
            -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0, 1.0,
             0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  1.0, 1.0,
             0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0, 0.0,
             0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0, 0.0,
            -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  0.0, 0.0,
            -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0, 1.0]

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, (c_float * len(vertices))(*vertices), GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(c_float), c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(c_float), c_void_p(12))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(c_float), c_void_p(24))
    glEnableVertexAttribArray(2)

    return vao


INPUT = None


def main():
    glfw_init()

    window = glfw_create_window()
    set_gl_options()
    print("GL version: %s" % glGetString(GL_VERSION))

    # A bit hacky?
    global INPUT
    INPUT = Input(window)

    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glfw.set_cursor_pos_callback(window, INPUT.mouse_callback)
    glfw.set_scroll_callback(window, INPUT.scroll_callback)
    glfw.set_key_callback(window, INPUT.key_callback)

    game_manager = GameManager()
    fps = FPSCounter()

    last_time = time.time()
    fov = 45.0
    while not glfw.window_should_close(window):
        fps.frame()

        now = time.time()
        dt = now - last_time
        last_time = now
        t = now

        glfw.poll_events()

        game_manager.update(dt)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        game_manager.render()

        glfw.swap_buffers(window)

        if INPUT.is_key_down(glfw.KEY_Q):
            glfw.set_window_should_close(window, True)

        INPUT.update_end()


    glfw.terminate();
    return


if __name__ == "__main__":
    main()
