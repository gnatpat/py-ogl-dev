import glob
import sys
import time

from collections import *
from ctypes import *
from enum import Enum
from math import *

import numpy as np
import quaternion

import OpenGL
#OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *
from OpenGL.GL import shaders as ogl_shaders

import glfw

from PIL import Image

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600


def clamp(a, b, c):
    return max(a, min(b, c))


def make_struct(name, fields):

    class Struct:
        __slots__ = list(fields.keys())

        def __init__(self, **kwargs):
            extra_fields = kwargs.keys() - fields.keys()
            assert len(extra_fields) == 0, "Passed fields not in %s - %s" % (name, ', '.join(extra_fields))

            for key, value in ChainMap(kwargs, fields).items():
                setattr(self, key, value)

        def __repr__(self):
            field_str = ', '.join(f'{field}={getattr(self, field)}' for field in fields)
            return f"{name} {{{field_str}}}" 

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


def verticies_to_ctype(vertices):
    raw_v = ((pos.x, pos.y, pos.z, uv[0], uv[1], normal.x, normal.y, normal.z) for pos, uv, normal in vertices)
    return (c_float * (len(vertices) * 8))(*[f for fs in raw_v for f in fs])


def get_window_width():
    return glutGet(GLUT_WINDOW_WIDTH)


def get_window_height():
    return glutGet(GLUT_WINDOW_HEIGHT)


class Texture:

    def __init__(self, filename, has_alpha=True, flip=False):
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
        if INPUT.key_down(glfw.KEY_W):
            self.pos += target * speed * dt
        if INPUT.key_down(glfw.KEY_S):
            self.pos += target * -speed * dt
        if INPUT.key_down(glfw.KEY_A):
            left = target.cross(Vector3f.UP)
            left.normalize()
            self.pos += left * speed * dt
        if INPUT.key_down(glfw.KEY_D):
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
        
        m = np.array([ [u.x, u.y, u.z, 0.0],
                       [v.x, v.y, v.z, 0.0],
                       [n.x, n.y, n.z, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])

        return m * to_translation_matrix(-1.0 * self.pos)

    def get_target_vector(self):
        target = Vector3f(1.0, 0.0, 0.0)

        assert False, "quaternions still need fixing!"
        h_rot = Quaternion.from_vector_and_angle(Vector3f.UP, self.yaw)
        target = h_rot.rotate(target)

        h_axis = Vector3f.UP.cross(target)
        v_rot = Quaternion.from_vector_and_angle(h_axis, self.pitch)
        target = v_rot.rotate(target)
        return target


def from_vector_and_angle(cls, v, theta_deg):
    theta = radians(theta_deg)
    q = quaternion(v.x * sin(theta/2),
                   v.y * sin(theta/2),
                   v.z * sin(theta/2),
                   cos(theta/2))
    return q


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
    m = np.array([ [v.x, 0.0, 0.0, 0.0],
                   [0.0, v.y, 0.0, 0.0],
                   [0.0, 0.0, v.z, 0.0],
                   [0.0, 0.0, 0.0, 1.0] ])
    return m


def to_rotation_matrix(v):
    x = radians(v.x)
    y = radians(v.y)
    z = radians(v.z)

    mx, my, mz = np.zeros( (4, 4) ), np.zeros( (4, 4) ), np.zeros( (4, 4) )
    
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
    m = np.array([ [ 1.0, 0.0, 0.0, v.x],
                   [ 0.0, 1.0, 0.0, v.y],
                   [ 0.0, 0.0, 1.0, v.z],
                   [ 0.0, 0.0, 0.0, 1.0] ])
    return m


def to_perspective_projection_matrix(projection):
    fov, width, height, z_near, z_far = projection

    ar = width / height
    z_range = z_near - z_far
    tan_half_fov = tan(radians(fov / 2.0))

    m = np.zeros( (3, 4) )
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

    m = np.zeros( (4, 4) )
    m[0][0] = 2.0 / (r - l)
    m[1][1] = 2.0 / (t - b)
    m[2][2] = -2 / (far - near)

    m[0][3] = - (r + l) / (r - l)
    m[1][3] = - (t + b) / (t - b)
    m[2][3] = - (far + near) / (far - near)
    m[3][3] = 1
    return m


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


PLAYER_VELOCITY = 500

Sprite = make_struct('Sprite', 
                     {'texture': None,
                      'position': Vector2f(0, 0),
                      'size': Vector2f(32, 32),
                      'rotation': 0.0,
                      'color': Vector3f(1.0, 1.0, 1.0)})

Brick = make_struct('Brick',
                    {'solid': False,
                     'brick_id': 0,
                     'destroyed': False,
                     'pos': Vector2f(0, 0)})

Level = make_struct('Level',
                    {'bricks': [],
                     'width': 0,
                     'height': 0})


Player = make_struct('Player',
                     {'size': Vector2f(100, 20),
                      'pos': Vector2f()})


Ball = make_struct('Ball',
                   {'radius': 12.5,
                    'pos': Vector2f(),
                    'velocity': Vector2f(),
                    'stuck': True})


BreakoutState = make_struct('GameData',
                            {'player': None,
                             'level': None,
                             'ball': None})

def draw_sprites(sprites, sprite_vao, shader, textures):
    shader.use()
    glBindVertexArray(sprite_vao)
    glActiveTexture(GL_TEXTURE0)
    
    for sprite in sprites:
        centre_point_translation = Vector3f(sprite.size.x * 0.5, sprite.size.y * 0.5, 0.0) 

        model = to_translation_matrix(Vector3f(sprite.position.x, sprite.position.y, 0))
        model = model @ to_translation_matrix(centre_point_translation)
        model = model @ to_rotation_matrix(Vector3f(0.0, 0.0, sprite.rotation))
        model = model @ to_translation_matrix(-1 * centre_point_translation)
        model = model @ to_scale_matrix(Vector3f(sprite.size.x, sprite.size.y))

        shader.set("model", model)
        shader.set("spriteColor", sprite.color)
        textures[sprite.texture].bind(GL_TEXTURE0)

        glDrawArrays(GL_TRIANGLES, 0, 6)


class GameState(Enum):
    GAME_ACTIVE = 0
    GAME_MENU = 1
    GAME_WIN = 2


def load_level(path):
    level_data = open(path).read()
    level = [[int(brick_id) for brick_id in line.split()] for line in level_data.split('\n')]

    bricks = []
    for y, row in enumerate(level):
        for x, brick_id in enumerate(row):
            if brick_id == 0:
                continue
            solid = (brick_id == 1)
            bricks.append(Brick(solid=solid, brick_id=brick_id, pos=Vector2f(x, y)))

    height = len(level)
    width = len(level[0])
    return Level(bricks=bricks, width=width, height=height)


BRICK_ID_TO_COLOR = {
        1: Vector3f(0.8, 0.8, 0.7),
        2: Vector3f(0.2, 0.6, 1.0),
        3: Vector3f(0.0, 0.7, 0.0),
        4: Vector3f(0.8, 0.8, 0.4),
        5: Vector3f(1.0, 0.5, 0.0) }


def render_breakout(breakout_state, width, height):
    sprites = []

    sprites.append(Sprite(texture="paddle", position=breakout_state.player.pos, size=breakout_state.player.size))

    level = breakout_state.level
    unit_width = width / level.width
    unit_height = height / level.height
    for brick in level.bricks:
        if brick.destroyed:
            continue
        sprites.append(Sprite(texture="solid" if brick.solid else "block",
                              position=Vector2f(brick.pos.x * unit_width, brick.pos.y * unit_height),
                              size=Vector2f(unit_width, unit_height),
                              color=BRICK_ID_TO_COLOR[brick.brick_id]))

    sprites.append(Sprite(texture='ball',
                          position=breakout_state.ball.pos,
                          size=Vector2f(breakout_state.ball.radius*2, breakout_state.ball.radius*2)))

    return sprites


class GameManager:

    def __init__(self):
        now = time.time()
        self._sprite_vao = gen_sprite_buffer()

        self._shader = ShaderProgram('shader.vs', 'shader.fs')
        self._shader.use()
        self._shader.set("projection", to_orthographic_projection(SCREEN_WIDTH, SCREEN_HEIGHT))
        self._shader.set("image", 0)

        self.textures = {
                'ball': Texture('awesomeface.png', True, flip=False),
                'block': Texture('block.png', False),
                'solid': Texture('block_solid.png', False),
                'background': Texture('background.jpg', False),
                'paddle': Texture('paddle.png', True),
                        }

        self.state = GameState.GAME_ACTIVE
        self.levels = sorted(glob.glob('levels/*.lvl'))

        player_size = Vector2f(100, 20)
        player = Player(pos=Vector2f((SCREEN_WIDTH - player_size.x)/2, SCREEN_HEIGHT - player_size.y),
                        size=player_size)
        ball = Ball()
        self.breakout_state = BreakoutState(level=load_level(self.levels[0]),
                                            player=player,
                                            ball=ball)

    def update(self, dt):
        if self.state == GameState.GAME_ACTIVE:
            player = self.breakout_state.player
            velocity = PLAYER_VELOCITY * dt
            if INPUT.key_down(glfw.KEY_A):
                player.pos.x -= velocity
            if INPUT.key_down(glfw.KEY_D):
                player.pos.x += velocity
            player.pos.x = clamp(0, player.pos.x, SCREEN_WIDTH - player.size.x)

            ball = self.breakout_state.ball
            if ball.stuck and INPUT.key_pressed(glfw.KEY_SPACE):
                ball.stuck = False
                ball.velocity = Vector2f(100, -350)

            if ball.stuck:
                ball.pos.x = player.pos.x + player.size.x/2 - ball.radius
                ball.pos.y = player.pos.y - player.size.y/2 - ball.radius
                ball.velocity.x = 0
                ball.velocity.y = 0
            ball.pos += ball.velocity * dt
            if ball.pos.x <= 0:
                ball.pos.x = 0
                ball.velocity.x *= -1
            if ball.pos.x + ball.radius * 2 >= SCREEN_WIDTH:
                ball.pos.x = SCREEN_WIDTH - ball.radius * 2
                ball.velocity.x *= -1
            if ball.pos.y <= 0:
                ball.pos.y = 0
                ball.velocity.y *= -1
            

    def render(self):
        sprites = []

        start = time.time()
        if self.state == GameState.GAME_ACTIVE:
            sprites.append(Sprite(texture='background',
                                  size=Vector2f(SCREEN_WIDTH,SCREEN_HEIGHT)))
            sprites.extend(render_breakout(self.breakout_state, SCREEN_WIDTH, SCREEN_HEIGHT/2))

        start = time.time()
        draw_sprites(sprites, self._sprite_vao, self._shader, self.textures)


class Input:

    def __init__(self, window):
        self.window = window
        self.mouse_pos = None
        self.dmouse = (0.0, 0.0)
        self.dscroll = 0.0
        self.keys_pressed = set()
        self.keys_released = set()
        self.keys_down = set()


    def key_down(self, key):
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
        if isinstance(value, np.ndarray):
            if value.shape == (4, 4):
                glUniformMatrix4fv(location, 1, GL_TRUE, value)
            elif value.shape == (3,):
                glUniform3fv(location, 1, value)
            else:
                raise ValueError("don't know how to pass a np array of size %s", str(value.shape))
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

        if INPUT.key_down(glfw.KEY_Q):
            glfw.set_window_should_close(window, True)

        INPUT.update_end()


    glfw.terminate();
    return


if __name__ == "__main__":
    main()
