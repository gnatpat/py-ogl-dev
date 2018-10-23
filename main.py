from collections import *
import sys, time
from math import *
from ctypes import *

from OpenGL.GL import *
from OpenGL.GL import shaders as ogl_shaders
from OpenGL.GLUT import *

from PIL import Image

PerspectiveProjection = namedtuple('PerspectiveProjection',
                                   ['fov', 'width', 'height', 'z_near', 'z_far'])

Vertex = namedtuple('Vertex', ['position', 'uv'])

def verticies_to_ctype(vertices):
    raw_v = ((pos.x, pos.y, pos.z, uv[0], uv[1]) for pos, uv in vertices)
    return (c_float * (len(vertices) * 5))(*[f for fs in raw_v for f in fs])


def get_window_width():
    return glutGet(GLUT_WINDOW_WIDTH)


def get_window_height():
    return glutGet(GLUT_WINDOW_HEIGHT)


class Texture:

    def __init__(self, texture_target, filename):
        self.texture_target = texture_target

        im = Image.open(filename)
        width, height = im.size
        pydata = [i for rgba in im.getdata() for i in rgba]
        data = (c_ubyte * (4 * width * height))(*pydata)

        self.texture_object = glGenTextures(1)
        glBindTexture(texture_target, self.texture_object)
        glTexImage2D(texture_target, 0, GL_RGBA, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, data)

        glTexParameterf(texture_target, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(texture_target, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def bind(self, texture_unit):
        glActiveTexture(texture_unit)
        glBindTexture(self.texture_target, self.texture_object)


class MouseTrap:

    def __init__(self, camera):
        self.camera = camera
        self.mouse_x = get_window_width()//2
        self.mouse_y = get_window_height()//2
        glutWarpPointer(self.mouse_x, self.mouse_y)

    def mouse(self, x, y):
        dx, self.mouse_x = x - self.mouse_x, x
        dy, self.mouse_y = y - self.mouse_y, y

        self.camera.mouse(dx, dy)

        mid_x, mid_y = get_window_width()//2, get_window_height()//2
        distance_from_centre = sqrt((x - mid_x)**2 + (y - mid_y)**2)
        if distance_from_centre > (get_window_width()//10):
            self.mouse_x = mid_x
            self.mouse_y = mid_y
            glutWarpPointer(mid_x, mid_y)

class Camera:

    def __init__(self, pos=None, target=None):
        self.pos = pos if pos else Vector3f(0.0, 0.0, 0.0)

        htarget = Vector3f(target.x, 0, target.z)
        htarget.normalize()
        self.angle_h = degrees(atan2(htarget.z, -htarget.x)) + 180
        self.angle_v = degrees(asin(target.y))

    def mouse(self, dx, dy):
        self.angle_h += dx / 20.0
        self.angle_v += dy / 20.0

    def keyboard(self, key, x, y):
        target = self._get_target_vector()
        speed = 0.1
        if key == GLUT_KEY_UP:
            self.pos += target * speed
        if key == GLUT_KEY_DOWN:
            self.pos += target * -speed
        if key == GLUT_KEY_LEFT:
            left = target.cross(Vector3f.UP)
            left.normalize()
            self.pos += left * speed
        if key == GLUT_KEY_RIGHT:
            right = self.up.cross(target)
            right.normalize()
            self.pos += right * speed

    def to_camera_transform_matrix(self):
        n = self._get_target_vector()
        u = Vector3f.UP.cross(n)
        u.normalize()
        v = n.cross(u)
        
        m = Matrix4f()
        m[0] = [u.x, u.y, u.z, 0]
        m[1] = [v.x, v.y, v.z, 0]
        m[2] = [n.x, n.y, n.z, 0]
        m[3][3] = 1.0
        return m

    def _get_target_vector(self):
        target = Vector3f(1.0, 0.0, 0.0)

        h_rot = Quaternion.from_vector_and_angle(Vector3f.UP, self.angle_h)
        target = h_rot.rotate(target)

        h_axis = Vector3f.UP.cross(target)
        v_rot = Quaternion.from_vector_and_angle(h_axis, self.angle_v)
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
        result = Matrix4f()
        for i in range(4):
            for j in range(4):
                result.values[4*i+j] = sum(self.values[4 * i + k] * other.values[4 * k + j] for k in range(4))
        return result

    def __repr__(self):
        return "[%3.2f %3.2f %3.2f %3.2f\n %3.2f %3.2f %3.2f %3.2f\n %3.2f %3.2f %3.2f %3.2f\n %3.2f %3.2f %3.2f %3.2f]" % tuple(self.values)

    def ctypes(self):
        return (c_float * 16)(*self.values)


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
        if isinstance(other, float):
            return Vector3f(other * self.x, other * self.y, other * self.z)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"Vector3f(x={self.x:2.2f}, y={self.y:2.2f}, z={self.z:2.2f})"

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
        scale_matrix = to_scale_matrix(self.scale)
        rotation_matrix = to_rotation_matrix(self.rotate)
        translation_matrix = to_translation_matrix(self.world_pos)

        camera_translation_matrix = to_translation_matrix(-1.0 * self.camera.pos)
        camera_rotation_matrix = self.camera.to_camera_transform_matrix()

        projection_matrix = to_perspective_projection_matrix(self.projection)
        return (projection_matrix * 
                camera_rotation_matrix * camera_translation_matrix * 
                translation_matrix * rotation_matrix * scale_matrix)


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


class GameManager:

    def __init__(self, camera, texture, mouse_trap):
        self.vao = None
        self.vbo = None
        self.ibo = None
        self.camera = camera
        self.mouse_trap = mouse_trap
        self.t = 0
        self.texture = texture

        self.technique = LightingTechnique()
        self.technique.enable()
        self.technique.set_texture_unit(0)

        self.directional_light = DirectionalLight(Vector3f(1.0, 1.0, 1.0), 0.5)

        self.fps = FPSCounter()

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT)
        self.t += 1.0
        pipeline = Pipeline(self.camera)
        pipeline.set_rotation(0.0, self.t, 0.0)
        pipeline.set_pos(0.0, 0.0, 3.0)
        pipeline.set_perspective_projection(
            PerspectiveProjection(30.0,
                                  glutGet(GLUT_WINDOW_WIDTH),
                                  glutGet(GLUT_WINDOW_HEIGHT),
                                  1.0,
                                  1000.0))
        
        self.technique.set_wvp(byref(pipeline.bake().ctypes()))
        self.technique.set_directional_light(self.directional_light)

        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # Doing sizeof here is a bit hacky - would be good to set up the
        # vertex class a bit more
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(c_float), None)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(c_float), c_void_p(12))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)

        self.texture.bind(GL_TEXTURE0)

        glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, None)

        glDisableVertexAttribArray(0)

        glutSwapBuffers()

        self.fps.frame()

    def keyboard(self, code, x, y):
        if code == b'q':
            sys.exit(0)
        if code == b'a':
            new = self.directional_light.ambient_intensity - 0.1
            self.directional_light = self.directional_light._replace(ambient_intensity=new)
        if code == b'd':
            new = self.directional_light.ambient_intensity + 0.1
            self.directional_light = self.directional_light._replace(ambient_intensity=new)


    def mouse(self, x, y):
        self.mouse_trap.mouse(x, y)

    def createBuffers(self):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        vertices = [
                Vertex(Vector3f(-1, -1, 0.5773), (0.0, 0.0)),
                Vertex(Vector3f(0, -1, -1.15475), (0.5, 0.0)),
                Vertex(Vector3f(1, -1, 0.5773), (1.0, 0.0)),
                Vertex(Vector3f(0, 1, 0), (0.5, 1.0))]

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, verticies_to_ctype(vertices),
                     GL_STATIC_DRAW)

        indices = (c_int * 16)(0, 3, 1, 1, 3, 2, 2, 3, 0, 0, 1, 2)
        self.ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices, GL_STATIC_DRAW)


Shader = namedtuple('Shader', ['file', 'type'])


def _compile_shader(shader):
    with open(shader.file) as f:
        shader_text = f.read()
    return ogl_shaders.compileShader(shader_text, shader.type)


class Technique:

    def __init__(self, shaders):
        compiled_shaders = [_compile_shader(shader) for shader in shaders]
        self.shader_program = ogl_shaders.compileProgram(*compiled_shaders)
        for shader in compiled_shaders:
            glDeleteShader(shader)

    def enable(self):
        glUseProgram(self.shader_program)

    def get_uniform_location(self, name):
        return glGetUniformLocation(self.shader_program, name)


DirectionalLight = namedtuple('DirectionalLight', ['colour', 'ambient_intensity'])


class LightingTechnique(Technique):

    def __init__(self):
        super().__init__([Shader('shader.vs', GL_VERTEX_SHADER),
                          Shader('shader.fs', GL_FRAGMENT_SHADER)])
        self.wvp_location = self.get_uniform_location('gWVP')
        self.sampler_location = self.get_uniform_location('gSampler')
        self.dir_light_color_location = self.get_uniform_location('gDirectionalLight.Color')
        self.dir_light_ambient_intensity = self.get_uniform_location('gDirectionalLight.AmbientIntensity')

    def set_wvp(self, wvp):
        glUniformMatrix4fv(self.wvp_location, 1, GL_TRUE, wvp)

    def set_texture_unit(self, texture_unit):
        glUniform1i(self.sampler_location, texture_unit)

    def set_directional_light(self, directional_light):
        glUniform3f(self.dir_light_color_location, directional_light.colour.x, directional_light.colour.y, directional_light.colour.z)
        glUniform1f(self.dir_light_ambient_intensity, directional_light.ambient_intensity)


def glut_init():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)


def glut_create_window(width=1920, height=1820, is_full_screen=False):
    if is_full_screen:
        glutGameModeString("%d:%d:32@60" % (width, height))
        glutEnterGameMode()
    else:
        glutInitWindowSize(width, height)
        glutCreateWindow("Python OpenGL playground")


def init_callbacks(game_manager):
    glutDisplayFunc(game_manager.render)
    glutIdleFunc(game_manager.render)
    glutKeyboardFunc(game_manager.keyboard)
    glutSpecialFunc(game_manager.keyboard)
    glutPassiveMotionFunc(game_manager.mouse)


def main():
    glut_init()
    glut_create_window()

    print("GL version: %s" % glGetString(GL_VERSION))

    glClearColor(0.0, 0.0, 0.0, 0.0)
    glFrontFace(GL_CW)
    glCullFace(GL_BACK)
    glEnable(GL_CULL_FACE)

    texture = Texture(GL_TEXTURE_2D, "test.png")

    camera = Camera(target=Vector3f.FORWARD, pos=Vector3f.ORIGIN)
    mouse_trap = MouseTrap(camera)
    game_manager = GameManager(camera, texture, mouse_trap)
    game_manager.createBuffers()


    init_callbacks(game_manager)
    glutMainLoop()


if __name__ == "__main__":
    main()
