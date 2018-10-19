from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLUT import *

from collections import *
import sys, time
from math import *
from ctypes import *

PerspectiveProjection = namedtuple('PerspectiveProjection',
                                   ['fov', 'width', 'height', 'z_near', 'z_far'])

class Camera:

    def __init__(self, pos=None, target=None, up=None):
        self.pos = pos if pos else Vector3f(0.0, 0.0, 0.0)

        self.target = target if target else Vector3f(0.0, 0.0, 1.0)
        self.target.normalize()
        self.up = up if up else Vector3f(0.0, 1.0, 0.0)
        self.up.normalize()

    def keyboard(self, key, x, y):
        speed = 0.1
        if key == GLUT_KEY_UP:
            self.pos += self.target * speed
        if key == GLUT_KEY_DOWN:
            self.pos += self.target * -speed
        if key == GLUT_KEY_LEFT:
            left = self.target.cross(self.up)
            left.normalize()
            self.pos += left * speed
        if key == GLUT_KEY_RIGHT:
            right = self.up.cross(self.target)
            right.normalize()
            self.pos += right * speed


class Quaternion:

    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def conj(self):
        return Quaternion(-self.x, -self.y, -self.z, w)

    def rotate(self, v):
        return ((self * v) * self.conj())._to_vec()

    @classmethod
    def from_vector_and_angle(v, theta_deg):
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
        return Quaternion(x=r0*q1 + r1*q1 - r2*q3 + r3*q2,
                          y=r0*q2 + r1*q3 + r2*q0 - r3*q1,
                          z=r0*q1 - r1*q2 + r2*q1 + r3*q0,
                          w=r0*q0 - r1*q1 - r2*q2 - r3*q3)


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
        self.x = x
        self.y = y
        self.z = z

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


def to_camera_transform_matrix(target, up):
    n = target.copy()
    n.normalize()
    u = up.cross(target)
    u.normalize()
    v = n.cross(u)
    
    m = Matrix4f()
    m[0] = [u.x, u.y, u.z, 0]
    m[1] = [v.x, v.y, v.z, 0]
    m[2] = [n.x, n.y, n.z, 0]
    m[3][3] = 1.0
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
        camera_rotation_matrix = to_camera_transform_matrix(self.camera.target, self.camera.up)

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

    def __init__(self, shader_program, camera):
        self.shader_program = shader_program
        self.vao = None
        self.vbo = None
        self.ibo = None
        self.camera = camera
        self.gWorldLocation = glGetUniformLocation(shader_program, "gWorld")
        self.t = 0

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
        
        glUniformMatrix4fv(self.gWorldLocation, 1, GL_TRUE, byref(pipeline.bake().ctypes()))

        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)

        #glDrawArrays(GL_TRIANGLES, 0, 3)
        glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, None)

        glDisableVertexAttribArray(0)

        glutSwapBuffers()

        self.fps.frame()

    def createBuffers(self):
        #self.vao = glGenVertexArrays(1)
        #glBindVertexArray(self.vao)

        #vertices = (c_float * 9)(-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0)
        vertices = (c_float * 16)(-1, -1, 0.5773, 
                                   0, -1, -1.15475,
                                   1, -1, 0.5773,
                                   0, 1, 0)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)

        indices = (c_int * 16)(0, 3, 1, 1, 3, 2, 2, 3, 0, 0, 1, 2)
        #indices = (c_uint * 3)(0, 1, 2)
        self.ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices, GL_STATIC_DRAW)


def compileShaders():
    with open('shader.vs') as f:
        vertex_shader_text = f.read()
    with open('shader.fs') as f:
        fragment_shader_text = f.read()

    vertex_shader = shaders.compileShader(vertex_shader_text, GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(fragment_shader_text, GL_FRAGMENT_SHADER)
    shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
    glUseProgram(shader_program)
    return shader_program


def keyboard(code, x, y):
    if code == b'q':
        sys.exit(0)
    print(code)


def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
    #glutInitWindowSize(1024, 768)
    #glutInitWindowPosition(100, 100)
    glutCreateWindow("Tutorial 10")

    print("GL version: %s" % glGetString(GL_VERSION))

    glClearColor(0.0, 0.0, 0.0, 0.0)

    shader_program = compileShaders()

    camera = Camera()

    game_manager = GameManager(shader_program, camera)
    game_manager.createBuffers()

    glutDisplayFunc(game_manager.render)
    glutIdleFunc(game_manager.render)
    glutKeyboardFunc(keyboard)
    glutSpecialFunc(camera.keyboard)

    glutMainLoop()


if __name__ == "__main__":
    main()
