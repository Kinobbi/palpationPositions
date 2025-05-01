import sys
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QSlider, QLabel, QOpenGLWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QSurfaceFormat
from OpenGL.GL import *
from OpenGL.GLU import *

# enable default multisampling for antialiasing
fmt = QSurfaceFormat()
fmt.setSamples(4)
QSurfaceFormat.setDefaultFormat(fmt)


# Quaternion utilities

def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    )


def quat_from_axis_angle(axis, angle):
    x, y, z = axis
    norm = math.sqrt(x*x + y*y + z*z)
    if norm == 0:
        return (1.0, 0.0, 0.0, 0.0)
    x, y, z = x/norm, y/norm, z/norm
    s = math.sin(angle/2)
    return (math.cos(angle/2), x*s, y*s, z*s)


def quat_to_matrix(q):
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    return [
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w), 0],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w), 0],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y), 0],
        [0, 0, 0, 1]
    ]


def quat_to_euler(q):
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    sinr = 2*(w*x + y*z)
    cosr = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr, cosr)
    sinp = 2*(w*y - z*x)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny = 2*(w*z + x*y)
    cosy = 1 - 2*(y*y + z*z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw

# Rotate a vector by quaternion q
def rotate_vector(v, q):
    w, x, y, z = q
    vx, vy, vz = v
    iw = -x*vx - y*vy - z*vz
    ix =  w*vx + y*vz - z*vy
    iy =  w*vy + z*vx - x*vz
    iz =  w*vz + x*vy - y*vx
    return (
        ix*w - iw*x - iy*z + iz*y,
        iy*w - iw*y - iz*x + ix*z,
        iz*w - iw*z - ix*y + iy*x
    )

class ModelWidget(QOpenGLWidget):
    def __init__(self, baby_path, pelvis_path, parent=None):
        super().__init__(parent)
        self.orientation = (1.0, 0.0, 0.0, 0.0)
        self.baby_vertices, self.baby_faces = [], []
        self.pelvis_vertices, self.pelvis_faces = [], []
        self.load_model(baby_path, self.baby_vertices, self.baby_faces)
        self.load_model(pelvis_path, self.pelvis_vertices, self.pelvis_faces)
        self.orientation = (1.0, 0.0, 0.0, 0.0)
        self.baby_vertices, self.baby_faces = [], []
        self.pelvis_vertices, self.pelvis_faces = [], []
        self.load_model(baby_path, self.baby_vertices, self.baby_faces)
        self.load_model(pelvis_path, self.pelvis_vertices, self.pelvis_faces)

    def load_model(self, path, verts, faces):
        verts.clear()
        faces.clear()
        with open(path, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == 'v':
                    x, y, z = map(float, parts[1:4])
                    verts.append((x, y, z))
                elif parts[0] == 'f':
                    idx = [int(p.split('/')[0]) - 1 for p in parts[1:4]]
                    faces.append(tuple(idx))
        self.update()

    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        # enable antialiasing for lines
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w/h if h else 1, 0.1, 100)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0, 0, -6)
        # draw baby with rotation
        glPushMatrix()
        m = quat_to_matrix(self.orientation)
        cm = [m[j][i] for i in range(4) for j in range(4)]
        glMultMatrixf(cm)
        self.draw_model(self.baby_vertices, self.baby_faces, color_by_position=True)
        self.draw_axis_halos()
        glPopMatrix()
        # draw pelvis without rotation
        self.draw_model(self.pelvis_vertices, self.pelvis_faces, color_by_position=True)

    def draw_model(self, verts, faces, color_by_position=True):
        glBegin(GL_TRIANGLES)
        scale = max(max(abs(x), abs(y), abs(z)) for x, y, z in verts) or 1.0
        for face in faces:
            for i in face:
                x, y, z = verts[i]
                if color_by_position:
                    glColor3f(abs(x)/scale, abs(y)/scale, abs(z)/scale)
                glVertex3f(x, y, z)
        glEnd()

    def draw_axis_halos(self):
        r, steps = 2.0, 64
        glLineWidth(2.0)
        for col, func in [
            ((1,0,0), lambda t: (0, math.cos(t)*r, math.sin(t)*r)),
            ((0,1,0), lambda t: (math.cos(t)*r, 0, math.sin(t)*r)),
            ((0,0,1), lambda t: (math.cos(t)*r, math.sin(t)*r, 0)),
        ]:
            glColor3f(*col)
            glBegin(GL_LINE_LOOP)
            for i in range(steps):
                t = 2*math.pi*i/steps
                glVertex3f(*func(t))
            glEnd()
        glLineWidth(1.0)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Single Viewport: Baby + Pelvis")
        cw = QWidget()
        self.setCentralWidget(cw)
        hl = QHBoxLayout(cw)
        self.view = ModelWidget(
            r"C:\Users\Kimon\Documents\projects\pyGraphs\baby.obj",
            r"C:\Users\Kimon\Documents\projects\pyGraphs\pelvis.obj"
        )
        hl.addWidget(self.view, 1)
        ctrl = QWidget()
        vl = QVBoxLayout(ctrl)
        self.last_x = self.last_y = self.last_z = 0.0
        for ax in ('X','Y','Z'):
            lbl = QLabel(f"{ax}: 0.0°")
            sld = QSlider(Qt.Horizontal)
            sld.setRange(0, 3600)
            sld.valueChanged.connect(getattr(self, f"{ax.lower()}_changed"))
            vl.addWidget(lbl)
            vl.addWidget(sld)
            setattr(self, f"{ax.lower()}_label", lbl)
            setattr(self, f"{ax.lower()}_slider", sld)
        self.pos_label = QLabel("Position: N/A")
        vl.addWidget(self.pos_label)
        vl.addStretch()
        hl.addWidget(ctrl)
        self.resize(800, 600)

    def x_changed(self, val):
        new = val / 10.0
        delta = math.radians(new - self.last_x)
        self.last_x = new
        axis = rotate_vector((1,0,0), self.view.orientation)
        q = quat_from_axis_angle(axis, delta)
        self.view.orientation = quat_mul(q, self.view.orientation)
        self.update_sliders()

    def y_changed(self, val):
        new = val / 10.0
        delta = math.radians(new - self.last_y)
        self.last_y = new
        axis = rotate_vector((0,1,0), self.view.orientation)
        q = quat_from_axis_angle(axis, delta)
        self.view.orientation = quat_mul(q, self.view.orientation)
        self.update_sliders()

    def z_changed(self, val):
        new = val / 10.0
        delta = math.radians(new - self.last_z)
        self.last_z = new
        axis = rotate_vector((0,0,1), self.view.orientation)
        q = quat_from_axis_angle(axis, delta)
        self.view.orientation = quat_mul(q, self.view.orientation)
        self.update_sliders()

    def update_sliders(self):
        rx, ry, rz = quat_to_euler(self.view.orientation)
        for ax, ang in zip(('x','y','z'), (rx, ry, rz)):
            deg = math.degrees(ang) % 360
            lbl = getattr(self, f"{ax}_label")
            sld = getattr(self, f"{ax}_slider")
            sld.blockSignals(True)
            sld.setValue(int(deg * 10))
            sld.blockSignals(False)
            lbl.setText(f"{ax.upper()}: {deg:.1f}°")
        pitch = math.degrees(ry) % 360
        prefix = 'S' if 90 < pitch < 270 else 'O'
        yaw = math.degrees(rz) % 360
        if 315 <= yaw or yaw < 45:
            sect = 'A'
        elif 45 <= yaw < 135:
            sect = 'LT'
        elif 135 <= yaw < 225:
            sect = 'P'
        else:
            sect = 'RT'
        if sect == 'A':
            code = prefix + 'A'
        elif sect == 'P':
            code = prefix + 'P'
        elif sect == 'LT':
            code = 'L' + prefix + 'T'
        else:
            code = 'R' + prefix + 'T'
        self.pos_label.setText(code)
        self.view.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
