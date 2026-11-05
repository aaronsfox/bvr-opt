"""Port of Autoscoper's MayaCam 1.0 camera model (Camera.cpp / CoordFrame.cpp)."""
import numpy as np

def _rotate(M, axis, angle_deg):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg)); oc = 1 - c
    x, y, z = a
    R = np.array([[oc*x*x + c,   oc*x*y + s*z, oc*x*z - s*y],
                  [oc*x*y - s*z, oc*y*y + c,   oc*y*z + s*x],
                  [oc*x*z + s*y, oc*y*z - s*x, oc*z*z + c]])
    return M @ R

def frame_from_xyzypr(p):
    """Returns (R, t); rows of R are the frame's local x,y,z axes in world coords."""
    M = np.eye(3)
    M = _rotate(M, M[2], p[3]); M = _rotate(M, M[1], p[4]); M = _rotate(M, M[0], p[5])
    return M, np.array(p[:3], float)

class MayaCam1:
    def __init__(self, path):
        v = [[float(x) for x in l.split(',')[:3]] for l in open(path) if l.strip()]
        pos, rot, _, (u0, v0, z), (_, W, H) = v
        self.R, self.t = frame_from_xyzypr([*pos, rot[2], rot[1], rot[0]])
        self.cx, self.cy, self.f = u0 - 1, v0 - 1, abs(z)
        self.W, self.H = int(W), int(H)
    @property
    def look(self):            # camera looks down its local -z
        return -self.R[2]
    def to_cam(self, P):
        return (np.atleast_2d(P) - self.t) @ self.R.T
    def project(self, P):
        """World points -> pixel (col, row) with row 0 = top of the TIFF."""
        pc = self.to_cam(P)
        col = self.cx - self.f * pc[:, 0] / pc[:, 2]
        row = self.H - self.cy + self.f * pc[:, 1] / pc[:, 2]
        return np.c_[col, row]
