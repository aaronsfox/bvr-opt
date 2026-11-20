"""Camera models: MayaCam 2.0 and MayaCam 1.0 calibrations, both held as an OpenCV pinhole camera.

MayaCam 2.0 files (written by XMALab) hold an OpenCV pinhole camera:
    x_cam = R @ X_world + t,   col = fx * x/z + cx,   row = fy * y/z + cy
with (col, row) in pixels, origin at the top-left pixel *centre* of the image. Autoscoper's
loader (Camera::loadMayaCam2) converts these to its own bottom-up frame; the geometry is the same.

MayaCam 1.0 files (5 lines of 3 comma-separated values) are converted to the same pinhole camera exactly as
Autoscoper interprets them. Autoscoper (libautoscoper/src/Camera.cpp) turns both formats into one internal
camera: a coordinate frame (rows = camera axes in world, translation = X-ray source position) and a viewport
computed by calculateViewport(cx', cy', fx, fy) with cy' measured bottom-up. Reading loadMayaCam2 gives:
    frame rows = diag(1, -1, -1) @ R,   source = -R.T @ t,   cx' = cx,   cy' = (H - 1) - cy.
loadMayaCam1 builds the frame with CoordFrame::from_xyzypr(x, y, z, rz, ry, rx) from lines 1-2 and calls
calculateViewport(u0 - 1, v0 - 1, |Z|, |Z|) from line 4. Inverting the MayaCam 2.0 relations therefore gives
the equivalent pinhole camera:
    R = diag(1, -1, -1) @ frame rows,   t = -R @ source,   fx = fy = |Z|,   cx = u0 - 1,   cy = H - v0.
(Autoscoper's image-plane placement, which differs between the formats, is display only.) Units are those of
the file, as in Autoscoper.

Throughout src, image coordinates are *continuous* pixel coordinates with pixel i spanning
[i, i+1] (centre i + 0.5), matching torch.grid_sample(align_corners=False). Hence the +0.5 below.
"""
import numpy as np

F = np.diag([1.0, -1.0, -1.0])     # OpenCV camera axes <-> Autoscoper internal camera axes (looks down -z)


class PinholeCamera:
    """OpenCV pinhole camera: W, H (pixels), R (3x3), t (3), fx, fy, cx, cy."""
    format = None

    def _finish(self):
        assert abs(np.linalg.det(self.R) - 1) < 1e-6, 'rotation is not proper'
        self.centre = -self.R.T @ self.t          # X-ray source position in world (mm)

    def to_cam(self, P):
        return np.atleast_2d(P) @ self.R.T + self.t

    def project(self, P):
        """World points (N,3) -> continuous pixel coordinates (N,2) = (col, row), row 0 at the top."""
        pc = self.to_cam(P)
        col = self.fx * pc[:, 0] / pc[:, 2] + self.cx + 0.5
        row = self.fy * pc[:, 1] / pc[:, 2] + self.cy + 0.5
        return np.c_[col, row]

    def ray_dirs_cam(self, col, row):
        """Unnormalised camera-frame ray directions for continuous pixel coordinates (arrays)."""
        return np.stack([(col - 0.5 - self.cx) / self.fx, (row - 0.5 - self.cy) / self.fy,
                         np.ones_like(col)], -1)


class MayaCam2(PinholeCamera):
    format = 'MayaCam 2.0'

    def __init__(self, path):
        lines = [l.strip() for l in open(path)]
        if 'image size' not in lines or 'camera matrix' not in lines:
            raise ValueError(f'{path}: not a MayaCam 2.0 calibration')
        def block(name, n):
            i = lines.index(name)
            return [[float(v) for v in lines[i + 1 + k].split(',')] for k in range(n)]
        (W, H), = block('image size', 1)
        self.W, self.H = int(W), int(H)
        K = np.array(block('camera matrix', 3))
        self.R = np.array(block('rotation', 3))
        self.t = np.array([r[0] for r in block('translation', 3)])
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        self._finish()


def _axis_rotation(axis, angle_deg):
    """Autoscoper CoordFrame::rotate's matrix R (row-major), applied as M <- M @ R."""
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    oc = 1.0 - c
    return np.array([[oc * x * x + c, oc * x * y + s * z, oc * x * z - s * y],
                     [oc * x * y - s * z, oc * y * y + c, oc * y * z + s * x],
                     [oc * x * z + s * y, oc * y * z - s * x, oc * z * z + c]])


def frame_from_xyzypr(yaw, pitch, roll):
    """Rows of Autoscoper CoordFrame::from_xyzypr's rotation: start at identity, rotate about the current
    z axis (row 3) by yaw, then the current y axis (row 2) by pitch, then the current x axis (row 1) by roll."""
    M = np.eye(3)
    for row, angle in ((2, yaw), (1, pitch), (0, roll)):
        M = M @ _axis_rotation(M[row].copy(), angle)
    return M


class MayaCam1(PinholeCamera):
    """MayaCam 1.0: line 1 source position; line 2 rotations about x, y, z (deg; applied z, y, x);
    line 3 image-plane position (unused by Autoscoper's viewport); line 4 u0, v0, Z (1-based, bottom-up
    principal point and focal length in pixels); line 5 scale, width, height (0 = 1024)."""
    format = 'MayaCam 1.0'

    def __init__(self, path):
        rows = []
        for line in open(path):
            if not line.strip():
                continue
            vals = [v for v in line.strip().split(',') if v.strip() != '']
            try:
                rows.append([float(v) for v in vals[:3]])
            except ValueError:
                raise ValueError(f'{path}: not a MayaCam 1.0 calibration (line {len(rows) + 1}: {line.strip()!r})')
            if len(rows) == 5:
                break
        if len(rows) != 5 or any(len(r) != 3 for r in rows):
            raise ValueError(f'{path}: not a MayaCam 1.0 calibration (expected 5 lines of 3 comma-separated values)')
        source, (rx, ry, rz), _, (u0, v0, Z), (_, W, H) = (np.array(r) for r in rows)
        self.W, self.H = int(W) or 1024, int(H) or 1024
        frame = frame_from_xyzypr(rz, ry, rx)
        self.R = F @ frame
        self.t = -self.R @ source
        self.fx = self.fy = abs(Z)
        self.cx, self.cy = u0 - 1.0, self.H - v0
        self._finish()


def load_camera(path):
    """MayaCam 2.0 or 1.0, chosen as Autoscoper does: 2.0 if the first line is 'image size'."""
    with open(path) as fh:
        first = fh.readline().strip()
    if first == 'image size':
        return MayaCam2(path)
    try:
        return MayaCam1(path)
    except ValueError as e:
        raise ValueError(f'{path}: not a MayaCam 2.0 or 1.0 calibration ({e})') from None
