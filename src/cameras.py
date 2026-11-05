"""MayaCam 2.0 camera model.

MayaCam 2.0 files (written by XMALab) hold an OpenCV pinhole camera:
    x_cam = R @ X_world + t,   col = fx * x/z + cx,   row = fy * y/z + cy
with (col, row) in pixels, origin at the top-left pixel *centre* of the image. Autoscoper's
loader (Camera::loadMayaCam2) converts these to its own bottom-up frame; the geometry is the same.

Throughout src, image coordinates are *continuous* pixel coordinates with pixel i spanning
[i, i+1] (centre i + 0.5), matching torch.grid_sample(align_corners=False). Hence the +0.5 below.
"""
import numpy as np


class MayaCam2:
    def __init__(self, path):
        lines = [l.strip() for l in open(path)]
        def block(name, n):
            i = lines.index(name)
            return [[float(v) for v in lines[i + 1 + k].split(',')] for k in range(n)]
        (W, H), = block('image size', 1)
        self.W, self.H = int(W), int(H)
        K = np.array(block('camera matrix', 3))
        self.R = np.array(block('rotation', 3))
        self.t = np.array([r[0] for r in block('translation', 3)])
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
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
