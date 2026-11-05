"""Poses, pivot-centred increments, interpolation, filtering and error measures.

Pose convention: world = R @ p_local + t, where p_local is in the CT volume's local frame
(Autoscoper's VolumeDescription: x = col*sx, y = (H-row)*sy, z = -slice*sz).
"""
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as Rot, RotationSpline, Slerp

from .device import DEVICE, to_np


# ---------------------------------------------------------------- differentiable increments
def skew(v):
    z = torch.zeros((), dtype=v.dtype)
    return torch.stack([torch.stack([z, -v[2], v[1]]),
                        torch.stack([v[2], z, -v[0]]),
                        torch.stack([-v[1], v[0], z])])


def rotvec_to_matrix(rv_deg):
    """Rodrigues; rotation vector in degrees (Autoscoper's increment convention). Differentiable."""
    rv = rv_deg * (np.pi / 180.0)
    th = torch.sqrt((rv ** 2).sum() + 1e-18)
    K = skew(rv / th)
    return torch.eye(3, dtype=rv.dtype) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


class Pose:
    """Bone pose. Increments are applied about the pivot in the bone frame, as in Autoscoper:
    new = current * V^-1 * Delta * V  (Tracker::optimize)."""

    def __init__(self, R, t, pivot):
        self.R = torch.as_tensor(np.asarray(R) if not torch.is_tensor(R) else R, dtype=torch.float64, device=DEVICE)
        self.t = torch.as_tensor(np.asarray(t) if not torch.is_tensor(t) else t, dtype=torch.float64, device=DEVICE)
        self.pivot = torch.as_tensor(pivot if torch.is_tensor(pivot) else np.asarray(pivot), dtype=torch.float64, device=DEVICE)

    @classmethod
    def from_matrix(cls, M, pivot):
        return cls(M[:3, :3], M[:3, 3], pivot)

    def apply(self, delta):
        """delta: tensor(6) = [dx, dy, dz (mm), rotation vector (deg)] -> (R, t), differentiable."""
        dR = rotvec_to_matrix(delta[3:])
        R = self.R @ dR
        t = self.t + self.R @ (self.pivot - dR @ self.pivot + delta[:3])
        return R, t

    def compose(self, delta):
        with torch.no_grad():
            R, t = self.apply(torch.as_tensor(np.asarray(delta, float), dtype=torch.float64, device=DEVICE))
        return Pose(R, t, self.pivot)

    def matrix(self):
        M = np.eye(4); M[:3, :3] = self.R.detach().cpu().numpy(); M[:3, 3] = self.t.detach().cpu().numpy()
        return M


def mat(R, t):
    M = np.eye(4); M[:3, :3] = to_np(R); M[:3, 3] = to_np(t); return M


# ---------------------------------------------------------------- interpolation between keyframes
def interpolate(Ms, key_idx, frames, pivot, kind='spline'):
    """Interpolate poses Ms[key_idx] (array (n,4,4) indexed by frame) to `frames`.

    Positions of the bone pivot and rotations are interpolated separately:
      kind='linear': linear pivot position + slerp
      kind='spline': cubic spline pivot position + cubic rotation spline (quaternion spline)
    Returns (len(frames), 4, 4)."""
    key_idx = np.asarray(key_idx); frames = np.asarray(frames)
    Rk = Rot.from_matrix(Ms[key_idx, :3, :3])
    pk = np.einsum('nij,j->ni', Ms[key_idx, :3, :3], pivot) + Ms[key_idx, :3, 3]
    if kind == 'linear':
        R = Slerp(key_idx, Rk)(frames)
        p = np.stack([np.interp(frames, key_idx, pk[:, i]) for i in range(3)], 1)
    elif kind == 'spline':
        R = RotationSpline(key_idx, Rk)(frames)
        p = CubicSpline(key_idx, pk, axis=0)(frames)
    else:
        raise ValueError(kind)
    out = np.tile(np.eye(4), (len(frames), 1, 1))
    out[:, :3, :3] = R.as_matrix()
    out[:, :3, 3] = p - np.einsum('nij,j->ni', out[:, :3, :3], pivot)
    return out


def butterworth(Ms, pivot, fs=250.0, fc=10.0, order=2):
    """Dual-pass (zero-lag) low-pass Butterworth of a pose sequence (n,4,4), as in Kessler 2019:
    pivot positions and sign-continuous quaternions filtered, quaternions renormalised."""
    b, a = butter(order, fc / (fs / 2))
    q = Rot.from_matrix(Ms[:, :3, :3]).as_quat()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    p = np.einsum('nij,j->ni', Ms[:, :3, :3], pivot) + Ms[:, :3, 3]
    qf = filtfilt(b, a, q, axis=0); qf /= np.linalg.norm(qf, axis=1, keepdims=True)
    pf = filtfilt(b, a, p, axis=0)
    out = np.tile(np.eye(4), (len(Ms), 1, 1))
    out[:, :3, :3] = Rot.from_quat(qf).as_matrix()
    out[:, :3, 3] = pf - np.einsum('nij,j->ni', out[:, :3, :3], pivot)
    return out


# ---------------------------------------------------------------- errors against the reference
def helical_error(M, G):
    """Helical-axis error of pose M relative to reference G (both 4x4, CT -> world).
    Returns (rotation angle phi in degrees, |translation along the helical axis| in mm).
    Both are invariant to the choice of origin and to expressing the difference in either frame."""
    E = np.linalg.inv(G) @ M
    rv = Rot.from_matrix(E[:3, :3]).as_rotvec()
    phi = np.linalg.norm(rv)
    if phi < 1e-9:
        return 0.0, float(np.linalg.norm(E[:3, 3]))
    return float(np.degrees(phi)), float(abs(np.dot(rv / phi, E[:3, 3])))


def pivot_error(M, G, pivot):
    """Distance (mm) between the bone pivot placed by M and by G."""
    return float(np.linalg.norm((M[:3, :3] - G[:3, :3]) @ pivot + M[:3, 3] - G[:3, 3]))


def axis_errors(M, G, axes, pivot):
    """Signed error resolved in the bone's principal axes (3x3, rows = long, mid, short axes in CT frame).
    Returns rotation (deg, small-angle rotation vector components) and translation of the pivot (mm),
    both expressed in the reference bone frame."""
    E = np.linalg.inv(G) @ M
    rv = np.degrees(Rot.from_matrix(E[:3, :3]).as_rotvec())
    dp = E[:3, :3] @ pivot + E[:3, 3] - pivot
    return axes @ rv, axes @ dp
