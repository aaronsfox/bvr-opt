"""Simulated rater error, calibrated to Welte et al. (2022).

A rater's keyframe for bone n at frame f is the reference pose with a rigid error applied in the bone's CT
frame, about the bone pivot p:

    M(f) = G(f) * D(f),   D(f) = [ R(r_f) , p - R(r_f) p + t_f ]

so the pivot is displaced by t_f and the bone is rotated by the rotation vector r_f (degrees) about it.
Each error vector (r for rotation, t for translation) has two parts, drawn per rater and bone:

    e_f = lam * ( o + a_f )
    o    ~ N(0, s_off^2 I)          constant offset over the whole trial (the rater's bias)
    a_f  = phi a_{f-1} + sqrt(1 - phi^2) * N(0, s_rand^2 I)    frame-to-frame part, AR(1), stationary
    lam  ~ LogNormal(0, between^2)  how good this rater is on this bone, relative to the typical rater

with s_off^2 = share * s^2 and s_rand^2 = (1 - share) * s^2. The scale s (per bone, for rotation and for
translation) is what the calibration fits; share, phi and between are set from Welte's figures. Rater
levels multiply s by the level (1 = trained, as in Welte).

Welte's metrics (helical rotation angle and translation along the helical axis) depend only on D, not on
the reference pose: the angle is |r| and the translation along the axis is |t . r/|r|| (the term
p - R p is perpendicular to the axis). The pivot is displaced by |t|.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def draw_track(rng, n_frames, s_rot, s_trans, share_rot, share_trans, phi, between):
    """One rater's error track for one bone: rotation vectors (deg) and translations (mm), (n_frames, 3) each."""
    lam = float(np.exp(rng.normal(0.0, between))) if between > 0 else 1.0
    out = []
    for s, share in ((s_rot, share_rot), (s_trans, share_trans)):
        o = rng.normal(0.0, np.sqrt(share) * s, 3)
        sr = np.sqrt(1.0 - share) * s
        a = np.empty((n_frames, 3)); a[0] = rng.normal(0.0, sr, 3)
        for f in range(1, n_frames):
            a[f] = phi * a[f - 1] + np.sqrt(1 - phi ** 2) * rng.normal(0.0, sr, 3)
        out.append(lam * (o + a))
    return out[0], out[1]


def helical_metrics(r, t):
    """Welte's metrics for error vectors r (deg) and t (mm): rotation angle (deg), |translation along the
    helical axis| (mm), and pivot displacement |t| (mm). r, t: (..., 3)."""
    ang = np.linalg.norm(r, axis=-1)
    axis = r / np.maximum(ang, 1e-12)[..., None]
    return ang, np.abs((t * axis).sum(-1)), np.linalg.norm(t, axis=-1)


def error_matrix(r, t, pivot):
    """4x4 error transform D for rotation vector r (deg) and pivot translation t (mm)."""
    R = Rot.from_rotvec(np.radians(np.asarray(r, float))).as_matrix()
    D = np.eye(4); D[:3, :3] = R; D[:3, 3] = pivot - R @ pivot + t
    return D
