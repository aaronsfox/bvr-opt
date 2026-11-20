"""Loading an Autoscoper trial folder.

Welte layout (see utilities/organise_welte2022.py):
    <trial>/<trial>.cfg, Calibration/*.txt (MayaCam 2.0; MayaCam 1.0 also read), Models/PV/*.tif, RadiographImages/cam1|cam2,
    Reference/<trial>_{cal,tal,tib}.tra, xParameters/control_settings.vie
Any other Autoscoper Version 1.1 .cfg also loads with Trial(folder, cfg=path): paths may use Windows
separators, 8- or 16-bit radiographs are scaled to [0, 1] by their dtype, VolumeFlip is applied, and the
renderer settings (.vie) are optional (see Trial.__init__).
"""
import glob
import math
import os
import warnings

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

from .cameras import load_camera
from .drr import Bone
from .device import dev
from .filters import parse_vie


def read_tra(path, order='column'):
    """Autoscoper .tra: one 4x4 per line (16 comma-separated values; NaN rows = untracked).
    Welte reference files are column-major (translation in elements 13-15). order: 'column', 'row', or 'auto'
    (detected by tra_order)."""
    if order == 'auto':
        order = tra_order(path)
    A = np.genfromtxt(path, delimiter=',').reshape(-1, 4, 4)
    return A.transpose(0, 2, 1) if order == 'column' else A


def tra_order(path, tol=1e-6):
    """Detect whether an Autoscoper .tra stores each 4x4 column-major ('column': translation in elements 13-15,
    as Welte's files and Autoscoper's default export) or row-major ('row': translation in elements 4, 8, 12).
    Decided from the tracked lines: the bottom row (0, 0, 0, 1) sits in elements 4, 8, 12, 16 when column-major
    and in elements 13-16 when row-major. Lines with zero translation fit both and are skipped; if every line
    is like that, 'column' is returned."""
    A = np.genfromtxt(path, delimiter=',').reshape(-1, 16)
    A = A[np.isfinite(A).all(1)]
    if not len(A):
        raise ValueError(f'{path}: no tracked frames')
    col = (np.abs(A[:, [3, 7, 11]]).max(1) < tol) & (np.abs(A[:, 15] - 1) < tol)
    row = (np.abs(A[:, [12, 13, 14]]).max(1) < tol) & (np.abs(A[:, 15] - 1) < tol)
    if (~(col | row)).any():
        raise ValueError(f'{path}: {int((~(col | row)).sum())} tracked lines are not rigid 4x4 poses in either '
                         'row- or column-major order (expected a bottom row of 0, 0, 0, 1)')
    if (col & ~row).any() and (row & ~col).any():
        raise ValueError(f'{path}: mixes row-major and column-major lines')
    return 'row' if (row & ~col).any() else 'column'


def write_tra(path, Ms, order='column'):
    """Write poses (n, 4, 4) as an Autoscoper .tra (one comma-separated 4x4 per line). Untracked frames are
    written as `NaN`, the exact spelling Autoscoper writes and tests for when it reads a file."""
    A = (Ms.transpose(0, 2, 1) if order == 'column' else Ms).reshape(len(Ms), 16)
    with open(path, 'w') as fh:
        for row in A:
            fh.write(','.join('NaN' if np.isnan(v) else f'{v:.6f}' for v in row) + '\n')


class Trial:
    BONES = {'cal': 'calcaneus', 'tal': 'talus', 'tib': 'tibia'}

    # Renderer settings used when a trial has no .vie file: the Welte 2022 values (Autoscoper GUI units)
    DEFAULT_RENDERER = {'SampleDistance': 0.62, 'RayIntensity': 0.49, 'Cutoff': 0.0}

    def __init__(self, folder, cfg=None, vie=None):
        """folder: trial folder (paths in the .cfg are relative to it). cfg: the .cfg path (default
        <folder>/<folder name>.cfg); the trial is named after the .cfg file. vie: Autoscoper renderer settings
        (default xParameters/control_settings.vie, else the only .vie in xParameters/ or the trial folder,
        else DEFAULT_RENDERER with a warning)."""
        self.folder = folder
        cfg = cfg or os.path.join(folder, os.path.basename(os.path.normpath(folder)) + '.cfg')
        self.name = os.path.splitext(os.path.basename(cfg))[0]
        lines = [l.split(None, 1) for l in open(cfg) if l.strip() and not l.lstrip().startswith('#')]
        cams, roots, vols, vox, flips = [], [], [], [], []
        for key, val in [(c[0], c[1].strip() if len(c) > 1 else '') for c in lines]:
            val = val.replace('\\', '/')                   # Windows separators (AutoscoperM writes them)
            if key == 'mayaCam_csv': cams.append(val)
            elif key == 'CameraRootDir': roots.append(val)
            elif key == 'VolumeFile': vols.append(val)
            elif key == 'VoxelSize': vox.append(tuple(float(x) for x in val.split()))
            elif key == 'VolumeFlip': flips.append(tuple(int(float(x)) for x in val.split()))
            elif key == 'RenderResolution': self.render_resolution = tuple(int(x) for x in val.split())
        if len(flips) < len(vols): flips += [(0, 0, 0)] * (len(vols) - len(flips))
        self.cams = [load_camera(os.path.join(folder, c)) for c in cams]   # MayaCam 2.0 or 1.0
        self.image_files = [sorted(f for e in ('*.tif', '*.tiff', '*.TIF', '*.TIFF')
                                   for f in glob.glob(os.path.join(folder, r, e))) for r in roots]
        if not all(self.image_files) or len({len(f) for f in self.image_files}) != 1:
            raise ValueError(f'{self.name}: expected the same number (> 0) of .tif images per camera, found '
                             f'{[len(f) for f in self.image_files]} in {roots}')
        self.n_frames = len(self.image_files[0])
        for i, cam in enumerate(self.cams):
            h, w = tifffile.TiffFile(self.image_files[i][0]).pages[0].shape[:2]
            if (h, w) != (cam.H, cam.W):
                raise ValueError(f'{self.name}: camera {i + 1} images are {w} x {h} px but its {cam.format} calibration '
                                 f'says {cam.W} x {cam.H} (MayaCam 1.0 assumes 1024 x 1024 when line 5 gives 0)')
        self.vie_path = vie or self._find_vie(folder)
        if self.vie_path:
            self.vie = parse_vie(self.vie_path)
        else:
            warnings.warn(f'{self.name}: no .vie renderer settings found; using {self.DEFAULT_RENDERER}')
            self.vie = {'rad': [], 'drr': [], 'renderer': dict(self.DEFAULT_RENDERER)}
        r = self.vie['renderer']
        self.sample_distance = math.exp(7 * r['SampleDistance'] - 5)   # mm
        self.ray_intensity = math.exp(15 * r['RayIntensity'] - 5)
        self.cutoff_frac = r['Cutoff']
        self.bones, self.ref = {}, {}
        for v, s, fl in zip(vols, vox, flips):
            key = os.path.splitext(os.path.basename(v))[0].split('_PV')[-1]
            self.bones[key] = Bone(os.path.join(folder, v), s, name=key, cutoff_frac=self.cutoff_frac, flip=fl)
            tra = os.path.join(folder, 'Reference', f'{self.name}_{key}.tra')
            if os.path.exists(tra):
                self.ref[key] = read_tra(tra)
        self._img = {}

    @staticmethod
    def _find_vie(folder):
        p = os.path.join(folder, 'xParameters', 'control_settings.vie')
        if os.path.exists(p):
            return p
        for d in (os.path.join(folder, 'xParameters'), folder):
            found = sorted(glob.glob(os.path.join(d, '*.vie')))
            if len(found) == 1:
                return found[0]
            if len(found) > 1:
                raise ValueError(f'several .vie files in {d}: pass the one to use')
        return None

    def raw_image(self, cam, frame):
        """Radiograph as float32 numpy in [0, 1], not cached. Integer images are scaled by their dtype's
        maximum (8-bit / 255, 16-bit / 65535); float images are assumed to be in [0, 1] already; colour
        images are averaged over their channels."""
        im = tifffile.imread(self.image_files[cam][frame])
        scale = float(np.iinfo(im.dtype).max) if np.issubdtype(im.dtype, np.integer) else 1.0
        im = im.astype(np.float32) / scale
        if im.ndim == 3:
            im = im[..., :3].mean(-1) if im.shape[-1] in (3, 4) else im.mean(0)
        return im

    def image(self, cam, frame, size=None):
        """Radiograph (float32 tensor in [0, 1], row 0 at top). size: bicubic-downsample to (size, size)
        with antialiasing (Williams 2026 used MATLAB imresize, which antialiases by default)."""
        k = (cam, frame, size)
        if k not in self._img:
            im = dev(torch.from_numpy(self.raw_image(cam, frame)))
            if size:
                im = F.interpolate(im[None, None], size=(size, size), mode='bicubic', antialias=True,
                                   align_corners=False)[0, 0].clamp(0, 1)
            self._img[k] = im
        return self._img[k]

    def ref_pose(self, bone, frame):
        return self.ref[bone][frame]
