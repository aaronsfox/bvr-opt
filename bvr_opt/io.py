"""Loading a Welte-style Autoscoper trial folder.

Layout (see scripts/organise_welte2022.py):
    <trial>/<trial>.cfg, Calibration/*.txt (MayaCam 2.0), Models/PV/*.tif, RadiographImages/cam1|cam2,
    Reference/<trial>_{cal,tal,tib}.tra, xParameters/control_settings.vie
"""
import glob
import math
import os

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

from .cameras import MayaCam2
from .drr import Bone
from .filters import parse_vie


def read_tra(path, order='column'):
    """Autoscoper .tra: one 4x4 per line (16 comma-separated values; NaN rows = untracked).
    Welte reference files are column-major (translation in elements 13-15)."""
    A = np.genfromtxt(path, delimiter=',').reshape(-1, 4, 4)
    return A.transpose(0, 2, 1) if order == 'column' else A


def write_tra(path, Ms, order='column'):
    A = Ms.transpose(0, 2, 1) if order == 'column' else Ms
    np.savetxt(path, A.reshape(len(A), 16), delimiter=',', fmt='%.6f')


class Trial:
    BONES = {'cal': 'calcaneus', 'tal': 'talus', 'tib': 'tibia'}

    def __init__(self, folder):
        self.folder = folder
        self.name = os.path.basename(os.path.normpath(folder))
        cfg = [l.split(None, 1) for l in open(os.path.join(folder, self.name + '.cfg')) if l.strip()]
        cams, roots, vols, vox = [], [], [], []
        for key, val in [(c[0], c[1].strip() if len(c) > 1 else '') for c in cfg]:
            if key == 'mayaCam_csv': cams.append(val)
            elif key == 'CameraRootDir': roots.append(val)
            elif key == 'VolumeFile': vols.append(val)
            elif key == 'VoxelSize': vox.append(tuple(float(x) for x in val.split()))
            elif key == 'RenderResolution': self.render_resolution = tuple(int(x) for x in val.split())
        self.cams = [MayaCam2(os.path.join(folder, c)) for c in cams]
        self.image_files = [sorted(glob.glob(os.path.join(folder, r, '*.tif'))) for r in roots]
        self.n_frames = len(self.image_files[0])
        self.vie = parse_vie(os.path.join(folder, 'xParameters', 'control_settings.vie'))
        r = self.vie['renderer']
        self.sample_distance = math.exp(7 * r['SampleDistance'] - 5)   # mm
        self.ray_intensity = math.exp(15 * r['RayIntensity'] - 5)
        self.cutoff_frac = r['Cutoff']
        self.bones, self.ref = {}, {}
        for v, s in zip(vols, vox):
            key = os.path.splitext(os.path.basename(v))[0].split('_PV')[-1]
            self.bones[key] = Bone(os.path.join(folder, v), s, name=key, cutoff_frac=self.cutoff_frac)
            tra = os.path.join(folder, 'Reference', f'{self.name}_{key}.tra')
            if os.path.exists(tra):
                self.ref[key] = read_tra(tra)
        self._img = {}

    def raw_image(self, cam, frame):
        """Radiograph as float32 numpy in [0, 1], not cached."""
        return tifffile.imread(self.image_files[cam][frame]).astype(np.float32) / 255.0

    def image(self, cam, frame, size=None):
        """Radiograph (float32 tensor in [0, 1], row 0 at top). size: bicubic-downsample to (size, size)
        with antialiasing (Williams 2026 used MATLAB imresize, which antialiases by default)."""
        k = (cam, frame, size)
        if k not in self._img:
            im = torch.from_numpy(self.raw_image(cam, frame))
            if size:
                im = F.interpolate(im[None, None], size=(size, size), mode='bicubic', antialias=True,
                                   align_corners=False)[0, 0].clamp(0, 1)
            self._img[k] = im
        return self._img[k]

    def ref_pose(self, bone, frame):
        return self.ref[bone][frame]
