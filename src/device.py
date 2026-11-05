"""Compute device. Set BVR_DEVICE=cuda|cpu (default: cuda when available).

On CUDA, torch's default device is set so that tensors created by torch factory functions land on the
GPU; arrays coming from numpy are moved with `dev`, and tensors going back to numpy use `to_np`.
"""
import os

import torch

_req = os.environ.get('BVR_DEVICE', 'auto').lower()
DEVICE = torch.device('cuda' if (_req == 'auto' and torch.cuda.is_available()) else ('cpu' if _req == 'auto' else _req))
if DEVICE.type == 'cuda':
    torch.set_default_device(DEVICE)


def dev(x):
    return x.to(DEVICE)


def to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x
