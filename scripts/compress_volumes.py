"""Losslessly compress TIFF volumes in place (LZW), so they fit within GitHub's file size limit.

Each file is compressed to a temporary copy, read back and checked voxel-for-voxel against the
original, and only then replaces the original. Files already compressed are skipped.
Do not crop volumes: Autoscoper places the volume frame at the volume's corner, so cropping would
shift the frame and break the reference poses.

Usage:
    python scripts/compress_volumes.py datasets/Welte2022/data/SOL001A/Models/PV
    python scripts/compress_volumes.py path/to/one_volume.tif
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tifffile


def compress(path: Path, method: str = "lzw") -> None:
    with tifffile.TiffFile(path) as tif:
        current = tif.pages[0].compression.name
    if current != "NONE":
        print(f"skip  {path.name}: already compressed ({current})")
        return

    original = tifffile.imread(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tifffile.imwrite(tmp, original, compression=method)

    if not np.array_equal(original, tifffile.imread(tmp)):
        tmp.unlink()
        sys.exit(f"ERROR {path.name}: compressed copy does not match; original left unchanged")

    before, after = path.stat().st_size, tmp.stat().st_size
    os.replace(tmp, path)
    print(f"done  {path.name}: {original.shape} {original.dtype}, "
          f"{before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="a .tif/.tiff file or a folder of them")
    parser.add_argument("--method", default="lzw", choices=["lzw", "zlib"], help="compression (default: lzw)")
    args = parser.parse_args()

    if args.target.is_dir():
        files = sorted(p for p in args.target.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    else:
        files = [args.target]
    if not files:
        sys.exit(f"No .tif files found in {args.target}")
    for f in files:
        compress(f, args.method)


if __name__ == "__main__":
    main()
