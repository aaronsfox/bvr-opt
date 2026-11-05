"""Reorganise the Welte et al. (2022) dataset into a per-trial layout with a relative-path .cfg.

Original layout (as downloaded):
    SOL001A/
        Calibration/MayaCam2/*_MayaCam.txt
        Models/IV/*.iv, Models/PV/*.tif
        <trial>/Autoscoper/POINTER_*.cfg, <trial>/Autoscoper/*.vie
        <trial>/BVR/cam1, <trial>/BVR/cam2
        <trial>/Reference/*.tra

New layout:
    SOL001A/
        <trial>/
            Calibration/*_MayaCam.txt
            Models/IV/*.iv
            Models/PV/*.tif
            RadiographImages/cam1, RadiographImages/cam2
            xParameters/*.vie
            Reference/*.tra
            <trial>.cfg          (Autoscoper config v1.1, paths relative to the trial folder)

Calibration and models are shared by all trials, so each trial receives its own copy and the
shared folders are then removed. Trial folders are any folder starting with "T".

Usage:
    python scripts/organise_welte2022.py datasets/Welte2022/data/SOL001A
    python scripts/organise_welte2022.py datasets/Welte2022/data/SOL001A --dry-run
"""
import argparse
import re
import shutil
import sys
from pathlib import Path, PureWindowsPath


def basename(value: str) -> str:
    """Last component of a path written with either / or \\ separators."""
    return PureWindowsPath(value.replace("/", "\\")).name


def rewrite_cfg(text: str) -> str:
    """Convert an absolute-path Autoscoper config into a v1.1 config relative to the trial folder."""
    out = ["Version 1.1"]
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Version"):
            continue
        key, _, value = line.partition(" ")
        value = value.strip()
        if key == "mayaCam_csv":
            out.append(f"mayaCam_csv Calibration/{basename(value)}")
        elif key == "CameraRootDir":
            out.append(f"CameraRootDir RadiographImages/{basename(value)}")
        elif key == "VolumeFile":
            out.append(f"VolumeFile Models/PV/{basename(value)}")
        elif key == "MeshFile":
            out.append(f"MeshFile Models/IV/{basename(value)}")
        else:
            out.append(f"{key} {value}")
    return "\n".join(out) + "\n"


def organise(root: Path, dry_run: bool = False) -> None:
    calib_files = sorted((root / "Calibration").rglob("*.txt")) + sorted((root / "Calibration").rglob("*.csv"))
    model_dir = root / "Models"
    trials = sorted(p for p in root.iterdir() if p.is_dir() and re.match(r"T\d", p.name))
    if not trials:
        sys.exit(f"No trial folders (T...) found in {root}")
    if not calib_files or not model_dir.is_dir():
        sys.exit(f"Expected shared Calibration/ and Models/ folders in {root}")

    def do(desc, fn, *args):
        print(("[dry-run] " if dry_run else "") + desc)
        if not dry_run:
            fn(*args)

    for trial in trials:
        print(f"\n== {trial.name}")
        # Calibration (flattened) and models: copied into each trial
        do(f"copy {len(calib_files)} calibration files -> Calibration/",
           lambda: [(trial / "Calibration").mkdir(exist_ok=True)] +
                   [shutil.copy2(f, trial / "Calibration" / f.name) for f in calib_files])
        for sub in ("IV", "PV"):
            if (model_dir / sub).is_dir():
                do(f"copy Models/{sub}/ -> Models/{sub}/",
                   shutil.copytree, model_dir / sub, trial / "Models" / sub)
        # Radiographs: BVR/cam* -> RadiographImages/cam*
        if (trial / "BVR").is_dir():
            do("move BVR/ -> RadiographImages/", shutil.move, trial / "BVR", trial / "RadiographImages")
        # Settings and config from the Autoscoper folder
        auto = trial / "Autoscoper"
        if auto.is_dir():
            vies = sorted(auto.glob("*.vie"))
            cfgs = sorted(auto.glob("*.cfg"))
            if len(cfgs) != 1:
                sys.exit(f"Expected one .cfg in {auto}, found {len(cfgs)}")
            do(f"move {len(vies)} .vie -> xParameters/",
               lambda: [(trial / "xParameters").mkdir(exist_ok=True)] +
                       [shutil.move(v, trial / "xParameters" / v.name) for v in vies])
            new_cfg = trial / f"{trial.name}.cfg"
            text = rewrite_cfg(cfgs[0].read_text())
            do(f"write {new_cfg.name} (relative paths) from {cfgs[0].name}", new_cfg.write_text, text)
            do(f"remove {cfgs[0].name} and Autoscoper/",
               lambda: (cfgs[0].unlink(), auto.rmdir()))
            if dry_run:
                print(text)

    print("\n== shared folders")
    do("remove shared Calibration/ and Models/",
       lambda: (shutil.rmtree(root / "Calibration"), shutil.rmtree(model_dir)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="the participant folder, e.g. .../data/SOL001A")
    parser.add_argument("--dry-run", action="store_true", help="print what would happen without changing files")
    args = parser.parse_args()
    organise(args.root, args.dry_run)


if __name__ == "__main__":
    main()
