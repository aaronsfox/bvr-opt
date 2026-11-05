# Welte hop108_0001 pilot (Analysis 1, perfect keyframes)

First run of the Analysis 1 pipeline on `T0014_hop108_0001`, window frames 20–40.

- **Keyframes:** reference (bead-based) poses every 10 frames over the whole trial (0, 10, …, 70, plus the last frame 73), i.e. a perfect rater. Hidden frames 21–29 and 31–39 are used only for scoring.
- **Scoring:** frame by frame against the reference: helical rotation (°), helical translation along the axis (mm), pivot position error (mm), joint-level error, and the minimum surface separation between neighbouring bones.

## Scripts

| Script | Purpose |
|---|---|
| `overlay.py` | Reference poses drawn as DRR silhouette outlines on both cameras (`figures/reference_overlay.png`) |
| `run_methods.py <arm>` | Runs one arm, writes `results/<arm>.npz` (+ `.json` metadata) |
| `score.py` | Scores all arms, writes `results/scores.csv`, `results/summary.md`, `figures/errors.png` |
| `queueA.sh`, `queueB.sh`, `queueC.sh` | Run queues used on the 2-core session |

## Arms

| Arm | Method |
|---|---|
| S1_spline / S1_linear | Interpolation between keyframes (cubic pivot spline + rotation spline / linear + slerp), Butterworth 10 Hz dual-pass 2nd order (Kessler 2019); `_raw` = unfiltered |
| S2 | Keyframes 10–50 refined by the Autoscoper simplex replica (filtered NCC, Welte `.vie` filters, 512² bicubic images), then spline + Butterworth |
| S3 | Per-frame Autoscoper PSO + simplex replica from the spline pose (512²), Autoscoper defaults (±3 limits, 1000 epochs, stall 25). Gap midpoints 25 and 35 only on this machine |
| P1 | Anchored gap fit (K = 3 sine correction on the spline path, hard anchors), signed gradient correlation + frame-to-frame, each bone alone |
| P2 | As P1, tibia + talus + calcaneus co-fitted and rendered together |
| P3 / P3tol | P2 + contact penalty, μ = 0.01 mm⁻², d_min = 0 / −2 mm |
| P2_drr / P2_f2f | Components: DRR term only / frame-to-frame term only |
| PF | No anchoring: each hidden frame fitted alone (co-fitted, DRR term) from the spline pose |

## Notes from setting up

- **Geometry check:** MayaCam 2.0 is an OpenCV pinhole and Welte `.tra` files are column-major. Reference-pose outlines match the bone edges in both cameras.
- **Reference interpenetration:** at the reference poses, the talus and calcaneus segmentations (v > 0) interpenetrate by up to 1.8 mm under load. The masks are 0.44 mm apart in the CT, and the talus moves about 10° and 1.4 mm relative to the calcaneus during stance. With bone-thresholded masks the overlap is about 1 mm. Tibia–talus keeps a 0.6–0.8 mm gap. A contact penalty with d_min = 0 therefore biases the subtalar joint away from the truth on this dataset.
- **Butterworth filtering:** filtering the reference motion itself at 10 Hz gives 0.9–1.0 mm median (max 1.8 mm) pivot error over frames 21–39 (landing). At 6 Hz the error is about 2.4 mm.
- **Compute:** one Autoscoper-replica cost evaluation (both cameras, 512 render) takes 0.2–0.35 s on 1 CPU thread. One co-fitted gap iteration takes 2.4 s at 256 and 7.4 s at 512 (2 threads). Full-resolution (1760 render) runs and S3 on all frames need the cluster.
