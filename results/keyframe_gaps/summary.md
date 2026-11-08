# Keyframe gaps results

Perfect keyframes; bookends at frame 5 and N − 6. Median [95th percentile], unfiltered, pooled over trials.

## All scored frames per spacing (primary)

Every non-keyframe between the bookends, for each spacing.

| Arm | Spacing | Bone-frames | Tibia pivot mm | Tibia rot ° | Talus pivot mm | Talus rot ° | Calcaneus pivot mm | Calcaneus rot ° | Max pivot mm | Tal–cal rot ° | Needs correction % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| keyframe_interp | 5 | 423 | 0.10 [0.37] | 0.14 [0.32] | 0.08 [0.33] | 0.20 [0.46] | 0.12 [0.37] | 0.22 [0.50] | 0.69 | 0.32 [0.54] | 0.0 |
| keyframe_interp | 10 | 477 | 0.48 [2.00] | 0.22 [0.69] | 0.53 [2.08] | 0.32 [0.86] | 0.57 [2.35] | 0.31 [0.86] | 3.26 | 0.32 [0.73] | 23.3 |
| keyframe_interp | 20 | 504 | 2.62 [11.40] | 0.87 [4.80] | 2.33 [8.96] | 0.84 [2.44] | 2.36 [7.68] | 1.05 [3.57] | 12.63 | 0.57 [3.04] | 73.2 |
| keyframe_interp | 30 | 513 | 3.68 [12.30] | 1.70 [5.33] | 3.11 [11.06] | 1.11 [2.44] | 2.87 [10.89] | 1.51 [3.56] | 12.63 | 1.00 [3.03] | 88.7 |
| keyframe_interp | 40 | 516 | 8.32 [37.73] | 3.43 [14.51] | 7.58 [37.56] | 1.68 [7.48] | 6.97 [37.29] | 2.89 [10.34] | 38.84 | 1.49 [6.01] | 94.2 |
| keyframe_interp | full | 519 | 18.76 [47.55] | 6.85 [16.76] | 19.50 [47.26] | 3.77 [9.33] | 19.77 [47.42] | 5.13 [12.75] | 49.80 | 3.49 [6.09] | 99.6 |

## Common frame set

Frames that are not a keyframe at any spacing, so every spacing is scored on identical frames.

| Arm | Spacing | Bone-frames | Tibia pivot mm | Tibia rot ° | Talus pivot mm | Talus rot ° | Calcaneus pivot mm | Calcaneus rot ° | Max pivot mm | Tal–cal rot ° | Needs correction % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| keyframe_interp | 5 | 423 | 0.10 [0.37] | 0.14 [0.32] | 0.08 [0.33] | 0.20 [0.46] | 0.12 [0.37] | 0.22 [0.50] | 0.69 | 0.32 [0.54] | 0.0 |
| keyframe_interp | 10 | 423 | 0.45 [1.89] | 0.22 [0.67] | 0.50 [2.04] | 0.31 [0.85] | 0.53 [2.22] | 0.31 [0.82] | 3.26 | 0.32 [0.71] | 21.7 |
| keyframe_interp | 20 | 423 | 2.45 [11.06] | 0.83 [4.80] | 2.29 [8.74] | 0.79 [2.43] | 2.21 [7.61] | 1.01 [3.53] | 12.63 | 0.52 [2.99] | 71.4 |
| keyframe_interp | 30 | 423 | 3.66 [12.30] | 1.64 [5.37] | 3.05 [10.96] | 1.07 [2.43] | 2.80 [10.79] | 1.43 [3.53] | 12.63 | 0.98 [2.99] | 87.5 |
| keyframe_interp | 40 | 423 | 7.83 [37.71] | 3.41 [14.51] | 7.54 [37.44] | 1.65 [7.48] | 6.68 [37.35] | 2.87 [10.34] | 38.84 | 1.44 [6.00] | 93.4 |
| keyframe_interp | full | 423 | 17.85 [47.43] | 6.67 [16.73] | 18.26 [47.18] | 3.72 [9.29] | 18.17 [47.24] | 5.09 [12.74] | 49.80 | 3.30 [6.15] | 99.5 |

## Frames needing correction by trial (% of bone-frames; gap lengths used in brackets)

| Arm | Spacing | T0014_hop108_0001 | T0014_hop108_0002 | T0019_jog0001 | T000AssessmentTrial |
|---|---|---|---|---|---|
| keyframe_interp | 5 | 0.0 (3,5) | 0.0 (5,6) | 0.0 (3,5) | 0.0 (5) |
| keyframe_interp | 10 | 36.3 (10,13) | 16.0 (6,10) | 24.0 (8,10) | 8.6 (10) |
| keyframe_interp | 20 | 86.7 (20,23) | 37.1 (16,20) | 97.5 (28) | 89.3 (10,20) |
| keyframe_interp | 30 | 92.3 (30,33) | 74.1 (26,30) | 97.5 (28) | 100.0 (30) |
| keyframe_interp | 40 | 84.7 (23,40) | 100.0 (56) | 97.5 (28) | 100.0 (30) |
| keyframe_interp | full | 100.0 (63) | 100.0 (56) | 97.5 (28) | 100.0 (30) |

## Median pivot error (mm) by distance to the nearest keyframe (frames)

| Arm | Spacing | 1–2 | 3–5 | 6–10 | 11–15 | 16–20 | >20 |
|---|---|---|---|---|---|---|---|
| keyframe_interp | 5 | 0.10 | 0.20 | – | – | – | – |
| keyframe_interp | 10 | 0.33 | 0.67 | 0.50 | – | – | – |
| keyframe_interp | 20 | 0.86 | 2.00 | 3.36 | 8.69 | – | – |
| keyframe_interp | 30 | 1.21 | 2.45 | 3.61 | 4.59 | 3.86 | – |
| keyframe_interp | 40 | 1.86 | 5.36 | 7.88 | 10.97 | 32.18 | 37.76 |
| keyframe_interp | full | 2.90 | 7.73 | 12.13 | 24.32 | 35.15 | 42.23 |

## Compute

Renders per bone-frame: DRR renders of one bone (both views, with gradients) per scored bone-frame. Wall time on the recorded device, with `workers` processes sharing it.

| Arm | Spacing | Bone-frames | Renders per bone-frame | Wall time per bone-frame (s) | Device-hours | Device | Workers |
|---|---|---|---|---|---|---|---|
| keyframe_interp | 5 | 423 | 0 | 0.00 | 0.00 | cpu | 1 |
| keyframe_interp | 10 | 477 | 0 | 0.00 | 0.00 | cpu | 1 |
| keyframe_interp | 20 | 504 | 0 | 0.00 | 0.00 | cpu | 1 |
| keyframe_interp | 30 | 513 | 0 | 0.00 | 0.00 | cpu | 1 |
| keyframe_interp | 40 | 516 | 0 | 0.00 | 0.00 | cpu | 1 |
| keyframe_interp | full | 519 | 0 | 0.00 | 0.00 | cpu | 1 |

