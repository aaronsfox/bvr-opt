# Methods comparison results

Pooled over trials and keyframe offsets; scored frames between keyframes only. Median [95th percentile].

## Primary: unfiltered

| Arm | Frames | Tibia pivot mm | Tibia rot ° | Talus pivot mm | Talus rot ° | Calcaneus pivot mm | Calcaneus rot ° | Tal–cal rot ° | Needs correction % |
|---|---|---|---|---|---|---|---|---|---|
| keyframe_interp | 666 | 0.51 [2.04] | 0.25 [0.71] | 0.55 [2.10] | 0.32 [0.88] | 0.60 [2.40] | 0.33 [0.88] | 0.33 [0.72] | 23.8 |
| pso_simplex | 666 | 30.74 [55.51] | 59.75 [78.74] | 11.94 [14.97] | 57.67 [67.91] | 11.70 [18.37] | 16.98 [22.44] | 57.08 [70.22] | 100.0 |
| new_opt | 666 | 0.24 [0.62] | 0.28 [0.61] | 0.32 [0.67] | 0.29 [0.71] | 0.24 [0.63] | 0.35 [0.74] | 0.40 [0.84] | 0.5 |
| keyframe_interp_linear | 666 | 1.22 [3.04] | 0.44 [0.89] | 1.17 [2.93] | 0.39 [1.10] | 1.19 [3.17] | 0.45 [1.08] | 0.34 [0.77] | 60.6 |
| pso_simplex_512 | 666 | 12.25 [42.72] | 17.21 [71.96] | 1.90 [9.96] | 3.95 [36.29] | 11.09 [26.58] | 19.18 [58.52] | 27.08 [71.16] | 92.5 |
| pso_simplex_single_pass | 180 | 74.97 [94.21] | 41.27 [75.47] | 12.02 [22.27] | 58.53 [76.91] | 12.01 [23.03] | 16.44 [21.21] | 62.19 [70.46] | 100.0 |
| new_opt_single_bone | 666 | 0.31 [0.54] | 0.32 [0.66] | 0.32 [0.53] | 0.33 [0.76] | 0.26 [0.45] | 0.36 [0.71] | 0.47 [0.91] | 0.0 |
| new_opt_drr_only | 666 | 0.34 [0.86] | 0.30 [0.65] | 0.55 [1.09] | 0.28 [0.70] | 0.35 [1.01] | 0.46 [0.88] | 0.45 [0.88] | 5.2 |

## Secondary: Butterworth 10 Hz on every arm

| Arm | Frames | Tibia pivot mm | Tibia rot ° | Talus pivot mm | Talus rot ° | Calcaneus pivot mm | Calcaneus rot ° | Tal–cal rot ° | Needs correction % |
|---|---|---|---|---|---|---|---|---|---|
| keyframe_interp | 666 | 0.94 [2.26] | 0.34 [0.81] | 0.91 [2.25] | 0.41 [1.05] | 0.92 [2.49] | 0.43 [1.05] | 0.33 [0.75] | 45.8 |
| pso_simplex | 666 | 24.23 [40.63] | 37.32 [63.64] | 10.57 [13.19] | 49.51 [61.59] | 9.94 [16.10] | 14.15 [17.99] | 50.80 [61.78] | 100.0 |
| new_opt | 666 | 0.93 [2.07] | 0.34 [0.80] | 0.87 [2.01] | 0.39 [0.96] | 0.85 [2.12] | 0.45 [1.03] | 0.38 [0.83] | 42.1 |
| keyframe_interp_linear | 666 | 1.58 [3.38] | 0.53 [0.99] | 1.54 [3.22] | 0.48 [1.23] | 1.55 [3.53] | 0.55 [1.25] | 0.35 [0.81] | 73.1 |
| pso_simplex_512 | 666 | 10.28 [27.09] | 13.66 [43.65] | 2.07 [5.63] | 3.35 [25.38] | 9.31 [20.17] | 15.80 [41.10] | 20.28 [52.15] | 97.1 |
| pso_simplex_single_pass | 180 | 62.95 [95.16] | 40.51 [73.59] | 12.68 [15.26] | 60.09 [69.61] | 10.68 [22.51] | 15.22 [19.84] | 58.04 [70.80] | 100.0 |
| new_opt_single_bone | 666 | 0.94 [2.11] | 0.38 [0.80] | 0.87 [1.99] | 0.40 [1.00] | 0.86 [2.10] | 0.46 [1.02] | 0.43 [0.87] | 42.2 |
| new_opt_drr_only | 666 | 0.94 [2.14] | 0.36 [0.81] | 0.95 [2.09] | 0.38 [0.96] | 0.87 [2.25] | 0.52 [1.10] | 0.42 [0.88] | 44.0 |

## Frames needing correction by trial (unfiltered, % of bone-frames)

| Arm | T0014_hop108_0001 | T0014_hop108_0002 | T0019_jog0001 | T000AssessmentTrial |
|---|---|---|---|---|
| keyframe_interp | 30.6 | 15.9 | 35.5 | 13.1 |
| pso_simplex | 100.0 | 100.0 | 100.0 | 100.0 |
| new_opt | 0.0 | 0.3 | 2.2 | 0.0 |
| keyframe_interp_linear | 58.8 | 54.9 | 67.9 | 67.5 |
| pso_simplex_512 | 88.2 | 89.4 | 100.0 | 99.7 |
| pso_simplex_single_pass | 100.0 | 100.0 | 100.0 | 100.0 |
| new_opt_single_bone | 0.0 | 0.0 | 0.0 | 0.0 |
| new_opt_drr_only | 3.8 | 5.6 | 9.3 | 3.1 |

## Compute

Renders per bone-frame: DRR renders of one bone (both views) per scored bone-frame; with gradients for new_opt arms. Wall time on the recorded device, with `workers` processes sharing it.

| Arm | Bone-frames | Renders per bone-frame | Wall time per bone-frame (s) | Total compute (device-hours) | Device | Workers |
|---|---|---|---|---|---|---|
| keyframe_interp | 1998 | 0 | 0.00 | 0.00 | cpu | 1 |
| pso_simplex | 1998 | 3764 | 231.55 | 128.51 | NVIDIA A100-SXM4-40GB | 4 |
| new_opt | 1998 | 40 | 4.71 | 2.62 | NVIDIA A100-SXM4-40GB | 4 |
| keyframe_interp_linear | 1998 | 0 | 0.00 | 0.00 | cpu | 1 |
| pso_simplex_512 | 1998 | 4391 | 134.20 | 74.48 | NVIDIA A100-SXM4-40GB | 4 |
| pso_simplex_single_pass | 588 | 3714 | 167.24 | 27.32 | NVIDIA A100-SXM4-40GB | 4 |
| new_opt_single_bone | 1998 | 40 | 4.37 | 2.43 | NVIDIA A100-SXM4-40GB | 4 |
| new_opt_drr_only | 1998 | 40 | 2.35 | 1.31 | NVIDIA A100-SXM4-40GB | 4 |
