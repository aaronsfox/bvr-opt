# Results

One subfolder per analysis. Each is written by that analysis's code and committed from the cluster.

| Folder | Analysis | Code |
|---|---|---|
| `methods_comparison/` | Methods comparison with perfect keyframes (Analysis 1 in the plan): one subfolder per arm (`keyframe_interp`, `pso_simplex`, `new_opt`, supplementary arms), plus `scores.csv`, `compute.csv`, `summary.md`, `figures/` | `methods_comparison/` |
| `keyframe_gaps/` | Keyframe gaps with perfect keyframes (Analysis 3 in the plan): `<arm>/s<spacing>/units|tracks` for spacings 5, 10, 20, 30, 40 and full trial, plus `scores.csv`, `compute.csv`, `summary.md`, `figures/` | `keyframe_gaps/` |
| `rater_model/` | Simulated raters calibrated to Welte 2022: `params.json`, `calibration.md`, `figures/`, `raters/L<level>/r<id>/<trial>.npz` | `rater_model/` |
| `rater_gaps/` | Keyframe gaps with simulated raters: `L<level>/r<id>/<arm>/s<spacing>/...`, plus `scores.csv`, `keyframes.csv`, `summary.md`, `figures/` | `rater_gaps/` |
