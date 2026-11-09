# Rater model calibration

Model and procedure: `src/raters.py`, `rater_model/calibrate.py`. Welte et al. (2022) values digitised from Figures 4 and 5. Simulated values: median across 4,000 simulated raters of each per-rater summary, with the 5th–95th percentile across raters in brackets.

Note on the digitised Figure 4: the "vertical line" values recorded as low/high match the minimum and maximum of New Operator 2's frames in Figure 5 (calcaneus 0.29/1.26 vs 0.29/1.28; tibia low 0.48 vs 0.48), not the interquartile range. They are compared here with each simulated rater's minimum and maximum over the 41 frames.

**Fitted:** AR(1) coefficient φ = 0.7 (simulated lag-1 autocorrelation 0.53, Figure 5 0.51); spread of rater quality (log-normal SD) 0.0 for rotation (operators' CV 0.14, simulated 0.23) and 0.0 for translation (operators' CV 0.34, simulated 0.34). Translation offset share 0.5, one value for all bones, fitted jointly to Figure 4's translation range and the spread of the operators' medians: calcaneus max/median 4.0 Welte vs 3.0 simulated, CV of medians 0.53 vs 0.32, talus max/median 2.1 Welte vs 3.1 simulated, CV of medians 0.32 vs 0.28, tibia max/median 2.0 Welte vs 3.0 simulated, CV of medians 0.16 vs 0.32. Within-rater range and between-rater spread pull in opposite directions (a larger offset share narrows the first and widens the second), and with three operators per bone neither is precise, so this is a compromise.

| Bone | Quantity | Scale (σ per axis) | Statistic | Welte: three operators | Welte mean | Simulated |
|---|---|---|---|---|---|---|
| calcaneus | rotation | 0.496 | median | 0.86, 0.69, 0.71 | 0.75 | 0.75 [0.42–1.28] |
| calcaneus | rotation | 0.496 | mean | 0.89, 0.74, 0.69 | 0.77 | 0.75 [0.44–1.28] |
| calcaneus | rotation | 0.496 | min | 0.51, 0.29, 0.36 | 0.39 | 0.31 [0.08–0.83] |
| calcaneus | rotation | 0.496 | max | 1.36, 1.26, 1.08 | 1.23 | 1.22 [0.82–1.78] |
| calcaneus | translation | 0.161 | median | 0.10, 0.16, 0.05 | 0.10 | 0.10 [0.06–0.22] |
| calcaneus | translation | 0.161 | mean | 0.16, 0.21, 0.09 | 0.15 | 0.12 [0.08–0.22] |
| calcaneus | translation | 0.161 | min | 0.01, 0.01, 0.01 | 0.01 | 0.00 [0.00–0.02] |
| calcaneus | translation | 0.161 | max | 0.49, 0.51, 0.20 | 0.40 | 0.33 [0.22–0.50] |
| talus | rotation | 0.260 | median | 0.46, 0.37, 0.37 | 0.40 | 0.40 [0.32–0.49] |
| talus | rotation | 0.260 | mean | 0.50, 0.45, 0.44 | 0.46 | 0.41 [0.34–0.49] |
| talus | rotation | 0.260 | min | 0.14, 0.08, 0.15 | 0.12 | 0.11 [0.04–0.18] |
| talus | rotation | 0.260 | max | 0.77, 0.89, 0.84 | 0.83 | 0.80 [0.64–1.02] |
| talus | translation | 0.469 | median | 0.42, 0.22, 0.30 | 0.31 | 0.31 [0.19–0.54] |
| talus | translation | 0.469 | mean | 0.37, 0.25, 0.32 | 0.31 | 0.36 [0.24–0.56] |
| talus | translation | 0.469 | min | 0.05, 0.01, 0.01 | 0.02 | 0.01 [0.00–0.05] |
| talus | translation | 0.469 | max | 0.75, 0.55, 0.58 | 0.63 | 1.00 [0.68–1.46] |
| tibia | rotation | 0.590 | median | 1.07, 0.77, 0.85 | 0.90 | 0.89 [0.50–1.52] |
| tibia | rotation | 0.590 | mean | 1.14, 0.81 | 0.97 | 0.89 [0.52–1.52] |
| tibia | rotation | 0.590 | min | 0.69, 0.48, 0.32 | 0.50 | 0.37 [0.10–0.99] |
| tibia | rotation | 0.590 | max | 1.41, 1.12, 1.30 | 1.28 | 1.45 [0.97–2.12] |
| tibia | translation | 0.513 | median | 0.37, 0.35, 0.27 | 0.33 | 0.34 [0.20–0.69] |
| tibia | translation | 0.513 | mean | 0.40, 0.33, 0.28 | 0.34 | 0.38 [0.25–0.70] |
| tibia | translation | 0.513 | min | 0.01, 0.03, 0.01 | 0.02 | 0.01 [0.00–0.07] |
| tibia | translation | 0.513 | max | 0.66, 0.67, 0.61 | 0.65 | 1.05 [0.70–1.59] |

## Implied pivot displacement (not observable in Welte)

Pivot displacement |t| per frame under the isotropic assumption, trained level (1×).

| Bone | Median mm | 75th pct mm | 95th pct mm | Helical translation median mm (Welte metric) |
|---|---|---|---|---|
| calcaneus | 0.25 | 0.33 | 0.45 | 0.11 |
| talus | 0.72 | 0.95 | 1.31 | 0.32 |
| tibia | 0.79 | 1.04 | 1.44 | 0.35 |

## Figure 5 checks: within-rater spread and time structure (rotation)

Figure 5 gives New Operator 2's rotation error at every frame, so the true quartiles of one operator are known. Simulated: raters with a median close to that operator's (within ±15 %).

| Bone | Welte NO2 q25 / median / q75 ° | Simulated q25 / median / q75 ° | Welte lag-1 / lag-10 | Simulated lag-1 / lag-10 |
|---|---|---|---|---|
| calcaneus | 0.57 / 0.70 / 0.86 | 0.56 / 0.70 / 0.84 | 0.47 / -0.34 | 0.60 / -0.10 |
| talus | 0.31 / 0.38 / 0.62 | 0.28 / 0.39 / 0.51 | 0.47 / 0.05 | 0.40 / -0.06 |
| tibia | 0.60 / 0.70 / 0.76 | 0.54 / 0.70 / 0.87 | 0.60 / -0.05 | 0.58 / -0.09 |

Rotation share > 2° and translation share > 1 mm across all simulated frames (trained level): calcaneus 0.001 / 0.000, talus 0.000 / 0.034, tibia 0.009 / 0.052.
