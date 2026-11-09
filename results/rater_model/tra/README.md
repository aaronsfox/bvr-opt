# Rater keyframes as Autoscoper .tra files

Written by `rater_model/export_tra.py` from the simulated raters in `../raters/` (the `.npz` files hold every
frame plus the error vectors; these hold keyframes only).

```
tra/
  reference/s<spacing>/<trial>_<bone>.tra        bead-based keyframes (perfect rater)
  L<level>/r<id>/s<spacing>/<trial>_<bone>.tra   simulated rater <id> at level <level> (1 = trained, 2x, 4x)
```

Bones: `cal` calcaneus, `tal` talus, `tib` tibia. Each file has one line per frame of the trial: the 4x4 pose,
column-major as in Welte's reference files (translation in elements 13-15), on keyframes, and NaN on every
other frame. Load in Autoscoper with: Matrix, Orientation Column, Comma, mm, degrees, one bone at a time
(Volume = Current).

Keyframes per spacing (bookends at frame 5 and N - 6; regular keyframes every s frames; `full` = bookends):

| Frames | Trial | s5 | s10 | s20 | s30 | s40 | sfull |
|---|---|---|---|---|---|---|---|
| 74 | T0014_hop108_0001 | 5,10,15,20,25,30,35,40,45,50,55,60,65,68 | 5,15,25,35,45,55,68 | 5,25,45,68 | 5,35,68 | 5,45,68 | 5,68 |
| 67 | T0014_hop108_0002 | 5,10,15,20,25,30,35,40,45,50,55,61 | 5,15,25,35,45,55,61 | 5,25,45,61 | 5,35,61 | 5,61 | 5,61 |
| 39 | T0019_jog0001 | 5,10,15,20,25,30,33 | 5,15,25,33 | 5,33 | 5,33 | 5,33 | 5,33 |
| 41 | T000AssessmentTrial | 5,10,15,20,25,30,35 | 5,15,25,35 | 5,25,35 | 5,35 | 5,35 | 5,35 |
