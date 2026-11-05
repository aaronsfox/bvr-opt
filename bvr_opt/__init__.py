"""bvr_opt: keyframe-anchored biplanar videoradiography (BVR) bone tracking.

Modules
-------
cameras    MayaCam 2.0 camera model (Autoscoper / XMALab calibration files)
io         Welte-style trial loading (cfg, calibration, volumes, reference .tra, radiographs, .vie)
geometry   Poses with pivot-centred increments, interpolation, filtering, helical-axis errors
drr        Differentiable PyTorch DRR renderer reproducing Autoscoper's ray caster
filters    PyTorch ports of Autoscoper's image filters and the .vie parser
costs      Image similarity terms (Autoscoper filtered NCC, signed gradient correlation, frame-to-frame)
collision  Signed-distance-field contact penalty between co-fitted bones
optim      Autoscoper PSO + downhill simplex replica
tracking   Keyframe-anchored gap fitting (proposed method)
"""
