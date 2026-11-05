# code

`bvr_opt/` is the project's Python package (cameras, data loading, DRR renderer, cost functions, Autoscoper
PSO + simplex replica, the anchored optimiser, collision). Analysis scripts add this folder to `sys.path`
and import from `bvr_opt`. The package keeps its own name because a top-level package called `code` would
shadow Python's standard-library `code` module.
