"""Finsler Flow Matching — Path A (Deterministic, Zero-Noise SBP limit).

Modules
-------
datasets  : the Dataset container and the dense build_transition_matrix
geometry  : data-derived Randers/Finsler metric F(x, v) from a transition matrix P
models    : PhiNet (geodesic interpolant) and VelocityNet (distilled flow field)
train     : Phase 1 (geodesic interpolant) and Phase 2 (velocity distillation)
"""
