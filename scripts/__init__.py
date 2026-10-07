"""Everything that is code, in one place; everything that is a run, in ``notebooks/``.

Three packages and one dependency direction, which is the whole reason the split exists:

* :mod:`scripts.core` — the engine.  Clouds, transition matrices, the Randers metric, the
  arms, the scoring.  It imports nothing above it, so it can be read on its own.
* :mod:`scripts.experiments` — one subpackage per benchmark: its cloud, its splits and
  its selection grid.  May import :mod:`scripts.core`, never :mod:`scripts.method`.
* :mod:`scripts.method` — the method as plain PyTorch modules, plus the shared
  presentation layer.  May import both of the above.  Nothing imports it back.

A notebook imports :mod:`scripts.method` and its own experiment, and holds the run: the
cloud, the protocol, the pasted hyper-parameters, the table and the prose.
"""
