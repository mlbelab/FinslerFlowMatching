"""Pancreas endocrinogenesis: the real-data benchmark the notebook drives.

:mod:`~scripts.pancreas.datasets` cuts the cloud,
:mod:`~scripts.pancreas.tune` is the record of the hyper-parameter search,
:mod:`~scripts.pancreas.train` fills the reported grid and caches it,
:mod:`~scripts.pancreas.report` and
:mod:`~scripts.pancreas.figures` read that cache back.
:mod:`~scripts.pancreas.paper` turns the shipped records into the paper's tables, and
``notebooks/pancreas.ipynb`` drives it.
"""
