"""Pancreas endocrinogenesis: the real-data benchmark the notebook drives.

:mod:`~scripts.experiments.pancreas.datasets` cuts the cloud,
:mod:`~scripts.experiments.pancreas.tune` is the record of the hyper-parameter search,
:mod:`~scripts.experiments.pancreas.train` fills the reported grid and caches it,
:mod:`~scripts.experiments.pancreas.report` and
:mod:`~scripts.experiments.pancreas.figures` read that cache back.
``notebooks/pancreas_ffm_paper.ipynb`` drives them and is still the artefact meant to be
read top to bottom.
"""
