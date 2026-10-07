"""The erythroid benchmark: our arms on Curly-FM's real-data experiment.

Petrović et al. (2025), *Curly Flow Matching*, §E.2 and Table 4 — 9815 mouse gastrulation
cells (Pijuan-Sala et al., 2019) subset to the erythroid lineage, pre-processed by
UniTVelo, binned into three equal-quantile latent-time marginals with the **middle one
withheld**, and scored at t = 1/2 in three spaces (d = 2, 20, 50).

It is a port of their evaluation — their reference-field estimator, their three
statistics, their marginal split, their spaces — run on our arms, with their baselines
transcribed rather than re-run, so the blocks share an evaluation and not a trainer.
Unlike theirs, ``P`` is rebuilt on the survivors whenever anything is held out.

Modules
-------
``reference_release``  verbatim transcription of their ``get_ut_knn_gaussian`` and cell-12
                       reduction, plus saturation diagnostics of that estimator
``datasets``           the cloud, the three marginals, the rebuilt ``P``, the split
``train``              sweep / test stages, the transcribed published block
``report``             E1 (their Table 4 shape) and E2 (the grid)
``figures``            UMAP of the marginals and the per-dimension comparison panel

Step 0 is :mod:`~scripts.experiments.erythroid.preprocess`, and it is the one module
here that runs in a *different* interpreter: UniTVelo pins TensorFlow.  It writes
``data/erythroid_cache/erythroid_d*.npz``, which are shipped, so nothing else in this
tree needs that environment.
"""
