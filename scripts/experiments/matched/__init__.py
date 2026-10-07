"""The matched-information comparison: four path objectives, one set of moments.

The published Pancreas table gives FFM a transition kernel ``P`` and gives Curly-FM
per-cell velocities.  That comparison cannot separate two things: the benefit of the
*action* this paper proposes, and the benefit of *converting velocities into a
neighbourhood-based transition representation* at all.  This tree removes the confound by
giving four objectives the identical inputs and changing nothing else.

Everything below the split is built **once** and handed to every variant unchanged --
``P``, its first moment :math:`m`, its second moment :math:`\\tilde D`, and the Gaussian
kernel that extends both off the samples.  The four variants differ only in how a path is
priced:

======================  =======================================================  ==========
variant                 cost density along the interpolant                       knobs
======================  =======================================================  ==========
``graph_drift``         :math:`\\tfrac1a\\|\\dot x - a\\,m(x)\\|^2`                ``a``
``aniso_fixed``         :math:`\\tfrac1a(\\dot x-a m)^\\top C_\\rho^{-1}(\\dot x-a m)`  ``rho``, ``a``
``ffm_second``          :math:`\\big(\\lambda\\sqrt{\\dot x^\\top C_\\rho^{-1}\\dot x}\\big)^2`  ``rho``
``ffm_full``            Eq. (12), squared -- the proposed method                 ``rho``, ``lam``
======================  =======================================================  ==========

and a fifth, ``ffm_riem``, is carried **only** to make a distinction the metric-ablation
caption currently blurs: deleting :math:`\\beta^\\top v` from Eq. (12) is *not* the same
ablation as setting :math:`m=0`, because the conformal factor
:math:`a^2=\\|m\\|^2_{C_\\rho^{-1}}+\\lambda^2` keeps first-moment information even after
the directional term is gone.  ``ffm_second`` is the :math:`m=0` control;
``ffm_riem`` is the one already in the paper; they differ by exactly
:math:`\\|m\\|^2_{C_\\rho^{-1}}`.

These are **controlled comparison variants and not published methods**.  ``graph_drift``
and ``aniso_fixed`` are fixed-time objectives built here to receive the same information
as ours; neither is Curly-FM, MFM or any other release, and no row in this tree should be
read as a reproduction of one.  The published baselines keep their own trees.

Two datasets and no more: the Sheet (a controlled synthetic route, 3-D) and Pancreas in
its 20-component PCA prefix (the existing real-data column).  Two coupling settings, both
reported: a **common** Euclidean entropic-OT coupling shared verbatim by all five arms,
which isolates the path-learning objective, and each arm's **own** learned path cost,
which evaluates the complete pipeline.  The interpolant is trained once per cell and
reused by both, so the second setting costs a distillation and not a run.

Layout
------
``variants``  the five cost objects and the two-coupling Path A
``problem``   the two datasets behind one interface: moments in, val/test scores out
``tune``      the search -- equal ladders for the two two-knob arms, ``plan/run/collect``
``bench``     the reported five paired seeds, and the paired-difference table
"""
