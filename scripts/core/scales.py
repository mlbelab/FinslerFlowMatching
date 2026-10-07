"""Hyper-parameter ladders in units the data supplies, rather than in bare numbers.

Three of this engine's knobs are floors and widths on tensors built out of squared
displacements, so their meaning moves with the cloud:

``rho``        the isotropic floor of M = ρI + Σ
``fw_lambda``  the quadrature floor λ on the FW conformal factor a = ‖b‖_{G_0}
``sigma``      Path B's bridge width

A grid of *absolute* values for any of them is a grid that means a different thing at
every dimension, and the failure is quiet rather than loud.  The erythroid tree is the
worked example: its ρ ladder ran ``0.0003 … 1.0`` at d = 2, 20 and 50 alike, but the
scale ρ is a floor *on* — ``mean_i tr Σ / d``, which the Σ temperature pins to
``target_trace / d`` — is 1.0, 0.1 and 0.04 at those three dimensions.  So the same rung
sat two and a half decades apart in the only units that matter, and eight rungs bought
an argmin that moved the score by 5.8 % where the gap to the baseline was 28 %.  Nothing
in the records said so, because every record said ``rho: 0.03`` and that looked fine.

The fix is the one :mod:`scripts.method.metric` already applies on the Pancreas and the
Sheet: never quote a rung, always quote a *multiple* of a measured scale.

    rho     = rho_mult   · rho_star     rho_star   = mean_i tr Σ(x_i) / d
    lambda  = lam_mult   · a_raw_bar    a_raw_bar  = mean_i ‖b(x_i)‖_{G_0(x_i)}
    sigma   = width_mult · 2ε/sqrt(m_bar)   -- i.e. width_mult smoothing bandwidths

so ``rho_mult = 0.3`` is "three tenths of the diffusion's own mean eigenvalue" in every
space, and one ladder is valid at every dimension.

One thing a multiple does *not* make portable is how far down a λ ladder can reach.  Under
``fw_lambda`` the admissibility margin is λ²/(a_raw² + λ²), which is analytically positive
for every λ > 0 but is representable only while it clears a few ulp of the working dtype;
below that β^T G^{-1} β rounds to 1 and the metric degenerates.  Where that happens depends
on the spread of ``a_raw`` over the cloud, so it is measured — :func:`lam_mult_floor` — and
enforced at resolve time rather than assumed by whoever wrote the grid.

Why this module and not :class:`scripts.method.metric.FWFactory`
----------------------------------------------------------------
It is the same arithmetic, deliberately.  ``FWFactory`` measures the scales off the
notebook implementation's raw moments; this measures them off :class:`~scripts.core
.geometry.FinslerGeometry`, whose Σ carries the ``target_trace`` temperature.  That
temperature is a **global scalar** s = target_trace / mean tr D̃, and under it

    C_core = ρ_c I + s·D̃ = s·(ρ_c/s·I + D̃),      a_core = a_method / sqrt(s),

so with ρ_c = s·ρ_m and λ_c = λ_m/sqrt(s) the two metrics are equal up to the global
factor 1/s — which the Phase-1 minimiser, the entropic coupling and ā's own rescale are
all invariant to.  Dividing through, ``rho/rho_star`` and ``lam/a_raw_bar`` come out
*identical* in the two trees.  The multiples are the invariant coordinates, which is why
the two implementations can share one ladder without sharing one temperature convention,
and it is what the verification suite in the research repository pins numerically — including the negative
control that handing one tree the other's *absolute* ρ is off by s, which is 184 on its
fixture.

Usage — the experiment layer resolves a grid point just before it builds the arm:

    hp, scales = resolve(ts, hp, {"rho_mult": 0.3, "lam_mult": 0.03}, device, dtype)
    res = run_arm("ffm", ts, hp, seed, device, dtype)

``resolve`` returns the measured scales alongside the HParams so a record can carry both
the multiple it was addressed by and the absolute it actually ran at.  Storing only one
of them loses something real: the multiple alone cannot be replayed against a changed
cloud, and the absolute alone cannot be compared across dimensions.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from scripts.core.arms import HParams, build_geometry
from scripts.core.protocol import TrainingSet

#: multiple -> the :class:`~scripts.core.arms.HParams` field it resolves to.  A grid
#: writes the left-hand names and an ``HParams`` only ever sees the right-hand ones, so
#: a ladder cannot be half-converted: an unresolved ``rho_mult`` reaching ``HParams``
#: is an unknown-field error rather than a silently ignored knob.
MULTIPLE_OF: dict[str, str] = {
    "rho_mult": "rho",
    "lam_mult": "fw_lambda",
    "width_mult": "sigma",
}


def split_multiples(point: dict) -> tuple[dict, dict]:
    """One grid point -> ``(multiples, absolutes)``, by key name alone.

    Kept as a function rather than inlined at each call site because "which of these
    keys is a multiple" has to be answered identically by the resolver, by the record
    writer and by the tuner's edge check, and three copies of a membership test against
    :data:`MULTIPLE_OF` is three chances to disagree about ``width_mult``.
    """
    mult = {k: v for k, v in point.items() if k in MULTIPLE_OF}
    plain = {k: v for k, v in point.items() if k not in MULTIPLE_OF}
    return mult, plain


def sigma_from_width(width_mult: float, eps_kernel: float, m_bar: float) -> float:
    """The σ whose bridge half-width at t = 1/2 is ``width_mult`` smoothing bandwidths.

    A conditional bridge has per-direction std ``σ·sqrt(t(1-t)·m_bar)``, so at t = 1/2 it
    is ``σ·sqrt(m_bar)/2``; setting that equal to ``width_mult·ε`` and solving gives the
    expression below.  Character-for-character
    :func:`scripts.method.metric.sigma_for`, which is the point — Path B's noise level is
    the one hyper-parameter with no Path A analogue, so if the two trees quoted it
    differently there would be nothing to compare the columns against.
    """
    return 2.0 * float(width_mult) * float(eps_kernel) / float(m_bar) ** 0.5


def measure(ts: TrainingSet, hp: HParams, device: torch.device,
            dtype: torch.dtype) -> dict:
    """Build the geometry ``hp`` asks for and read the three scales off it.

    One build, so ``a_raw_bar`` is measured at the ρ this cell actually runs at rather
    than at a reference ρ — it depends on ρ through G_0 = (ρI + Σ)^-1, and pretending
    otherwise would make ``lam_mult`` mean a slightly different thing on every rung of
    the ρ ladder.  :class:`scripts.method.metric.FWFactory` measures it the same way for
    the same reason.

    The build is charged to every cell that quotes a multiple, and it is a real cost:
    2.5 s at d = 2 and 10.6 s at d = 50 on CPU, against 30 s and 330 s of training.  It
    is not cached, because the alternative is a cache keyed on (cloud, split, ρ) that
    would have to be invalidated by hand the first time the training set moves — and a
    stale scale is a silently mis-scaled ladder, which is the failure this whole module
    exists to remove.
    """
    geom = build_geometry("randers", ts, hp, device, dtype)
    return {
        "rho_star": geom.rho_star,
        "a_raw_bar": float(geom.a_raw_bar),
        "a_raw_max": float(geom.a_raw_max),
        "eps_kernel": float(geom.eps_kernel),
        "m_bar": geom.mean_mobility_eig(),
    }


#: how many ulp of the working dtype the admissibility margin must clear.
#:
#: The hard limit is half an ulp: the largest float below 1.0 is 1 - eps/2, so a margin
#: under that cannot make ``β^T G^-1 β < 1`` true no matter how exactly it is computed.
#: Eight is that limit with a factor of sixteen of headroom, which buys two things — the
#: round-off of the d x d inverse and the two contractions
#: :meth:`~scripts.core.geometry.FinslerGeometry.verify` forms the quadratic with (a
#: factor under 2 on this engine's clouds), and ``verify`` probing at points *jittered*
#: off the data, where a_raw can exceed the measured maximum this floor is built from.
#:
#: It is not fitted to any tree's ladder, but it does reproduce the boundary the erythroid
#: re-sweep ran into: ``lam_mult = 0.001`` is refused at all three dimensions, having
#: actually died at d = 20 and d = 50 and survived at d = 2 by two ulp, and ``0.003`` is
#: admitted at all three, the tightest being d = 20 at 33 % above the floor.
MARGIN_ULP = 8.0


def lam_mult_floor(a_raw_max: float, a_raw_bar: float, dtype: torch.dtype) -> float:
    """The smallest ``lam_mult`` at which the metric is still Finsler in ``dtype``.

    Under ``fw_lambda`` admissibility is analytic — β^T G^{-1} β = 1 - λ²/(a_raw² + λ²)
    is below 1 for every λ > 0 — but *representability* is not.  The margin is smallest
    where a_raw is largest, and once ``λ²/a_raw_max²`` drops under a few ulp the
    quadratic form rounds to exactly 1: the strong-convexity condition fails, F vanishes
    on a direction, and the cell dies in ``verify`` after paying for a geometry build.

    Solving ``λ²/(a_max² + λ²) = MARGIN_ULP·eps`` for λ and dividing by ā gives the
    multiple below.  It is a property of the *cloud* and not of the tree that quotes it:
    the same ladder is fine on one data set and not on another, which is why the floor is
    measured here rather than written into any one experiment's grid.  On the Pancreas in
    float32 it comes out below 0.001 and the published ladder is untouched; on the
    erythroid cloud it lands between 0.001 and 0.003, which is exactly where that tree's
    bottom rung was failing at d = 20 and d = 50 and passing by two ulp at d = 2.
    """
    eps = float(torch.finfo(dtype).eps)
    m = MARGIN_ULP * eps
    return float(a_raw_max) * (m / (1.0 - m)) ** 0.5 / float(a_raw_bar)


def resolve(ts: TrainingSet, hp: HParams, point: dict, device: torch.device,
            dtype: torch.dtype) -> tuple[HParams, dict]:
    """``(hp with every multiple resolved to an absolute, the scales it was resolved at)``.

    ``point`` may mix multiples and ordinary overrides; the ordinary ones are applied
    unchanged, so an arm whose knob is ``curly_alpha`` or ``land_rho`` passes through
    here untouched and without paying for a geometry build.

    ``lam_mult`` is refused unless the metric is the one λ exists in.  Under
    ``metric_form="randers"`` there is no λ at all — the 1-form is renormalised to
    ‖β‖ ≡ c and the magnitude of the drift is discarded — so a λ ladder there would be
    eight identical runs, which is exactly the dead search axis this module was written
    after.  Better to refuse it than to measure it.

    That refusal is checked against the **staged** HParams and not against ``hp``, because
    the caller may well be setting the form in this very call: an experiment tree pins its
    metric in one dict and sweeps the multiples in another, and the two arrive here merged.
    Checking ``hp`` would refuse every such point for carrying the default it is about to
    overwrite.

    ``lam_mult`` is refused a second time if it is below :func:`lam_mult_floor` — small
    enough that the admissibility margin is not representable in ``dtype``.  That check is
    here and not in the grid because the floor is a measured property of the cloud: the
    same eight rungs are all admissible on the Pancreas and the bottom one is not on the
    erythroid data.  Refusing at resolve time costs one geometry build; the alternative is
    the same cell dying two minutes later inside
    :meth:`~scripts.core.geometry.FinslerGeometry.verify` with ``max β^T G^-1 β = 1.0000``
    and nothing saying which knob to move.
    """
    mult, plain = split_multiples(point)
    staged = replace(hp, **plain) if plain else hp
    if not mult:
        return staged, {}

    if "lam_mult" in mult:
        assert staged.metric_form == "fw_lambda", (
            f"lam_mult needs metric_form='fw_lambda' (λ is that form's regulariser), "
            f"got {staged.metric_form!r}; under 'randers' β is renormalised to ‖β‖ ≡ c "
            f"and λ does not exist, so the ladder would be one point measured eight times")

    # ρ first and on its own: it is the only multiple whose scale can be known before
    # the geometry exists (the Σ temperature pins mean tr Σ/d to target_trace/d), and
    # the other two must be measured at the ρ this cell runs at, not at hp's default.
    rho_star = staged.target_trace / ts.X.shape[1]
    if "rho_mult" in mult:
        staged = replace(staged, rho=float(mult["rho_mult"]) * rho_star)

    scales = measure(ts, staged, device, dtype)
    assert abs(scales["rho_star"] - rho_star) <= 1e-6 * max(1.0, rho_star), (
        f"the Σ temperature did not pin mean tr Σ/d: predicted {rho_star:.6g} from "
        f"target_trace/d, measured {scales['rho_star']:.6g}.  Every rho_mult rung is "
        f"quoted against the predicted value, so they are not the ladder that ran")

    edits: dict = {}
    if "lam_mult" in mult:
        floor = lam_mult_floor(scales["a_raw_max"], scales["a_raw_bar"], dtype)
        assert float(mult["lam_mult"]) >= floor, (
            f"lam_mult = {float(mult['lam_mult']):.3g} is below this cloud's "
            f"admissibility floor {floor:.3g} in {dtype}: the margin "
            f"1 - β^T G^-1 β = λ²/(a_raw² + λ²) would be under {MARGIN_ULP:g} ulp at "
            f"a_raw_max = {scales['a_raw_max']:.4g}, so it rounds to zero and F vanishes "
            f"on a direction.  The rung is not a Finsler metric at this precision — move "
            f"the ladder's bottom up to the first rung above {floor:.3g} rather than "
            f"recording a degenerate cell")
        edits["fw_lambda"] = float(mult["lam_mult"]) * scales["a_raw_bar"]
    if "width_mult" in mult:
        edits["sigma"] = sigma_from_width(
            mult["width_mult"], scales["eps_kernel"], scales["m_bar"])
    return replace(staged, **edits), {**scales,
                                      **{f"resolved_{k}": v for k, v in edits.items()}}
