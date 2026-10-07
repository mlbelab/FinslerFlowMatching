# Finsler Flow Matching

Given a transition matrix `P` over a point cloud, we take two spatial moments of it,

```
b₀(xᵢ) = Σⱼ Pᵢⱼ Δᵢⱼ                 the local drift
D̃(xᵢ) = Σⱼ Pᵢⱼ Δᵢⱼ Δᵢⱼᵀ             the local spread        (Δᵢⱼ = xⱼ − xᵢ)
```

and build the regularised Freidlin–Wentzell action of the SDE that `P` estimates:

```
F(x, v) = √(vᵀ C_ρ⁻¹ v) · √(‖b₀‖²_{C_ρ⁻¹} + λ²)  −  vᵀ C_ρ⁻¹ b₀ ,     C_ρ = D̃ + ρI
```

Training is three phases:

1. **Geodesic interpolant** — fit `φ` by minimising `E F(x_t, ẋ_t)²`.
2. **Finsler-cost OT** — score each `(x₀, x₁)` pair by the mean Finsler energy along the
   frozen interpolant, then entropic Sinkhorn for the coupling `π*`.
3. **Distillation** — regress `v_θ` onto `ẋ_t` under `π*`, then RK4 from `p₀`.

## Install and run

```bash
pip install -r requirements.txt    
jupyter lab notebooks/
```

Headless:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 \
  --inplace notebooks/sheet_ffm_paper.ipynb
```

A fast wiring check — one seed, short budgets, artefacts written to separate `*_smoke/`
trees so nothing reported is overwritten:

```bash
NB_SMOKE=1 jupyter nbconvert --to notebook --execute --allow-errors \
  --output-dir=/tmp --output smoke.ipynb notebooks/pancreas_ffm_paper.ipynb
```

Notebooks find the project root by walking up from their own directory, so they run from
anywhere inside `FFM/`.

| Variable | Effect |
|---|---|
| `NB_SMOKE=1` | one seed, cut budgets, separate output trees |
| `NB_CPU=1` | force CPU |
| `NB_DIMS=2,20` | restrict the dimensions swept (pancreas and erythroid; pancreas must include 2) |
| `NB_SENS_DIM=20` | dimension the erythroid sensitivity table walks |
| `NB_BAND_DIR=...` | where the pancreas band figure reads its records |
| `FINSLER_OUT=...` | put run trees somewhere other than `outputs/` |
| `PANCREAS_CACHE`, `ERYTHROID_CACHE`, `ITRACER_CACHE` | point a cache at your own copy |


## The five benchmarks

| Notebook | Cloud | Task | Train cost |
|---|---|---|---|
| `sheet_ffm_paper` | 4 000 points on a saddle in ℝ³ | withheld strip between `p₀` and `p₁` | ~20 GPU-min |
| `forksheet_ffm_paper` | 3 000 points, bifurcating sheet | withheld stretch where the arms resolve | ~50 GPU-min |
| `pancreas_ffm_paper` | scVelo endocrinogenesis, 3 696 cells | middle `latent_time` marginal withheld, `d` = 2/10/20/50 | ~90 GPU-min |
| `erythroid_ffm_paper` | mouse gastrulation erythroid, 9 815 cells | 3 spaces × 7 arms × 5 seeds | ~4.5 GPU-h |
| `itracer_r2_lineage_kernel` | iTracer R2 hindbrain, 2 901 cells | `P` from expression × lineage × velocity kernels | ~3.5 GPU-h |



Every notebook trains its baselines, OT-CFM, MFM/LAND, Curly-FM, through the same
trainer, budget and network as FFM, so a column difference is the geometry, not the
harness.



## Layout

```
FFM/
├── notebooks/        five benchmarks and the extension study bottom
│   ├── out/          figures (.png + .pdf)
│   └── tables/       the numbers they print
├── scripts/
│   ├── core/         geometry, arms, bridge, metrics, data loaders
│   │   └── vendor/   MIT-licensed upstream code for the MFM and Curly-FM baselines
│   ├── method/       the same maths in plain PyTorch, as the notebooks read it
│   └── experiments/  one package per benchmark
├── data/             derived caches (39 MB)
└── outputs/          
```

`scripts/core/` produced the paper's tables; `scripts/method/` is a second, independent
implementation of the same mathematics. Both ship, deliberately.

## Data and outputs

The derived caches ship; the raw downloads do not.

| Path | What | Rebuild |
|---|---|---|
| `pancreas_cache/` | 5 MB — the PCA/velocity cloud | `python -m scripts.core.scvelo_data --rebuild --path <h5ad>` |
| `erythroid_cache/` | 9 MB — three PCA prefixes, `d` = 2/20/50 | `python -m scripts.experiments.erythroid.preprocess` (needs the 1.4 GB Pijuan-Sala h5ad and a UniTVelo/TensorFlow env) |
| `erythroid_sweep/` | 2 MB — the selection grid the sensitivity table reads | `scripts/experiments/erythroid/tune.py` |
| `itracer/` | 23 MB — the R2 cloud and its velocity-alignment cache | needs the Mendeley supplement and a velocity h5ad |
| `band/` | 95 KB — band-access records for the pancreas figure | `scripts/experiments/pancreas/band.py` |

`outputs/` is rebuilt rather than versioned, with eight exceptions (34 MB) that would
otherwise cost two GPU-days before a single number appeared:

| Path | Buys you |
|---|---|
| `matched_bench/`, `pancreas_band/`, `pancreas_noise/` | the pancreas ablation sections, ~6.5 GPU-h |
| `erythroid_bench/` | 105 run records, ~4.5 GPU-h |
| `itracer_bench/` | 108 run records (seeds 3–11 at `d` = 2 for `itr_seeds`), ~6 GPU-h |
| `pancreas_review/`, `pancreas_review_hw/` | the pancreas robustness grid, 2 123 runs, ~38 GPU-h |
| `erythroid_review/` | the erythroid seed pool and perturbations, 127 runs, ~1 GPU-h |

Records are metrics, not trajectories: alongside the two benchmarks we ship only the
seed-0, `d` = 2 trajectory slabs the figures draw, and the pancreas grid keeps the
`t = 1/2` slice of its `d` = 2 runs. Every stage skips a run whose record already exists,
so delete a record to retrain it and an interrupted pass resumes where it stopped.


## Credits

The baselines are the authors' own code, vendored under their MIT licences in
`scripts/core/vendor/`, with the upstream file and line recorded per function:

- Kapusniak et al. (2024), *Metric Flow Matching for Smooth Interpolations on the Data
  Manifold* — the MFM/LAND arm.
- Petrović et al. (2025), *Curly Flow Matching* — the Curly-FM arm and the erythroid
  benchmark's shape.
- Tong et al. (2024), *Simulation-free Schrödinger bridges via score and flow matching* —
  the OT-CFM arm and the bridge formulation Path B follows.

Data: Bastidas-Ponce et al. (2019) pancreatic endocrinogenesis, via scVelo; Pijuan-Sala et
al. (2019) mouse gastrulation atlas; He et al. (2022), *Lineage recording in human
cerebral organoids*.

