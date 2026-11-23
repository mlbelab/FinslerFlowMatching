# API

The method is exposed as plain PyTorch modules in `ffm.method`; the most common entry points are re-exported at the top level, e.g. `ffm.path_a`.

```python
import ffm

cfg = ffm.Run.from_env()
m, C = ffm.moments_from(P, X)
metric = ffm.FWFactory(X, C, m, cfg, rho_mult=1.0, lam_mult=0.3).base
X0, X1 = cfg.tensor(X[source]), cfg.tensor(X[target])
det = ffm.path_a(metric, X0, X1, seed=0, cfg=cfg)
sde = ffm.path_b(metric, det["phi"], det["coupler"], X0, ffm.sigma_for(metric, 0.3), seed=0, cfg=cfg)
```

## Configuration

```{eval-rst}
.. autoclass:: ffm.method.Run
   :members: from_env, but
```

## Transition moments and metric

```{eval-rst}
.. autofunction:: ffm.method.moments_from
.. autofunction:: ffm.method.rho_star
.. autoclass:: ffm.method.TrueFW
.. autoclass:: ffm.method.FWFactory
   :members: at, rho_rungs, lam_rungs
.. autofunction:: ffm.method.sigma_for
.. autofunction:: ffm.method.c_eff_of
```

## Training and inference

```{eval-rst}
.. autofunction:: ffm.method.train_geodesic
.. autofunction:: ffm.method.finsler_cost
.. autofunction:: ffm.method.build_ot
.. autofunction:: ffm.method.train_cfm
.. autofunction:: ffm.method.integrate
.. autofunction:: ffm.method.path_a
.. autofunction:: ffm.method.train_sbm
.. autofunction:: ffm.method.path_b
```

## Evaluation

```{eval-rst}
.. autofunction:: ffm.method.euclidean_w2
.. autofunction:: ffm.method.finsler_w2
```
