# Installation

The package imports as `ffm`:

```bash
pip install git+https://github.com/mlbelab/FinslerFlowMatching.git
```

To reproduce the paper, install the repository in editable mode; the notebooks, the experiment code in `scripts/`, the run records in `outputs/` and the derived caches in `data/` are part of the repository, not of the package:

```bash
git clone https://github.com/mlbelab/FinslerFlowMatching.git
cd FinslerFlowMatching
pip install -e ".[notebooks]"
```

`requirements-lock.txt` pins the environment of the reported runs. Rebuilding the single-cell caches needs `pip install -e ".[singlecell]"`, and the Erythroid caches a separate UniTVelo/TensorFlow environment.

Each notebook reads the run records under `outputs/` and recreates missing records with the drivers that produced them. `FFM_NO_TRAIN=1` turns a missing record into an error instead of a training run, and `NB_CPU=1` forces the CPU.
