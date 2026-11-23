import pathlib
import shutil
import sys

DOCS = pathlib.Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT))

import ffm  # noqa: E402

project = "Finsler Flow Matching"
author = "Niklas Canova, Jonas Simon Fleck"
copyright = "2026, Niklas Canova, Jonas Simon Fleck"
release = ffm.__version__

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]
myst_enable_extensions = ["dollarmath", "amsmath"]
nb_execution_mode = "off"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
suppress_warnings = ["mystnb.unknown_mime_type"]

html_theme = "furo"
html_title = "Finsler Flow Matching"
html_static_path = ["_static"]

notebooks = DOCS / "notebooks"
notebooks.mkdir(exist_ok=True)
for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
    shutil.copy2(nb, notebooks / nb.name)
