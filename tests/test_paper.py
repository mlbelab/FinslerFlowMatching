import importlib
import os

import pytest

os.environ["FFM_NO_TRAIN"] = "1"

EXPECTED = {
    "sheet": {
        "mid_end": ["0.1588 ± 0.0096", "0.1449 ± 0.0116", "0.5221 ± 0.0016", "0.0646"],
        "ablation": ["0.7165 ± 0.0136", "0.6672 ± 0.0160", "0.958"],
        "solvers": ["0.103 ± 0.007", "0.452 ± 0.063", "0.134 ± 0.034"],
    },
    "forksheet": {
        "table9": ["69.7 ± 13.2", "92.1 ± 9.5", "95.9 ± 5.9", "100.0 ± 0.0"],
    },
    "pancreas": {
        "table1": ["10.978 ± 0.166", "10.942 ± 0.075", "12.785 ± 0.088"],
        "table1_bottom": ["11.92 ± 0.06", "10.94 ± 0.15", "10.93 ± 0.14"],
        "figure4a": ["11.2195 ± 0.1615", "1.4791 ± 0.0168"],
        "figure4b": ["10.9619 ± 0.0989", "131.2148 ± 4.3699"],
        "figure4c": ["4.8931 ± 0.0907"],
        "table6": ["9.27 ± 0.26", "11.51 ± 0.06"],
        "table8": ["1.90 ± 0.08", "12.89 ± 0.11"],
        "table10": ["10.93 ± 0.14", "12.34 ± 0.10"],
        "table2": ["0.160", "0.129 ± 0.020", "0.519 ± 0.030", "0.098 ± 0.009", "0.366 ± 0.019"],
    },
    "erythroid": {
        "table1": ["4.778 ± 0.013", "6.588 ± 0.309", "8.181 ± 0.180"],
        "table7_pseudotime": ["0.245 ± 0.034", "6.103 ± 0.074", "7.084 ± 0.021"],
        "table7_realtime": ["5.755 ± 0.172", "8.136 ± 0.095"],
    },
    "eb": {
        "table1": ["7.794 ± 0.143", "10.522 ± 0.217", "11.174 ± 0.303"],
    },
    "itracer": {
        "table2": ["12.85 ± 0.07", "12.74 ± 0.07", "16.76 ± 0.21"],
    },
    "larry": {
        "table2": ["0.552 ± 0.012", "0.487 ± 0.017", "0.883 ± 0.030", "0.441 ± 0.013", "0.404 ± 0.016", "0.512 ± 0.015"],
    },
    "geodesic": {
        "table3": ["0.75", "3.9", "4.1"],
    },
}


@pytest.mark.parametrize("dataset", sorted(EXPECTED))
def test_paper_tables(dataset):
    paper = importlib.import_module(f"scripts.{dataset}.paper")
    paper.ensure()
    for fn, values in EXPECTED[dataset].items():
        text = getattr(paper, fn)().to_string()
        for v in values:
            assert v in text, f"{dataset}.{fn}: {v} not found"
