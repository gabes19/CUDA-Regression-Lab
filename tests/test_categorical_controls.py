import math
from pathlib import Path

import pandas as pd
import pytest

from regressionlab.services.regression import fit_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = PROJECT_ROOT / "uploads" / "wage_education_sample.csv"


@pytest.mark.parametrize(
    "category_values",
    [
        pytest.param(
            ["A", "B", "A", "B", "A", "B", "A", "B"],
            id="binary-text",
        ),
        pytest.param(
            [
                "North",
                "South",
                "East",
                "North",
                "South",
                "East",
                "North",
                "South",
            ],
            id="three-level-text",
        ),
        pytest.param(
            [True, False, True, False, True, False, True, False],
            id="boolean",
        ),
    ],
)
def test_regression_accepts_different_categorical_controls(category_values):
    df = pd.DataFrame(
        {
            "outcome": [12, 15, 17, 22, 24, 27, 31, 34],
            "main_predictor": [1, 2, 3, 4, 5, 6, 7, 8],
            "category": category_values,
        }
    )

    results = fit_models(
        df=df,
        dependent_variable="outcome",
        main_independent_variable="main_predictor",
        controls=["category"],
    )

    assert len(results) == 2
    assert results[-1]["controls"] == ["category"]
    assert math.isfinite(results[-1]["coefficient"])


def test_wage_regression_accepts_gender_control():
    df = pd.read_csv(SAMPLE_CSV)

    results = fit_models(
        df=df,
        dependent_variable="wage",
        main_independent_variable="education",
        controls=["experience", "gender"],
    )

    assert len(results) == 3
    assert results[-1]["controls"] == ["experience", "gender"]
    assert math.isfinite(results[-1]["coefficient"])
