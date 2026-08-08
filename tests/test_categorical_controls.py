import math
from pathlib import Path

import pandas as pd
import pytest

from regressionlab.services.regression import fit_models
from regressionlab.services.data_processing import prepare_analysis_data


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

    prepared = prepare_analysis_data(
        df=df,
        dependent_variable="outcome",
        main_independent_variable="main_predictor",
        controls=["category"],
    )

    results = fit_models(
        data=prepared,
        dependent_variable="outcome",
        main_independent_variable="main_predictor",
        controls=["category"],
    )

    assert len(results) == 2
    assert results[-1]["controls"] == ["category"]
    assert math.isfinite(results[-1]["coefficient"])


def test_wage_regression_accepts_gender_control():
    df = pd.read_csv(SAMPLE_CSV)

    prepared = prepare_analysis_data(
        df=df,
        dependent_variable="wage",
        main_independent_variable="education",
        controls=["experience", "gender"],
    )

    results = fit_models(
        data=prepared,
        dependent_variable="wage",
        main_independent_variable="education",
        controls=["experience", "gender"],
    )

    assert len(results) == 3
    assert results[-1]["controls"] == ["experience", "gender"]
    assert math.isfinite(results[-1]["coefficient"])

def test_prepare_analysis_data_encodes_categorical_control():
    df = pd.DataFrame({
        "outcome": [12, 15, 17, 22],
        "main_predictor": [1, 2, 3, 4],
        "group": ["A", "B", "A", "B"],
    })

    prepared = prepare_analysis_data(
        df=df,
        dependent_variable="outcome",
        main_independent_variable="main_predictor",
        controls=["group"],
    )

    assert prepared.y.dtype == float
    assert prepared.X.columns.tolist() == [
        "main_predictor",
        "group_B",
    ]
    assert prepared.term_map == {
        "main_predictor": ["main_predictor"],
        "group": ["group_B"],
    }
