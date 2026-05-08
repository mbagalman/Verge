"""Smoke tests for the typing.Literal aliases on ModelFit / GrowthAnalysis.

Python doesn't enforce ``Literal`` at runtime, so these tests pin the
*contents* of the aliases rather than assertion-firing on misuse. If a
new candidate model is added, or an indeterminate reason is renamed, the
membership tests below need an explicit update -- which is the point.
"""

from typing import get_args

import numpy as np

from project_verge import (
    IndeterminateReason,
    ModelName,
    PreferredModel,
    analyze_growth,
)


def _logistic_series(k=120.0, r=0.7, t0=6.0, n=22, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


def test_model_name_literal_membership():
    assert set(get_args(ModelName)) == {
        "exponential",
        "linear",
        "logistic",
        "power_law",
    }


def test_preferred_model_literal_excludes_power_law_and_includes_indeterminate():
    # power-law is diagnostic-only and cannot become a verdict; the
    # indeterminate gate fires first on a power-law-leading BIC.
    members = set(get_args(PreferredModel))
    assert members == {"exponential", "linear", "logistic", "indeterminate"}
    assert "power_law" not in members
    assert "indeterminate" in members


def test_indeterminate_reason_literal_membership_in_precedence_order():
    # Order matches the precedence chain in analyze_growth -- not a typing
    # constraint, but a useful documentation invariant.
    expected = (
        "neither_model_fits",
        "power_law_shape",
        "ambiguous_evidence",
        "logistic_unidentifiable",
        "signal_disagreement",
        "fragile_verdict",
    )
    assert get_args(IndeterminateReason) == expected


def test_runtime_preferred_model_value_is_in_literal_set():
    time, values = _logistic_series()
    result = analyze_growth(time, values, n_boot=0)

    assert result.preferred_model in set(get_args(PreferredModel))


def test_runtime_modelfit_names_are_in_literal_set():
    time, values = _logistic_series()
    result = analyze_growth(time, values, n_boot=0)

    expected = set(get_args(ModelName))
    assert result.exponential_fit.model_name in expected
    assert result.linear_fit.model_name in expected
    assert result.logistic_fit.model_name in expected
    assert result.power_law_fit.model_name in expected
