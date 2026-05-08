import math

import numpy as np

from project_verge import analyze_growth
from project_verge._diagnostics import _ljung_box_pvalue


def _exp_series(a=4.0, r=0.16, n=15, start=0.0, stop=10.0):
    time = np.linspace(start, stop, n)
    values = a * np.exp(r * (time - time[0]))
    return time, values


def _logistic_series(k=120.0, r=0.7, t0=6.0, n=22, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


def test_diagnostics_exposes_residual_pvalue_fields():
    time, values = _logistic_series(n=22)
    diag = analyze_growth(time, values, n_boot=0).diagnostics

    assert hasattr(diag, "residual_normality_pvalue")
    assert hasattr(diag, "residual_autocorr_pvalue")
    assert hasattr(diag, "assumption_warnings")


def test_assumption_checks_skip_when_residuals_are_at_numerical_floor():
    # Clean v1 inputs produce log-residuals dominated by optimizer noise
    # (variance ~ 1e-17). Tests must report NaN p-values and emit no
    # warnings rather than over-fire on numerical artifacts.
    time, values = _logistic_series(n=22)
    diag = analyze_growth(time, values, n_boot=0).diagnostics

    assert math.isnan(diag.residual_normality_pvalue)
    assert math.isnan(diag.residual_autocorr_pvalue)
    assert diag.assumption_warnings == ()


def test_clean_exponential_assumption_checks_skip_too():
    time, values = _exp_series(n=15)
    diag = analyze_growth(time, values, n_boot=0).diagnostics

    assert math.isnan(diag.residual_normality_pvalue)
    assert math.isnan(diag.residual_autocorr_pvalue)
    assert diag.assumption_warnings == ()


def test_step_change_input_triggers_autocorrelation_warning():
    # Flat-then-exponential is a structural misfit for every candidate.
    # The leading model's log-residuals have a clear positive run on one
    # half and a clear negative run on the other -- exactly the pattern
    # Ljung-Box is built to catch.
    time = np.linspace(0.0, 10.0, 16)
    flat = np.full(8, 1.0)
    expo = 1.0 * np.exp(0.5 * (time[8:] - time[8]))
    values = np.concatenate([flat, expo])

    diag = analyze_growth(time, values, n_boot=0).diagnostics

    assert math.isfinite(diag.residual_autocorr_pvalue)
    assert diag.residual_autocorr_pvalue < 0.05
    assert any(
        "serial correlation" in warning for warning in diag.assumption_warnings
    )


def test_world_population_exponential_fit_shows_autocorrelated_residuals():
    # Real data: world-population estimates 1750-2022 fit by exponential
    # have a visible smile/frown pattern in the log-residuals (the curve
    # accelerates faster than constant exponential rate). The Ljung-Box
    # test catches it.
    import csv
    from pathlib import Path

    csv_path = Path("examples/data/un_population.csv")
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    years = np.array([float(r["year"]) for r in rows])
    pop = np.array([float(r["population_billions"]) for r in rows])

    diag = analyze_growth(years, pop, n_boot=0).diagnostics

    assert math.isfinite(diag.residual_autocorr_pvalue)
    assert diag.residual_autocorr_pvalue < 0.05
    assert any(
        "serial correlation" in warning for warning in diag.assumption_warnings
    )


def test_ljung_box_helper_returns_high_p_for_white_noise():
    rng = np.random.default_rng(0)
    residuals = rng.standard_normal(200)

    p_value = _ljung_box_pvalue(residuals, h=10)

    # Genuine i.i.d. noise should not look autocorrelated (p well above 0.05).
    assert p_value > 0.05


def test_ljung_box_helper_returns_low_p_for_strongly_autocorrelated_series():
    # AR(1) with phi = 0.8 -- residuals are obviously serially correlated.
    rng = np.random.default_rng(0)
    n = 200
    eps = rng.standard_normal(n)
    series = np.zeros(n)
    for i in range(1, n):
        series[i] = 0.8 * series[i - 1] + eps[i]

    p_value = _ljung_box_pvalue(series, h=10)
    assert p_value < 0.001
