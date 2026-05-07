import numpy as np
import pytest

from project_verge import analyze_growth, fit_power_law


def _exp_series(a=4.0, r=0.16, n=15, start=0.0, stop=10.0):
    time = np.linspace(start, stop, n)
    values = a * np.exp(r * (time - time[0]))
    return time, values


def _logistic_series(k=120.0, r=0.7, t0=6.0, n=22, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


def test_fit_power_law_recovers_exponent_on_clean_power_law_data():
    time = np.linspace(0.0, 9.0, 20)
    # y = 2 * (t + 1)^1.7 -- the +1 shift matches the fitter's internal offset.
    values = 2.0 * (time + 1.0) ** 1.7
    fit = fit_power_law(time, values)

    assert fit.converged
    assert fit.parameters["a"] == pytest.approx(2.0, rel=1e-6)
    assert fit.parameters["k"] == pytest.approx(1.7, rel=1e-6)
    assert fit.log_r_squared > 0.999


@pytest.mark.parametrize("k_true", [0.5, 1.5, 2.0, 3.0])
def test_polynomial_growth_classifies_as_power_law_shape_at_default_threshold(k_true):
    # Acceptance criterion: cubic and square-root-style series force the
    # power_law_shape indeterminate reason at the *default* min_fit_quality.
    time = np.linspace(1.0, 12.0, 18)
    values = time ** k_true

    result = analyze_growth(time, values, n_boot=0)

    assert result.is_indeterminate
    assert result.indeterminate_reason == "power_law_shape"
    assert result.p_power_law > 0.9


def test_clean_logistic_does_not_lose_to_power_law():
    time, values = _logistic_series(k=100.0, r=0.75, t0=5.0, n=22)

    result = analyze_growth(time, values, n_boot=0)

    assert result.preferred_model == "logistic"
    assert result.indeterminate_reason is None
    assert result.p_logistic > 0.95
    assert result.p_power_law < 0.05


def test_clean_exponential_does_not_lose_to_power_law():
    time, values = _exp_series(a=3.0, r=0.18, n=18)

    result = analyze_growth(time, values, n_boot=0)

    assert result.preferred_model == "exponential"
    assert result.indeterminate_reason is None
    assert result.p_exponential > 0.95
    assert result.p_power_law < 0.05


def test_clean_linear_does_not_lose_to_power_law():
    # Asymptotically y = a + b*t looks like y ~ t^1, so power-law could
    # plausibly steal weight from linear. Verify it does not.
    time = np.linspace(0.0, 10.0, 16)
    values = 5.0 + 2.0 * time

    result = analyze_growth(time, values, n_boot=0)

    assert result.preferred_model == "linear"
    assert result.indeterminate_reason is None
    assert result.p_linear > 0.95
    assert result.p_power_law < 0.05


def test_growth_analysis_exposes_power_law_fit_and_weight():
    time, values = _exp_series()
    result = analyze_growth(time, values, n_boot=0)

    assert hasattr(result, "p_power_law")
    assert 0.0 <= result.p_power_law <= 1.0
    assert result.power_law_fit.model_name == "power_law"
    assert "a" in result.power_law_fit.parameters
    assert "k" in result.power_law_fit.parameters


def test_weight_intervals_include_power_law_when_bootstrap_runs():
    time, values = _logistic_series(n=20)
    result = analyze_growth(time, values, n_boot=100, bootstrap_seed=0)

    assert result.weight_intervals is not None
    assert hasattr(result.weight_intervals, "p_power_law")
    interval = result.weight_intervals.p_power_law
    assert 0.0 <= interval.low <= interval.median <= interval.high <= 1.0
