import numpy as np
import pytest

from project_verge import analyze_growth, fit_exponential, fit_linear, fit_logistic


def _exp_series(a=2.5, r=0.18, n=14, start=0.0, stop=10.0):
    time = np.linspace(start, stop, n)
    values = a * np.exp(r * (time - time[0]))
    return time, values


def _logistic_series(k=120.0, r=0.7, t0=6.0, n=18, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


def test_fit_exponential_recovers_parameters_on_clean_data():
    time, values = _exp_series(a=3.0, r=0.12, n=16)
    fit = fit_exponential(time, values)

    assert fit.converged
    assert fit.parameters["a"] == pytest.approx(3.0, rel=1e-2)
    assert fit.parameters["r"] == pytest.approx(0.12, rel=1e-2)


def test_fit_logistic_recovers_parameters_on_clean_data():
    time, values = _logistic_series(k=90.0, r=0.8, t0=5.0, n=24)
    fit = fit_logistic(time, values)

    assert fit.converged
    assert fit.parameters["K"] == pytest.approx(90.0, rel=5e-2)
    assert fit.parameters["r"] == pytest.approx(0.8, rel=5e-2)
    assert fit.parameters["t0"] == pytest.approx(5.0, rel=5e-2)


def test_fit_linear_recovers_parameters_on_clean_data():
    time = np.linspace(0.0, 10.0, 16)
    values = 5.0 + 2.0 * time
    fit = fit_linear(time, values)

    assert fit.converged
    assert fit.parameters["a"] == pytest.approx(5.0, rel=1e-2)
    assert fit.parameters["b"] == pytest.approx(2.0, rel=1e-2)


def test_analyze_growth_prefers_exponential_for_clear_exponential_series():
    time, values = _exp_series(a=4.0, r=0.16, n=15)
    result = analyze_growth(time, values)

    assert result.p_exponential > 0.80
    assert result.p_logistic < 0.20
    assert result.preferred_model == "exponential"
    assert not result.is_indeterminate
    assert result.indeterminate_reason is None


def test_analyze_growth_prefers_logistic_for_clear_logistic_series():
    time, values = _logistic_series(k=100.0, r=0.75, t0=5.0, n=22)
    result = analyze_growth(time, values)

    assert result.p_logistic > 0.80
    assert result.p_exponential < 0.20
    assert result.preferred_model == "logistic"
    assert not result.is_indeterminate
    assert result.indeterminate_reason is None


def test_analyze_growth_prefers_linear_for_clean_linear_series():
    time = np.linspace(0.0, 10.0, 16)
    values = 5.0 + 2.0 * time
    result = analyze_growth(time, values)

    assert result.p_linear > 0.80
    assert result.preferred_model == "linear"
    assert not result.is_indeterminate
    assert result.indeterminate_reason is None


def test_analyze_growth_marks_early_logistic_as_indeterminate():
    time, values = _logistic_series(k=200.0, r=0.35, t0=12.0, n=10, start=0.0, stop=5.0)
    result = analyze_growth(time, values)

    assert result.is_indeterminate
    assert result.preferred_model == "indeterminate"
    assert result.indeterminate_reason == "logistic_unidentifiable"
    assert result.diagnostics.identifiability_warnings


def test_analyze_growth_marks_polynomial_as_power_law_shape_at_default_threshold():
    # Cubic growth is neither exponential, linear, nor logistic. T-27 added a
    # power-law candidate so polynomial inputs now fire "power_law_shape" at
    # the default min_fit_quality (no manual tuning needed); pre-T-27 they
    # silently classified as logistic.
    time = np.linspace(1.0, 12.0, 16)
    values = time ** 3

    result = analyze_growth(time, values)

    assert result.is_indeterminate
    assert result.preferred_model == "indeterminate"
    assert result.indeterminate_reason == "power_law_shape"
    assert result.power_law_fit.log_r_squared > result.logistic_fit.log_r_squared
    assert result.p_power_law > 0.95


def test_analyze_growth_marks_step_change_as_neither_model_fits():
    # A flat-then-exponential series fits none of the four candidates well;
    # log-space R^2 stays below the default 0.85 floor across the board, so
    # the verdict is the structured "neither_model_fits" indeterminate.
    time = np.linspace(0.0, 10.0, 16)
    flat = np.full(8, 1.0)
    expo = 1.0 * np.exp(0.5 * (time[8:] - time[8]))
    values = np.concatenate([flat, expo])

    result = analyze_growth(time, values)

    assert result.is_indeterminate
    assert result.indeterminate_reason == "neither_model_fits"


def test_log_r_squared_is_near_one_for_clean_exponential():
    time, values = _exp_series(a=3.0, r=0.12, n=16)
    fit = fit_exponential(time, values)

    assert fit.log_r_squared > 0.999


@pytest.mark.parametrize("bad_quality", [float("nan"), float("inf"), 1.5])
def test_analysis_rejects_invalid_min_fit_quality(bad_quality):
    time, values = _exp_series()

    with pytest.raises(ValueError):
        analyze_growth(time, values, min_fit_quality=bad_quality)


def test_fit_functions_accept_custom_min_points():
    time = np.linspace(0.0, 4.0, 6)
    values = 2.0 * np.exp(0.2 * time)

    exponential_fit = fit_exponential(time, values, min_points=6)
    logistic_fit = fit_logistic(time, values, min_points=6)

    assert exponential_fit.converged
    assert logistic_fit.converged


def test_fit_functions_enforce_custom_min_points():
    time = np.linspace(0.0, 4.0, 5)
    values = 2.0 * np.exp(0.2 * time)

    with pytest.raises(ValueError, match="at least 6"):
        fit_exponential(time, values, min_points=6)

    with pytest.raises(ValueError, match="at least 6"):
        fit_logistic(time, values, min_points=6)


def test_internal_fit_model_invariant_guard_distinguishes_from_user_input_error():
    # Defense-in-depth: if a future internal caller forgets to validate
    # before calling the lower-level fit_*_model functions, they raise
    # RuntimeError with a "internal precondition violated" message --
    # clearly distinct from the user-facing ValueError that prepare_inputs
    # raises ("at least N observations are required"). This pins the
    # invariant introduced when T-22 deduplicated the min_points check.
    from project_verge._fit import fit_logistic_model, fit_power_law_model

    time = np.linspace(0.0, 3.0, 4)
    values = 2.0 * np.exp(0.2 * time)

    # Call the internal fit_*_model directly, requesting a min_points
    # larger than the array we passed.
    with pytest.raises(RuntimeError, match="internal precondition violated"):
        fit_logistic_model(time, values, min_points=8)

    with pytest.raises(RuntimeError, match="internal precondition violated"):
        fit_power_law_model(time, values, min_points=8)


@pytest.mark.parametrize("prior", [float("nan"), float("inf")])
def test_analysis_rejects_non_finite_priors(prior):
    time, values = _exp_series()

    with pytest.raises(ValueError, match="finite"):
        analyze_growth(time, values, prior_exponential=prior)


@pytest.mark.parametrize(
    ("time", "values", "message"),
    [
        ([0, 1, 2], [1, 2], "same length"),
        ([0, 0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8], "strictly increasing"),
        ([0, 1, 2, 3, 4, 5, 6, 7], [1, -1, 2, 3, 4, 5, 6, 7], "strictly positive"),
        ([0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 1.5, 3, 4, 5, 6, 7], "nondecreasing"),
        ([0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7], "at least 8"),
    ],
)
def test_input_validation_rejects_invalid_series(time, values, message):
    with pytest.raises(ValueError, match=message):
        analyze_growth(time, values)


def test_analysis_handles_large_time_origin_and_value_scale():
    time = np.linspace(2000.0, 2014.0, 15)
    values = 2.5e6 * np.exp(0.09 * (time - time[0]))
    result = analyze_growth(time, values)

    assert np.isfinite(result.p_exponential)
    assert np.isfinite(result.p_logistic)
    assert result.p_exponential > result.p_logistic


def test_fitted_values_are_read_only():
    time, values = _exp_series()
    fit = fit_exponential(time, values)

    with pytest.raises(ValueError):
        fit.fitted_values[0] = -1.0
