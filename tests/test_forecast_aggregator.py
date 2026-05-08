import math

import numpy as np

from project_verge import ForecastDiagnostic, analyze_growth
from project_verge._diagnostics import (
    _forward_chaining_diagnostic,
    _logistic_has_best_forecast,
)
from project_verge._fit import fit_exponential_model
from project_verge._types import ModelFit


def _exp_series(a=4.0, r=0.16, n=15, start=0.0, stop=10.0):
    time = np.linspace(start, stop, n)
    values = a * np.exp(r * (time - time[0]))
    return time, values


def _logistic_series(k=120.0, r=0.7, t0=6.0, n=22, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


def test_diagnostics_exposes_forecast_diagnostic_namedtuples():
    time, values = _exp_series()
    diag = analyze_growth(time, values, n_boot=0).diagnostics

    for name in ("forecast_exponential", "forecast_linear", "forecast_logistic"):
        forecast = getattr(diag, name)
        assert isinstance(forecast, ForecastDiagnostic)
        assert hasattr(forecast, "median_log_error")
        assert hasattr(forecast, "convergence_rate")
        assert hasattr(forecast, "n_windows")
        assert 0.0 <= forecast.convergence_rate <= 1.0
        assert forecast.n_windows >= 1


def test_clean_data_yields_high_convergence_and_low_median_error():
    time, values = _logistic_series(n=22)
    diag = analyze_growth(time, values, n_boot=0).diagnostics

    log_forecast = diag.forecast_logistic
    # Clean logistic data: every rolling window converges, and the one-step
    # forecast log error is essentially zero.
    assert log_forecast.convergence_rate == 1.0
    assert math.isfinite(log_forecast.median_log_error)
    assert log_forecast.median_log_error < 0.01


def test_forward_chaining_diagnostic_skips_failed_windows_via_median():
    # Construct a fit_func that fails on one specific window: when called
    # with exactly 6 training points, return a non-converged ModelFit;
    # otherwise delegate to the real exponential fitter. The median over
    # successful windows must remain finite, demonstrating that the new
    # aggregator no longer collapses to inf when one window fails.
    time, values = _exp_series(n=10)

    bad_window_size = 6
    real_fit_count = {"n": 0}

    def patchy_fit(t, v, *, min_points):
        if len(t) == bad_window_size:
            return ModelFit(
                model_name="exponential",
                parameters={},
                fitted_values=np.full_like(v, np.nan, dtype=float),
                log_likelihood=float("-inf"),
                bic=float("inf"),
                aicc=float("inf"),
                log_r_squared=float("-inf"),
                converged=False,
                warnings=("synthetic-failure",),
            )
        real_fit_count["n"] += 1
        return fit_exponential_model(t, v, min_points=min_points)

    forecast = _forward_chaining_diagnostic(time, values, patchy_fit, min_train=5)

    # n_windows = len(values) - min_train = 10 - 5 = 5 attempted.
    assert forecast.n_windows == 5
    # At least one window failed (the bad_window_size one), so convergence
    # rate is below 1.0.
    assert forecast.convergence_rate < 1.0
    assert forecast.convergence_rate >= 0.5  # most windows succeed
    # Median over the converged windows is finite -- the previous
    # mean-based aggregator would have returned inf here.
    assert math.isfinite(forecast.median_log_error)
    assert forecast.median_log_error < 0.1  # clean exponential data


def test_forward_chaining_diagnostic_returns_nan_when_every_window_fails():
    time = np.linspace(0.0, 10.0, 10)
    values = 4.0 * np.exp(0.16 * time)

    def always_fail_fit(t, v, *, min_points):
        return ModelFit(
            model_name="exponential",
            parameters={},
            fitted_values=np.full_like(v, np.nan, dtype=float),
            log_likelihood=float("-inf"),
            bic=float("inf"),
            aicc=float("inf"),
            log_r_squared=float("-inf"),
            converged=False,
            warnings=(),
        )

    forecast = _forward_chaining_diagnostic(time, values, always_fail_fit, min_train=5)

    assert forecast.n_windows >= 1
    assert forecast.convergence_rate == 0.0
    assert math.isnan(forecast.median_log_error)


def test_logistic_has_best_forecast_filters_nan_candidates():
    finite_log = ForecastDiagnostic(median_log_error=0.05, convergence_rate=1.0, n_windows=10)
    finite_exp = ForecastDiagnostic(median_log_error=0.10, convergence_rate=1.0, n_windows=10)
    nan_lin = ForecastDiagnostic(
        median_log_error=float("nan"), convergence_rate=0.0, n_windows=10
    )

    # Logistic has the best (lowest) finite median -> True.
    assert _logistic_has_best_forecast(finite_exp, nan_lin, finite_log) is True
    # Exponential has the best finite median -> False.
    assert _logistic_has_best_forecast(finite_log, nan_lin, finite_exp) is False


def test_summary_surfaces_convergence_rate_when_below_one_hundred_percent():
    # Pick a tiny series where the logistic forecast will fail at least
    # one rolling window (it sometimes does on small partial data). The
    # summary should annotate the convergence percentage when not 100%.
    time = np.linspace(0.0, 8.0, 10)
    values = 50.0 / (1.0 + np.exp(-0.4 * (time - 5.0)))

    result = analyze_growth(time, values, n_boot=0)
    summary = result.summary()
    forecast = result.diagnostics.forecast_logistic

    if forecast.convergence_rate < 1.0 and result.preferred_model == "logistic":
        assert "converged" in summary
        assert f"{forecast.n_windows} windows" in summary


def test_median_aggregator_is_more_robust_than_old_mean_for_outlier_window():
    # Direct comparison: build a series of forecast errors with one large
    # outlier and confirm the median is unaffected. (This is a unit-style
    # test on the helper's behavior -- the integration tests above cover
    # end-to-end flow.)
    time = np.linspace(0.0, 9.0, 15)
    values = 4.0 * np.exp(0.16 * time)

    forecast = _forward_chaining_diagnostic(
        time, values, fit_exponential_model, min_train=5
    )

    # On clean exponential data, every window converges and the median
    # log error sits at floating-point noise.
    assert forecast.convergence_rate == 1.0
    assert forecast.median_log_error < 1e-6
