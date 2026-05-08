from __future__ import annotations

import math
from typing import Callable, NamedTuple, Tuple

import numpy as np
from scipy import stats

from ._fit import (
    exponential_curve,
    fit_exponential_model,
    fit_linear_model,
    fit_logistic_model,
    linear_curve,
    logistic_curve,
)
from ._types import Diagnostics, ForecastDiagnostic, ModelFit, SignalAgreement

# One-sided p-value threshold for treating a supporting signal as
# "significantly negative". The threshold sits at the standard 0.05; future
# work could expose this as a user-facing parameter (see TICKETS T-14).
_SIGNAL_SIGNIFICANCE = 0.05


class _SlopeStats(NamedTuple):
    coef: float
    intercept: float
    std_err: float
    t_stat: float
    p_value: float  # one-sided H1: coef < 0


class _CurvatureStats(NamedTuple):
    coef: float
    std_err: float
    t_stat: float
    p_value: float  # one-sided H1: coef < 0


def build_diagnostics(
    time: np.ndarray,
    values: np.ndarray,
    *,
    exponential_fit: ModelFit,
    linear_fit: ModelFit,
    logistic_fit: ModelFit,
    leading_fit: ModelFit,
) -> Diagnostics:
    pc = _per_capita_regression(time, values)
    cv = _residual_curvature_score(time, values)
    exp_forecast = _forward_chaining_diagnostic(time, values, fit_exponential_model, min_train=5)
    lin_forecast = _forward_chaining_diagnostic(time, values, fit_linear_model, min_train=5)
    log_forecast = _forward_chaining_diagnostic(time, values, fit_logistic_model, min_train=6)
    identifiability_warnings = _logistic_identifiability_warnings(time, values, logistic_fit)

    signal_agreement = SignalAgreement(
        per_capita_slope_negative=pc.p_value < _SIGNAL_SIGNIFICANCE,
        residual_curvature_negative=cv.p_value < _SIGNAL_SIGNIFICANCE,
        logistic_has_best_forecast=_logistic_has_best_forecast(
            exp_forecast, lin_forecast, log_forecast
        ),
    )

    normality_p, autocorr_p = _check_log_normal_assumptions(
        values, leading_fit.fitted_values
    )
    assumption_warnings = _build_assumption_warnings(
        normality_p, autocorr_p, leading_model_name=leading_fit.model_name
    )

    return Diagnostics(
        per_capita_slope=pc.coef,
        per_capita_intercept=pc.intercept,
        per_capita_slope_std_err=pc.std_err,
        per_capita_slope_t_stat=pc.t_stat,
        per_capita_slope_p_value=pc.p_value,
        residual_curvature_score=cv.coef,
        residual_curvature_std_err=cv.std_err,
        residual_curvature_t_stat=cv.t_stat,
        residual_curvature_p_value=cv.p_value,
        forecast_exponential=exp_forecast,
        forecast_linear=lin_forecast,
        forecast_logistic=log_forecast,
        signal_agreement=signal_agreement,
        residual_normality_pvalue=normality_p,
        residual_autocorr_pvalue=autocorr_p,
        fit_warnings=logistic_fit.warnings,
        identifiability_warnings=identifiability_warnings,
        assumption_warnings=assumption_warnings,
    )


def _per_capita_regression(time: np.ndarray, values: np.ndarray) -> _SlopeStats:
    # This diagnostic depends on time differences, not on the absolute time origin,
    # so shifting the series to start at zero does not change the fitted slope/intercept.
    delta_t = np.diff(time)
    delta_y = np.diff(values)
    per_capita_growth = delta_y / (delta_t * values[:-1])
    levels = values[:-1]
    n = len(per_capita_growth)
    design = np.column_stack([np.ones_like(levels), levels])
    coeffs, *_ = np.linalg.lstsq(design, per_capita_growth, rcond=None)
    intercept, slope = coeffs

    predictions = design @ coeffs
    residuals = per_capita_growth - predictions
    df = max(n - 2, 1)
    residual_var = float(np.sum(residuals**2) / df)
    XtX_inv = np.linalg.pinv(design.T @ design)
    slope_var = max(residual_var * float(XtX_inv[1, 1]), 0.0)
    slope_std_err = math.sqrt(slope_var)

    if slope_std_err > 0.0 and math.isfinite(slope_std_err):
        t_stat = float(slope / slope_std_err)
        p_value = float(stats.t.cdf(t_stat, df=df))
    else:
        t_stat = float("nan")
        p_value = 1.0

    return _SlopeStats(
        coef=float(slope),
        intercept=float(intercept),
        std_err=float(slope_std_err),
        t_stat=t_stat,
        p_value=p_value,
    )


def _residual_curvature_score(time: np.ndarray, values: np.ndarray) -> _CurvatureStats:
    span = max(float(time[-1] - time[0]), 1.0)
    scaled_time = (time - np.mean(time)) / span
    log_values = np.log(values)
    n = len(log_values)
    # ``np.polyfit`` returns coefficients highest-power-first and a covariance
    # matrix (already scaled by residual variance) when ``cov=True``.
    coeffs, cov = np.polyfit(scaled_time, log_values, deg=2, cov=True)
    quadratic_coef = float(coeffs[0])
    quadratic_var = max(float(cov[0, 0]), 0.0)
    quadratic_std_err = math.sqrt(quadratic_var)
    df = max(n - 3, 1)

    if quadratic_std_err > 0.0 and math.isfinite(quadratic_std_err):
        t_stat = float(quadratic_coef / quadratic_std_err)
        p_value = float(stats.t.cdf(t_stat, df=df))
    else:
        t_stat = float("nan")
        p_value = 1.0

    return _CurvatureStats(
        coef=quadratic_coef,
        std_err=float(quadratic_std_err),
        t_stat=t_stat,
        p_value=p_value,
    )


def _check_log_normal_assumptions(
    values: np.ndarray, fitted_values: np.ndarray
) -> Tuple[float, float]:
    """Test the leading model's log-residuals against the package's stated
    log-normal observation assumption.

    Returns ``(normality_pvalue, autocorr_pvalue)``. Each is in ``[0, 1]`` when
    the corresponding test was applicable, or NaN when residual variance is
    too small for the test to be meaningful (typical of perfectly clean v1
    inputs, where log-residuals are at floating-point noise floor).
    """

    log_values = np.log(values)
    log_fitted = np.log(np.clip(fitted_values, np.finfo(float).tiny, None))
    residuals = log_values - log_fitted
    n = len(residuals)
    nan = float("nan")

    # Both tests need a non-degenerate residual distribution. Clean v1 inputs
    # produce log-residuals dominated by numerical-optimizer noise (variance
    # in the 1e-17 to 1e-30 range); real-data residuals on noisy inputs sit
    # well above 1e-3. The 1e-12 floor cleanly separates the two so we
    # don't waste tests (and emit spurious warnings) on optimizer jitter.
    residual_var = float(np.var(residuals))
    if residual_var <= 1e-12:
        return nan, nan

    normality_p = nan
    if n >= 3:
        try:
            _, normality_p_raw = stats.shapiro(residuals)
            normality_p = float(normality_p_raw)
        except Exception:
            normality_p = nan

    autocorr_p = nan
    h = min(10, max(1, n // 4))
    if n > h + 1:
        try:
            autocorr_p = _ljung_box_pvalue(residuals, h)
        except Exception:
            autocorr_p = nan

    return normality_p, autocorr_p


def _ljung_box_pvalue(residuals: np.ndarray, h: int) -> float:
    """Ljung-Box one-sided p-value for H0 of no autocorrelation up to lag ``h``.

    Computed by hand (rather than via statsmodels) to keep the dependency
    footprint to numpy + scipy.
    """

    n = len(residuals)
    centered = residuals - np.mean(residuals)
    denom = float(np.sum(centered ** 2))
    if denom <= 0.0:
        return float("nan")

    statistic = 0.0
    for k in range(1, h + 1):
        autocov_k = float(np.sum(centered[:-k] * centered[k:]))
        rho_k = autocov_k / denom
        statistic += rho_k * rho_k / (n - k)
    statistic *= n * (n + 2)
    return float(stats.chi2.sf(statistic, df=h))


def _build_assumption_warnings(
    normality_p: float,
    autocorr_p: float,
    *,
    leading_model_name: str,
) -> Tuple[str, ...]:
    """Format human-readable warnings when log-normal assumption tests fail."""

    warnings: list = []
    if math.isfinite(normality_p) and normality_p < _SIGNAL_SIGNIFICANCE:
        warnings.append(
            f"{leading_model_name} log-residuals fail Shapiro-Wilk "
            f"(p = {normality_p:.3g}); the log-normal observation model is "
            f"in question, so BIC/AICc weights and bootstrap intervals may "
            f"be biased."
        )
    if math.isfinite(autocorr_p) and autocorr_p < _SIGNAL_SIGNIFICANCE:
        warnings.append(
            f"{leading_model_name} log-residuals show serial correlation "
            f"(Ljung-Box p = {autocorr_p:.3g}); effective sample size is "
            f"smaller than n, so the criterion penalty is too lenient."
        )
    return tuple(warnings)


def _logistic_has_best_forecast(
    exp_forecast: ForecastDiagnostic,
    lin_forecast: ForecastDiagnostic,
    log_forecast: ForecastDiagnostic,
) -> bool:
    """True iff logistic has the lowest median log error among the three.

    Models with no converged windows (median NaN) are filtered out -- a
    model that failed to forecast cannot be the "best" forecaster, even
    if everything else fared worse. Ties are resolved by the order
    candidates are listed (alphabetical), which matters only on
    pathologically symmetric data."""
    candidates = [
        (exp_forecast.median_log_error, "exponential"),
        (lin_forecast.median_log_error, "linear"),
        (log_forecast.median_log_error, "logistic"),
    ]
    finite = [(m, name) for m, name in candidates if math.isfinite(m)]
    if not finite:
        return False
    finite.sort()
    return finite[0][1] == "logistic"


def _forward_chaining_diagnostic(
    time: np.ndarray,
    values: np.ndarray,
    fit_func: Callable[..., ModelFit],
    *,
    min_train: int,
) -> ForecastDiagnostic:
    """Roll forward through ``values`` fitting on each prefix and forecasting
    one step ahead, then aggregate.

    The previous version of this function used the *mean* of all forecast
    errors, marking failed-to-converge windows with ``inf``. A single
    failed window therefore poisoned the aggregate. The replacement uses
    the median of the converged windows' errors and exposes the
    convergence rate separately, so consumers can decide how much to
    trust the median given how many fits succeeded.

    Logistic has one more curve parameter than exponential, so callers pass
    a larger ``min_train`` for it to ensure the prefix has enough degrees
    of freedom for a logistic fit to be defined.
    """
    errors = []
    n_attempted = 0
    n_converged = 0

    for split_index in range(min_train, len(values)):
        n_attempted += 1
        train_time = time[:split_index]
        train_values = values[:split_index]
        try:
            fit = fit_func(train_time, train_values, min_points=min_train)
        except ValueError:
            continue
        if not fit.converged:
            continue

        future_time = np.array([time[split_index]])
        if fit.model_name == "exponential":
            prediction = exponential_curve(future_time, fit.parameters["a"], fit.parameters["r"])
        elif fit.model_name == "linear":
            prediction = linear_curve(future_time, fit.parameters["a"], fit.parameters["b"])
        else:
            prediction = logistic_curve(
                future_time,
                fit.parameters["K"],
                fit.parameters["r"],
                fit.parameters["t0"],
            )

        predicted_value = float(np.clip(prediction[0], np.finfo(float).tiny, None))
        error = abs(float(np.log(values[split_index])) - float(np.log(predicted_value)))
        if not math.isfinite(error):
            # Defensive: a wildly out-of-range prediction can produce a
            # non-finite error even though the fit "converged". Don't let
            # that contaminate the median.
            continue
        errors.append(error)
        n_converged += 1

    if errors:
        median_log_error = float(np.median(errors))
    else:
        median_log_error = float("nan")
    convergence_rate = (n_converged / n_attempted) if n_attempted > 0 else 0.0
    return ForecastDiagnostic(
        median_log_error=median_log_error,
        convergence_rate=float(convergence_rate),
        n_windows=n_attempted,
    )


def _logistic_identifiability_warnings(
    time: np.ndarray,
    values: np.ndarray,
    logistic_fit: ModelFit,
) -> Tuple[str, ...]:
    if not logistic_fit.parameters:
        return ("Logistic parameters were not identified.",)

    warnings = []
    span = max(float(time[-1] - time[0]), 1.0)
    observed_max = float(np.max(values))
    capacity_ratio = logistic_fit.parameters["K"] / observed_max
    midpoint = logistic_fit.parameters["t0"]
    observed_fraction_of_capacity = observed_max / logistic_fit.parameters["K"]

    if capacity_ratio > 100.0:
        warnings.append("Logistic carrying capacity is far above observed values.")
    if midpoint > time[-1] + 0.5 * span and observed_fraction_of_capacity < 0.25:
        warnings.append("Observed data does not reach enough of the logistic bend to identify saturation.")
    if midpoint > time[-1] + 2.0 * span:
        warnings.append("Logistic midpoint sits far beyond the observed window.")
    if midpoint < time[0] - 2.0 * span:
        warnings.append("Logistic midpoint sits far before the observed window.")
    if not logistic_fit.converged:
        warnings.append("Logistic optimization did not converge cleanly.")

    return tuple(warnings)
