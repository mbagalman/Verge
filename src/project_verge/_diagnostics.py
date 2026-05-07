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
from ._types import Diagnostics, ModelFit, SignalAgreement

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
) -> Diagnostics:
    pc = _per_capita_regression(time, values)
    cv = _residual_curvature_score(time, values)
    exp_forecast_mae = _forward_chaining_mae(time, values, fit_exponential_model, min_train=5)
    lin_forecast_mae = _forward_chaining_mae(time, values, fit_linear_model, min_train=5)
    log_forecast_mae = _forward_chaining_mae(time, values, fit_logistic_model, min_train=6)
    identifiability_warnings = _logistic_identifiability_warnings(time, values, logistic_fit)

    signal_agreement = SignalAgreement(
        per_capita_slope_negative=pc.p_value < _SIGNAL_SIGNIFICANCE,
        residual_curvature_negative=cv.p_value < _SIGNAL_SIGNIFICANCE,
        logistic_has_best_forecast=_logistic_has_best_forecast(
            exp_forecast_mae, lin_forecast_mae, log_forecast_mae
        ),
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
        forecast_mae_exponential=float(exp_forecast_mae),
        forecast_mae_linear=float(lin_forecast_mae),
        forecast_mae_logistic=float(log_forecast_mae),
        signal_agreement=signal_agreement,
        fit_warnings=logistic_fit.warnings,
        identifiability_warnings=identifiability_warnings,
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


def _logistic_has_best_forecast(
    exp_mae: float,
    lin_mae: float,
    log_mae: float,
) -> bool:
    candidates = [
        (exp_mae, "exponential"),
        (lin_mae, "linear"),
        (log_mae, "logistic"),
    ]
    finite = [(m, name) for m, name in candidates if math.isfinite(m)]
    if not finite:
        return False
    finite.sort()
    return finite[0][1] == "logistic"


def _forward_chaining_mae(
    time: np.ndarray,
    values: np.ndarray,
    fit_func: Callable[..., ModelFit],
    *,
    min_train: int,
) -> float:
    errors = []

    # Logistic has one more curve parameter than exponential, so we require one
    # additional training observation before attempting rolling forecasts.
    for split_index in range(min_train, len(values)):
        train_time = time[:split_index]
        train_values = values[:split_index]
        fit = fit_func(train_time, train_values, min_points=min_train)
        if not fit.converged:
            errors.append(np.inf)
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

        error = abs(np.log(values[split_index]) - np.log(np.clip(prediction[0], np.finfo(float).tiny, None)))
        errors.append(float(error))

    if not errors:
        return float("inf")
    return float(np.mean(errors))


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
