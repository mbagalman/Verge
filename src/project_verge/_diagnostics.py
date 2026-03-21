from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from ._fit import (
    exponential_curve,
    fit_exponential_model,
    fit_logistic_model,
    logistic_curve,
)
from ._types import Diagnostics, ModelFit


def build_diagnostics(
    time: np.ndarray,
    values: np.ndarray,
    *,
    exponential_fit: ModelFit,
    logistic_fit: ModelFit,
) -> Diagnostics:
    per_capita_intercept, per_capita_slope = _per_capita_regression(time, values)
    residual_curvature_score = _residual_curvature_score(time, values)
    exp_forecast_mae = _forward_chaining_mae(time, values, fit_exponential_model, min_train=5)
    log_forecast_mae = _forward_chaining_mae(time, values, fit_logistic_model, min_train=6)
    identifiability_warnings = _logistic_identifiability_warnings(time, values, logistic_fit)

    return Diagnostics(
        per_capita_slope=float(per_capita_slope),
        per_capita_intercept=float(per_capita_intercept),
        residual_curvature_score=float(residual_curvature_score),
        forecast_mae_exponential=float(exp_forecast_mae),
        forecast_mae_logistic=float(log_forecast_mae),
        fit_warnings=logistic_fit.warnings,
        identifiability_warnings=identifiability_warnings,
    )


def _per_capita_regression(time: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
    # This diagnostic depends on time differences, not on the absolute time origin,
    # so shifting the series to start at zero does not change the fitted slope/intercept.
    delta_t = np.diff(time)
    delta_y = np.diff(values)
    per_capita_growth = delta_y / (delta_t * values[:-1])
    levels = values[:-1]
    design = np.column_stack([np.ones_like(levels), levels])
    intercept, slope = np.linalg.lstsq(design, per_capita_growth, rcond=None)[0]
    return float(intercept), float(slope)


def _residual_curvature_score(time: np.ndarray, values: np.ndarray) -> float:
    span = max(float(time[-1] - time[0]), 1.0)
    scaled_time = (time - np.mean(time)) / span
    coeffs = np.polyfit(scaled_time, np.log(values), deg=2)
    return float(coeffs[0])


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
