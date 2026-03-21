from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from ._types import ModelFit

ArrayPair = Tuple[np.ndarray, np.ndarray]

_TINY = np.finfo(float).tiny


def prepare_inputs(
    time: Sequence[float],
    values: Sequence[float],
    *,
    min_points: int,
) -> ArrayPair:
    """Validate the user-facing data contract and normalize time to zero."""

    time_array = np.asarray(time, dtype=float)
    value_array = np.asarray(values, dtype=float)

    if time_array.ndim != 1 or value_array.ndim != 1:
        raise ValueError("time and values must be one-dimensional sequences")
    if len(time_array) != len(value_array):
        raise ValueError("time and values must have the same length")
    if len(time_array) < min_points:
        raise ValueError(f"at least {min_points} observations are required")
    if not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(value_array)):
        raise ValueError("time and values must contain only finite numbers")
    if np.any(np.diff(time_array) <= 0.0):
        raise ValueError("time must be strictly increasing")
    if np.any(value_array <= 0.0):
        raise ValueError("values must be strictly positive")
    if np.any(np.diff(value_array) < 0.0):
        raise ValueError("values must be nondecreasing for the v1 API")

    normalized_time = time_array - time_array[0]
    return normalized_time, value_array


def exponential_curve(time: np.ndarray, a: float, r: float) -> np.ndarray:
    exponent = np.clip(r * time, -700.0, 700.0)
    return a * np.exp(exponent)


def logistic_curve(time: np.ndarray, k: float, r: float, t0: float) -> np.ndarray:
    return k * expit(r * (time - t0))


def fit_exponential_model(
    time: np.ndarray,
    values: np.ndarray,
    *,
    min_points: int,
) -> ModelFit:
    return _fit_model(
        time,
        values,
        model_name="exponential",
        parameter_names=("a", "r"),
        model_func=lambda t, p: exponential_curve(t, p[0], p[1]),
        initial_guess=_initial_guess_exponential(time, values),
        bounds=_bounds_exponential(time, values),
        min_points=min_points,
    )


def fit_logistic_model(
    time: np.ndarray,
    values: np.ndarray,
    *,
    min_points: int,
) -> ModelFit:
    return _fit_model(
        time,
        values,
        model_name="logistic",
        parameter_names=("K", "r", "t0"),
        model_func=lambda t, p: logistic_curve(t, p[0], p[1], p[2]),
        initial_guess=_initial_guess_logistic(time, values),
        bounds=_bounds_logistic(time, values),
        min_points=min_points,
    )


def _fit_model(
    time: np.ndarray,
    values: np.ndarray,
    *,
    model_name: str,
    parameter_names: Tuple[str, ...],
    model_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    initial_guess: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    min_points: int,
) -> ModelFit:
    if len(time) < min_points:
        raise ValueError(f"{model_name} fitting requires at least {min_points} points")

    warnings = []

    def residuals(params: np.ndarray) -> np.ndarray:
        prediction = np.clip(model_func(time, params), _TINY, None)
        return np.log(values) - np.log(prediction)

    try:
        result = least_squares(
            residuals,
            x0=initial_guess,
            bounds=bounds,
            method="trf",
            max_nfev=20000,
        )
        fitted_values = np.clip(model_func(time, result.x), _TINY, None)
        resids = np.log(values) - np.log(fitted_values)
        rss = float(np.sum(resids**2))
        sigma2 = max(rss / len(values), 1e-12)
        log_likelihood = -0.5 * len(values) * (np.log(2.0 * np.pi * sigma2) + 1.0)
        parameter_count = len(parameter_names) + 1
        bic = parameter_count * np.log(len(values)) - 2.0 * log_likelihood
        if not result.success:
            warnings.append(result.message)
        return ModelFit(
            model_name=model_name,
            parameters={name: float(value) for name, value in zip(parameter_names, result.x)},
            fitted_values=fitted_values,
            log_likelihood=float(log_likelihood),
            bic=float(bic),
            converged=bool(result.success),
            warnings=tuple(warnings),
        )
    except Exception as exc:
        warnings.append(f"{model_name} fit failed: {exc}")
        return ModelFit(
            model_name=model_name,
            parameters={},
            fitted_values=np.full_like(values, np.nan, dtype=float),
            log_likelihood=float("-inf"),
            bic=float("inf"),
            converged=False,
            warnings=tuple(warnings),
        )


def _initial_guess_exponential(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    span = max(float(time[-1] - time[0]), 1.0)
    growth_rate = max((np.log(values[-1]) - np.log(values[0])) / span, 1e-4)
    return np.array([values[0], growth_rate], dtype=float)


def _bounds_exponential(time: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    span = max(float(time[-1] - time[0]), 1.0)
    max_value = max(float(np.max(values)), 1.0)
    lower = np.array([1e-12, 1e-12], dtype=float)
    upper = np.array([max_value * 1e4, max(10.0, 25.0 / span)], dtype=float)
    return lower, upper


def _initial_guess_logistic(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    span = max(float(time[-1] - time[0]), 1.0)
    growth_rate = max((np.log(values[-1]) - np.log(values[0])) / span, 1e-4)
    carrying_capacity = max(float(np.max(values)) * 2.0, values[-1] + 1e-6)
    midpoint = float(time[len(time) // 2])
    return np.array([carrying_capacity, growth_rate, midpoint], dtype=float)


def _bounds_logistic(time: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    span = max(float(time[-1] - time[0]), 1.0)
    max_value = max(float(np.max(values)), 1.0)
    lower = np.array([max_value * (1.0 + 1e-6), 1e-12, time[0] - 4.0 * span], dtype=float)
    upper = np.array([max_value * 1e4, max(10.0, 25.0 / span), time[-1] + 4.0 * span], dtype=float)
    return lower, upper

