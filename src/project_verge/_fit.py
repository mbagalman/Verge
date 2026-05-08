from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from ._types import ModelFit

ArrayPair = Tuple[np.ndarray, np.ndarray]

_TINY = np.finfo(float).tiny


def smooth_to_monotone(
    values: Sequence[float],
    *,
    window: int = 3,
) -> np.ndarray:
    """Rolling-median smoother followed by cumulative-max enforcement.

    The rolling median dampens small noisy excursions; the cumulative-max
    pass guarantees the result satisfies Verge's nondecreasing input
    contract. The combination is parameter-free aside from ``window``,
    robust to point outliers (median is breakdown 50%), and produces
    output of the same length as the input.

    Trade-off: a genuine real-world *decrease* in the underlying process
    is mapped to a flat segment by ``cumulative-max``, biasing the fit
    upward. For series where you expect occasional dips that should be
    preserved, do your own pre-processing rather than relying on this
    helper.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing window must be a positive odd integer")
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("values must be a one-dimensional sequence")
    n = len(arr)
    half = window // 2
    smoothed = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed[i] = float(np.median(arr[lo:hi]))
    return np.maximum.accumulate(smoothed)


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


def linear_curve(time: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * time


def power_law_curve(time: np.ndarray, a: float, k: float) -> np.ndarray:
    """Evaluate ``y = a * (t + 1)**k`` in normalized time.

    The ``+1`` shift matches what :func:`fit_power_law_model` uses, so
    ``log(t)`` stays finite at the normalized origin ``t = 0``. Power-law
    is a diagnostic-only candidate in v1; this curve helper exists for
    completeness (and for any future plot or prediction usage) but is not
    consulted by :meth:`GrowthAnalysis.predict` because power-law never
    becomes the preferred verdict.
    """
    return a * np.power(time + 1.0, k)


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


def fit_linear_model(
    time: np.ndarray,
    values: np.ndarray,
    *,
    min_points: int,
) -> ModelFit:
    return _fit_model(
        time,
        values,
        model_name="linear",
        parameter_names=("a", "b"),
        model_func=lambda t, p: linear_curve(t, p[0], p[1]),
        initial_guess=_initial_guess_linear(time, values),
        bounds=_bounds_linear(time, values),
        min_points=min_points,
    )


def fit_power_law_model(
    time: np.ndarray,
    values: np.ndarray,
    *,
    min_points: int,
) -> ModelFit:
    """Fit ``y = a * (t + 1)**k`` via OLS in log-log space.

    A diagnostic-only candidate. The ``+1`` shift on time keeps ``log(t)``
    finite at ``t = 0`` after the package's standard time-origin
    normalization. Verge does not use power-law for prediction or for the
    headline verdict; it competes on BIC only so that polynomial /
    power-law growth has somewhere honest to land instead of silently
    misclassifying as logistic.
    """
    if len(time) < min_points:
        raise ValueError(
            f"power-law fitting requires at least {min_points} points"
        )

    n = len(values)
    log_y = np.log(values)
    log_t = np.log(time + 1.0)

    design = np.column_stack([np.ones(n), log_t])
    coeffs, *_ = np.linalg.lstsq(design, log_y, rcond=None)
    log_a = float(coeffs[0])
    k = float(coeffs[1])
    a = float(np.exp(log_a))

    fitted_log_y = log_a + k * log_t
    fitted_values = np.clip(np.exp(fitted_log_y), _TINY, None)
    rss = float(np.sum((log_y - fitted_log_y) ** 2))
    sigma2 = max(rss / n, 1e-12)
    log_likelihood = -0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0)
    # Two curve parameters plus the shared observation-noise scale, matching
    # the information-criterion bookkeeping used by the other model fits.
    parameter_count = 2 + 1
    bic = parameter_count * np.log(n) - 2.0 * log_likelihood
    aicc = _aicc(log_likelihood, parameter_count, n)
    log_r_squared = _log_space_r_squared(values, rss)

    return ModelFit(
        model_name="power_law",
        parameters={"a": a, "k": k},
        fitted_values=fitted_values,
        log_likelihood=float(log_likelihood),
        bic=float(bic),
        aicc=float(aicc),
        log_r_squared=float(log_r_squared),
        converged=True,
        warnings=(),
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
    # Internal callers also use these fitters directly on smaller rolling windows,
    # so we keep the minimum-length guard here instead of relying only on prepare_inputs.
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
        # Count the observation-noise scale parameter alongside the curve parameters
        # so the information criteria reflect the full log-normal observation model.
        parameter_count = len(parameter_names) + 1
        n = len(values)
        bic = parameter_count * np.log(n) - 2.0 * log_likelihood
        aicc = _aicc(log_likelihood, parameter_count, n)
        log_r_squared = _log_space_r_squared(values, rss)
        if not result.success:
            warnings.append(result.message)
        return ModelFit(
            model_name=model_name,
            parameters={name: float(value) for name, value in zip(parameter_names, result.x)},
            fitted_values=fitted_values,
            log_likelihood=float(log_likelihood),
            bic=float(bic),
            aicc=float(aicc),
            log_r_squared=float(log_r_squared),
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
            aicc=float("inf"),
            log_r_squared=float("-inf"),
            converged=False,
            warnings=tuple(warnings),
        )


def _log_space_r_squared(values: np.ndarray, rss: float) -> float:
    log_values = np.log(values)
    tss = float(np.sum((log_values - np.mean(log_values)) ** 2))
    if tss <= 0.0:
        # All observations equal on the log scale; any sensible fit is perfect.
        return 1.0
    return 1.0 - rss / tss


def _aicc(log_likelihood: float, parameter_count: int, n: int) -> float:
    """Akaike Information Criterion with the standard small-sample correction.

    AICc = AIC + 2k(k+1)/(n - k - 1). When ``n - k - 1 <= 0`` the correction
    blows up; we return +inf so the model gets zero weight in the AICc-based
    competition. This corner case is reachable only in the bootstrap path
    with ``min_points = _BOOTSTRAP_MIN_POINTS = 4`` and a four-parameter
    fit (logistic with the noise scale).
    """
    aic = 2.0 * parameter_count - 2.0 * log_likelihood
    denom = n - parameter_count - 1
    if denom <= 0:
        return float("inf")
    return aic + 2.0 * parameter_count * (parameter_count + 1) / denom


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
    observed_max = float(np.max(values))
    carrying_capacity = max(observed_max * 1.5, values[-1] + max(observed_max - values[0], observed_max * 0.1))
    half_capacity = 0.5 * carrying_capacity
    if values[-1] < half_capacity:
        midpoint = float(time[-1] + 0.5 * span)
    else:
        midpoint = float(time[np.argmin(np.abs(values - half_capacity))])
    return np.array([carrying_capacity, growth_rate, midpoint], dtype=float)


def _bounds_logistic(time: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    span = max(float(time[-1] - time[0]), 1.0)
    max_value = max(float(np.max(values)), 1.0)
    # K is the logistic ceiling, so it must remain just above the observed maximum.
    lower = np.array([max_value * (1.0 + 1e-6), 1e-12, time[0] - 4.0 * span], dtype=float)
    upper = np.array([max_value * 1e4, max(10.0, 25.0 / span), time[-1] + 4.0 * span], dtype=float)
    return lower, upper


def _initial_guess_linear(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    span = max(float(time[-1] - time[0]), 1.0)
    slope = max((float(values[-1]) - float(values[0])) / span, 0.0)
    intercept = max(float(values[0]), 1e-6)
    return np.array([intercept, slope], dtype=float)


def _bounds_linear(time: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    span = max(float(time[-1] - time[0]), 1.0)
    max_value = max(float(np.max(values)), 1.0)
    # The intercept lives in y-units and must be positive so log(a + b*t) stays
    # finite under the shared log-normal observation model. The slope is
    # nonnegative because the v1 input contract is nondecreasing.
    lower = np.array([1e-12, 0.0], dtype=float)
    upper = np.array([max_value * 1e4, max_value * 1e4 / span], dtype=float)
    return lower, upper
