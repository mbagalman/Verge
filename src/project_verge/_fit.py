from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.special import expit

from ._types import ModelFit

ArrayPair = Tuple[np.ndarray, np.ndarray]

_TINY = np.finfo(float).tiny


def smooth_to_monotone(
    values: npt.ArrayLike,
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
    time: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    min_points: int,
    min_relative_range: float = 0.01,
) -> ArrayPair:
    """Validate the user-facing data contract and normalize time to zero.

    ``min_relative_range`` rejects data with no meaningful growth signal:
    if ``(max(values) - min(values)) / max(values) < min_relative_range``,
    the growth-vs-leveling-off question Verge exists to answer is
    ill-posed and the result would be a numerical artifact of which
    trivial-parameter fit wins on no information. Pass ``0`` to disable
    the check.
    """

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
        raise ValueError(
            "values are decreasing in places; Verge analyzes growth, not "
            "decline, so a decreasing series is outside its scope. Pass "
            "allow_smoothing=True to coerce noisy nondecreasing data."
        )
    if min_relative_range > 0.0:
        max_value = float(np.max(value_array))
        observed_range = max_value - float(np.min(value_array))
        if max_value > 0.0 and (observed_range / max_value) < min_relative_range:
            raise ValueError(
                f"no growth signal detected: values span only "
                f"{(observed_range / max_value):.4%} of their maximum "
                f"(threshold {min_relative_range:.2%}). Verge's "
                f"growth-vs-leveling-off question is ill-posed on data "
                f"this flat. Pass min_relative_range=0 to disable this check."
            )

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
        initial_guesses=[_initial_guess_exponential(time, values)],
        bounds=_bounds_exponential(time, values),
        min_points=min_points,
    )


def fit_logistic_model(
    time: np.ndarray,
    values: np.ndarray,
    *,
    min_points: int,
    n_starts: int = 1,
) -> ModelFit:
    """Fit a logistic curve, optionally with multi-start optimization.

    The logistic likelihood surface is multi-modal in (K, r, t0) -- in
    particular the K bound interacts badly with partial-S data, where the
    optimizer can land at the K upper bound from one initial guess and at
    a much better (lower-RSS) interior point from another. Setting
    ``n_starts > 1`` runs the optimizer from several diverse initial
    guesses and keeps the lowest-RSS solution. The bootstrap path uses
    the default ``n_starts=1`` because each resample is similar enough
    to the data that single-start works and 8x cost in a 500-iteration
    loop is wasteful.
    """
    return _fit_model(
        time,
        values,
        model_name="logistic",
        parameter_names=("K", "r", "t0"),
        model_func=lambda t, p: logistic_curve(t, p[0], p[1], p[2]),
        initial_guesses=_multi_start_guesses_logistic(time, values, n_starts=n_starts),
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
        initial_guesses=[_initial_guess_linear(time, values)],
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
        # Internal invariant guard. ``prepare_inputs`` enforces the user-facing
        # ``min_points`` check on the public path; the bootstrap loop in
        # _uncertainty.py and the forward-chaining loop in _diagnostics.py
        # ensure the precondition by construction (resample size = input size;
        # split_index >= min_train). If this fires, an internal caller is
        # violating the precondition -- it is not a user-input error.
        raise RuntimeError(
            f"fit_power_law_model: internal precondition violated -- "
            f"len(time)={len(time)} < min_points={min_points}. "
            "Callers must validate before invoking."
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
    initial_guesses,
    bounds: Tuple[np.ndarray, np.ndarray],
    min_points: int,
) -> ModelFit:
    # Internal invariant guard. ``prepare_inputs`` enforces the user-facing
    # ``min_points`` check on the public path; the bootstrap loop in
    # _uncertainty.py and the forward-chaining loop in _diagnostics.py
    # ensure the precondition by construction (resample size = input size;
    # split_index >= min_train, with min_train passed as min_points). If
    # this fires, an internal caller is violating the precondition -- it
    # is not a user-input error.
    if len(time) < min_points:
        raise RuntimeError(
            f"_fit_model({model_name}): internal precondition violated -- "
            f"len(time)={len(time)} < min_points={min_points}. "
            "Callers must validate before invoking."
        )
    if len(initial_guesses) == 0:
        raise ValueError("at least one initial guess is required")

    def residuals(params: np.ndarray) -> np.ndarray:
        prediction = np.clip(model_func(time, params), _TINY, None)
        return np.log(values) - np.log(prediction)

    # Multi-start loop: run least_squares from each initial guess and keep the
    # lowest-RSS converged solution. For ``len(initial_guesses) == 1`` this
    # collapses to the previous single-start behaviour exactly.
    best_result = None
    best_rss = float("inf")
    last_failure_message = None

    for guess in initial_guesses:
        try:
            result = least_squares(
                residuals,
                x0=guess,
                bounds=bounds,
                method="trf",
                max_nfev=20000,
            )
            fitted_values = np.clip(model_func(time, result.x), _TINY, None)
            iter_resids = np.log(values) - np.log(fitted_values)
            rss = float(np.sum(iter_resids ** 2))
            if rss < best_rss:
                best_rss = rss
                best_result = result
        except Exception as exc:
            last_failure_message = f"{model_name} fit failed: {exc}"
            continue

    if best_result is None:
        warnings = [last_failure_message] if last_failure_message else []
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

    fitted_values = np.clip(model_func(time, best_result.x), _TINY, None)
    resids = np.log(values) - np.log(fitted_values)
    rss = float(np.sum(resids ** 2))
    sigma2 = max(rss / len(values), 1e-12)
    log_likelihood = -0.5 * len(values) * (np.log(2.0 * np.pi * sigma2) + 1.0)
    # Count the observation-noise scale parameter alongside the curve parameters
    # so the information criteria reflect the full log-normal observation model.
    parameter_count = len(parameter_names) + 1
    n = len(values)
    bic = parameter_count * np.log(n) - 2.0 * log_likelihood
    aicc = _aicc(log_likelihood, parameter_count, n)
    log_r_squared = _log_space_r_squared(values, rss)
    warnings = []
    if not best_result.success:
        warnings.append(best_result.message)
    return ModelFit(
        model_name=model_name,
        parameters={name: float(value) for name, value in zip(parameter_names, best_result.x)},
        fitted_values=fitted_values,
        log_likelihood=float(log_likelihood),
        bic=float(bic),
        aicc=float(aicc),
        log_r_squared=float(log_r_squared),
        converged=bool(best_result.success),
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


def _multi_start_guesses_logistic(
    time: np.ndarray,
    values: np.ndarray,
    *,
    n_starts: int,
) -> list:
    """Generate ``n_starts`` initial guesses for the logistic fit.

    The first guess is always the data-driven heuristic from
    :func:`_initial_guess_logistic`. Additional guesses sweep K across
    geometrically-spaced multiples of the observed maximum and t0 across
    early/late positions in the observed window, so a single bad initial
    placement is unlikely to monopolize the multi-start ensemble.
    """
    if n_starts < 1:
        raise ValueError("n_starts must be >= 1")
    base = _initial_guess_logistic(time, values)
    guesses = [base]
    if n_starts == 1:
        return guesses

    span = max(float(time[-1] - time[0]), 1.0)
    max_value = max(float(np.max(values)), 1.0)
    base_r = float(base[1])

    # K factors swept geometrically; t0 positions split between early and
    # late within (and slightly past) the observed window.
    k_factors = np.geomspace(1.5, 500.0, num=4)
    t0_positions = [time[0] + 0.25 * span, time[0] + 0.75 * span]

    for t0 in t0_positions:
        for factor in k_factors:
            if len(guesses) >= n_starts:
                return guesses
            guesses.append(
                np.array([max_value * float(factor), base_r, float(t0)], dtype=float)
            )
    return guesses[:n_starts]


def _bounds_logistic(time: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    span = max(float(time[-1] - time[0]), 1.0)
    # ``max_value`` here is the *raw* observed maximum: it sets the K lower
    # bound (the carrying capacity must sit above any observed value) and
    # scales the upper bound. A previous version floored this at 1.0 to
    # protect against degenerate input, but the floor created an
    # inconsistency with ``_initial_guess_logistic``: when observed values
    # are well below 1.0 the heuristic's K_init would land *below* the
    # floored lower bound and least_squares would refuse the start with
    # "Initial guess is outside of provided bounds". Multi-start (T-16)
    # surfaced this bug in the very-small-values case.
    max_value = float(np.max(values))
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
