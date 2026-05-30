"""Bootstrap uncertainty intervals for the logistic growth fit."""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from ._fit import (
    exponential_curve,
    fit_exponential_model,
    fit_linear_model,
    fit_logistic_model,
    fit_power_law_model,
    linear_curve,
    logistic_curve,
    prepare_inputs,
)
from ._types import BootstrapIntervals, Interval, WeightIntervals

# Bootstrap resamples can be lopsided (many duplicates of a few points), so we
# allow the inner fitter to attempt a fit on smaller effective samples than the
# public API would. The primary fit's quality is governed by the user-facing
# ``min_points`` knob; here we just want as many converged bootstrap fits as
# possible to keep the percentile intervals stable.
_BOOTSTRAP_MIN_POINTS = 4


def bootstrap_logistic_intervals(
    time: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    n_boot: int = 500,
    horizons: Optional[npt.ArrayLike] = None,
    confidence: float = 0.90,
    seed: Optional[int] = None,
) -> BootstrapIntervals:
    """Pair-bootstrap percentile intervals for the logistic K, r, t0, and predictions.

    Resamples ``(time, values)`` pairs with replacement, refits the logistic
    model on each resample, and returns percentile intervals at the requested
    ``confidence`` level. Failed (non-converged) resamples are skipped and
    counted via ``n_successful``; if too few resamples succeed the intervals
    will be NaN.
    """

    _validate_confidence(confidence)
    if n_boot < 0:
        raise ValueError("n_boot must be non-negative")

    time_norm, values_arr, origin = _prepare_bootstrap_inputs(time, values)

    horizons_arr = np.asarray(
        horizons if horizons is not None else (), dtype=float
    )
    if horizons_arr.ndim != 1:
        raise ValueError("horizons must be a one-dimensional sequence")

    # Shift horizons into the same zero-origin frame the bootstrap loop fits
    # in, so K, r, and predicted values remain interpretable in user units.
    horizons_norm = horizons_arr - origin

    rng = np.random.default_rng(seed)
    n = len(time_norm)

    K_samples: List[float] = []
    r_samples: List[float] = []
    t0_samples: List[float] = []
    pred_samples: List[List[float]] = [[] for _ in range(len(horizons_arr))]

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        order = np.argsort(time_norm[idx])
        t_boot = time_norm[idx][order]
        y_boot = values_arr[idx][order]

        try:
            fit = fit_logistic_model(t_boot, y_boot, min_points=_BOOTSTRAP_MIN_POINTS)
        except ValueError:
            continue
        if not fit.converged or not fit.parameters:
            continue

        K = fit.parameters["K"]
        r = fit.parameters["r"]
        t0 = fit.parameters["t0"]
        K_samples.append(K)
        r_samples.append(r)
        t0_samples.append(t0)
        if len(horizons_norm):
            preds = logistic_curve(horizons_norm, K, r, t0)
            for i, p in enumerate(preds):
                pred_samples[i].append(float(p))

    n_successful = len(K_samples)
    K_interval = _percentile_interval(K_samples, confidence)
    r_interval = _percentile_interval(r_samples, confidence)
    t0_interval_normalized = _percentile_interval(t0_samples, confidence)
    t0_interval = Interval(
        low=t0_interval_normalized.low + origin,
        median=t0_interval_normalized.median + origin,
        high=t0_interval_normalized.high + origin,
    )
    predicted_intervals = tuple(
        _percentile_interval(samples, confidence) for samples in pred_samples
    )

    return BootstrapIntervals(
        n_boot=n_boot,
        n_successful=n_successful,
        confidence=float(confidence),
        K=K_interval,
        r=r_interval,
        t0=t0_interval,
        horizons=tuple(float(h) for h in horizons_arr),
        predicted_intervals=predicted_intervals,
    )


def bootstrap_predictions(
    time: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    model_name: str,
    prediction_times: npt.ArrayLike,
    n_boot: int = 200,
    confidence: float = 0.9,
    seed: Optional[int] = None,
) -> List[Interval]:
    """Pair-bootstrap percentile intervals for a single model's predictions.

    Resamples ``(time, values)`` with replacement, refits the named model on
    each resample, evaluates the fitted curve at every prediction time, and
    returns one :class:`Interval` per prediction time. Resamples whose fit
    fails to converge are skipped; intervals are NaN if no resample succeeds.
    """

    _validate_confidence(confidence)
    if n_boot < 0:
        raise ValueError("n_boot must be non-negative")

    fitter, curve = _model_fitter_and_curve(model_name)

    time_norm, values_arr, origin = _prepare_bootstrap_inputs(time, values)
    pred_times_arr = np.asarray(prediction_times, dtype=float)
    if pred_times_arr.ndim != 1:
        raise ValueError("prediction_times must be a one-dimensional sequence")

    pred_times_norm = pred_times_arr - origin

    rng = np.random.default_rng(seed)
    n = len(time_norm)
    pred_samples: List[List[float]] = [[] for _ in range(len(pred_times_norm))]

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        order = np.argsort(time_norm[idx])
        t_boot = time_norm[idx][order]
        y_boot = values_arr[idx][order]

        try:
            fit = fitter(t_boot, y_boot, min_points=_BOOTSTRAP_MIN_POINTS)
        except ValueError:
            continue
        if not fit.converged or not fit.parameters:
            continue

        preds = curve(pred_times_norm, fit.parameters)
        for i, p in enumerate(preds):
            pred_samples[i].append(float(p))

    return [_percentile_interval(samples, confidence) for samples in pred_samples]


def _model_fitter_and_curve(model_name: str):
    """Return ``(fit_function, curve_function)`` for the named model."""

    if model_name == "exponential":
        return (
            fit_exponential_model,
            lambda t, params: exponential_curve(t, params["a"], params["r"]),
        )
    if model_name == "linear":
        return (
            fit_linear_model,
            lambda t, params: linear_curve(t, params["a"], params["b"]),
        )
    if model_name == "logistic":
        return (
            fit_logistic_model,
            lambda t, params: logistic_curve(
                t, params["K"], params["r"], params["t0"]
            ),
        )
    raise ValueError(f"unsupported model_name: {model_name!r}")


def bootstrap_model_weights(
    time: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    prior_exponential: float = 0.5,
    prior_linear: float = 0.5,
    prior_logistic: float = 0.5,
    prior_power_law: float = 0.5,
    criterion: str = "aicc",
    n_boot: int = 500,
    confidence: float = 0.90,
    seed: Optional[int] = None,
) -> WeightIntervals:
    """Pair-bootstrap percentile intervals for the information-criterion
    posterior weights.

    Resamples ``(time, values)`` pairs with replacement, refits all four
    candidate models (exponential, linear, logistic, power-law) on each
    resample, recomputes the four-way weights using the requested
    ``criterion`` ("aicc" or "bic"), and returns percentile intervals at the
    requested ``confidence`` level. Resamples where every fit fails to
    produce a finite criterion score are skipped and counted via
    ``n_successful``.
    """

    _validate_confidence(confidence)
    _validate_weight_priors(
        prior_exponential, prior_linear, prior_logistic, prior_power_law
    )
    if criterion not in ("aicc", "bic"):
        raise ValueError("criterion must be 'aicc' or 'bic'")
    if n_boot < 0:
        raise ValueError("n_boot must be non-negative")

    time_norm, values_arr, _origin = _prepare_bootstrap_inputs(time, values)

    rng = np.random.default_rng(seed)
    n = len(time_norm)

    p_exp_samples: List[float] = []
    p_lin_samples: List[float] = []
    p_log_samples: List[float] = []
    p_pow_samples: List[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        order = np.argsort(time_norm[idx])
        t_boot = time_norm[idx][order]
        y_boot = values_arr[idx][order]

        try:
            exp_fit = fit_exponential_model(t_boot, y_boot, min_points=_BOOTSTRAP_MIN_POINTS)
            lin_fit = fit_linear_model(t_boot, y_boot, min_points=_BOOTSTRAP_MIN_POINTS)
            log_fit = fit_logistic_model(t_boot, y_boot, min_points=_BOOTSTRAP_MIN_POINTS)
            pow_fit = fit_power_law_model(t_boot, y_boot, min_points=_BOOTSTRAP_MIN_POINTS)
        except ValueError:
            continue

        score_field = "bic" if criterion == "bic" else "aicc"
        weights = _four_way_weights(
            getattr(exp_fit, score_field),
            getattr(lin_fit, score_field),
            getattr(log_fit, score_field),
            getattr(pow_fit, score_field),
            prior_exponential=prior_exponential,
            prior_linear=prior_linear,
            prior_logistic=prior_logistic,
            prior_power_law=prior_power_law,
        )
        if weights is None:
            continue
        p_exp, p_lin, p_log, p_pow = weights
        p_exp_samples.append(p_exp)
        p_lin_samples.append(p_lin)
        p_log_samples.append(p_log)
        p_pow_samples.append(p_pow)

    n_successful = len(p_exp_samples)
    return WeightIntervals(
        n_boot=n_boot,
        n_successful=n_successful,
        confidence=float(confidence),
        p_exponential=_percentile_interval(p_exp_samples, confidence),
        p_linear=_percentile_interval(p_lin_samples, confidence),
        p_logistic=_percentile_interval(p_log_samples, confidence),
        p_power_law=_percentile_interval(p_pow_samples, confidence),
    )


def _four_way_weights(
    score_exp: float,
    score_lin: float,
    score_log: float,
    score_pow: float,
    *,
    prior_exponential: float,
    prior_linear: float,
    prior_logistic: float,
    prior_power_law: float,
) -> Optional[tuple]:
    """Normalize information-criterion weights across four models. ``score_*``
    is whichever criterion the caller selected (BIC or AICc); the math is
    identical. Returns ``None`` when no model has a finite score."""

    scores = np.array([score_exp, score_lin, score_log, score_pow], dtype=float)
    priors = np.array(
        [prior_exponential, prior_linear, prior_logistic, prior_power_law],
        dtype=float,
    )
    finite_mask = np.isfinite(scores)
    if not np.any(finite_mask):
        return None
    min_score = np.min(scores[finite_mask])
    log_weights = np.full(4, -np.inf, dtype=float)
    log_weights[finite_mask] = np.log(priors[finite_mask]) - 0.5 * (scores[finite_mask] - min_score)
    normalization = np.logaddexp.reduce(log_weights[finite_mask])
    probabilities = np.zeros(4, dtype=float)
    probabilities[finite_mask] = np.exp(log_weights[finite_mask] - normalization)
    return tuple(float(p) for p in probabilities)


def _prepare_bootstrap_inputs(time, values):
    """Strict input validation shared between every bootstrap entry point.

    Routes through :func:`prepare_inputs` so the public bootstrap helpers
    enforce the same growth-only contract that :func:`analyze_growth` does:
    strictly increasing time, nondecreasing strictly positive finite values,
    length-matched 1-D sequences. Without this, a direct caller could bypass
    the package's stated contract and receive intervals that the main analysis
    would refuse to produce -- and an unsorted ``time`` would make the
    bootstrap's reported ``t0`` coordinate depend on the arbitrary first
    element rather than the time origin.

    The bootstrap-specific deviations from ``analyze_growth``'s defaults:

    - ``min_points`` is ``_BOOTSTRAP_MIN_POINTS = 4`` so the bootstrap can
      operate on sparser inputs than the public ``analyze_growth`` minimum.
      The inner fits operate on resamples (with duplicates), not on this
      raw input, so the stricter user-facing floor would be over-restrictive.
    - ``min_relative_range`` is ``0.0`` because the relative-range check is
      an ``analyze_growth``-level scope decision, not a numerical
      precondition of the bootstrap math.

    Returns ``(time_norm, values_arr, origin)`` where ``time_norm`` is the
    zero-origin time array, ``values_arr`` is the validated values array,
    and ``origin`` is the original ``time[0]`` so callers can shift the
    reported ``t0`` interval back into the user's time units.
    """

    time_arr = np.asarray(time, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    time_norm, values_validated = prepare_inputs(
        time_arr,
        values_arr,
        min_points=_BOOTSTRAP_MIN_POINTS,
        min_relative_range=0.0,
    )
    # ``prepare_inputs`` validated that time is strictly increasing, so the
    # first element is guaranteed to be the minimum -- a safe origin choice.
    origin = float(time_arr[0])
    return time_norm, values_validated, origin


def _validate_weight_priors(*priors: float) -> None:
    for prior in priors:
        if not math.isfinite(prior):
            raise ValueError("weight priors must be finite")
        if prior <= 0.0:
            raise ValueError("weight priors must be strictly positive")


def _percentile_interval(samples: List[float], confidence: float) -> Interval:
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        nan = float("nan")
        return Interval(low=nan, median=nan, high=nan)
    alpha = (1.0 - confidence) / 2.0
    return Interval(
        low=float(np.percentile(arr, 100.0 * alpha)),
        median=float(np.percentile(arr, 50.0)),
        high=float(np.percentile(arr, 100.0 * (1.0 - alpha))),
    )


def _validate_confidence(confidence: float) -> None:
    if not math.isfinite(confidence):
        raise ValueError("confidence must be finite")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in the open interval (0, 1)")
