"""Bootstrap uncertainty intervals for the logistic growth fit."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

from ._fit import fit_logistic_model, logistic_curve
from ._types import BootstrapIntervals, Interval

# Bootstrap resamples can be lopsided (many duplicates of a few points), so we
# allow the inner fitter to attempt a fit on smaller effective samples than the
# public API would. The primary fit's quality is governed by the user-facing
# ``min_points`` knob; here we just want as many converged bootstrap fits as
# possible to keep the percentile intervals stable.
_BOOTSTRAP_MIN_POINTS = 4


def bootstrap_logistic_intervals(
    time: Sequence[float],
    values: Sequence[float],
    *,
    n_boot: int = 500,
    horizons: Optional[Sequence[float]] = None,
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

    time_arr = np.asarray(time, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if time_arr.ndim != 1 or values_arr.ndim != 1:
        raise ValueError("time and values must be one-dimensional sequences")
    if len(time_arr) != len(values_arr):
        raise ValueError("time and values must have the same length")
    if len(time_arr) < _BOOTSTRAP_MIN_POINTS:
        raise ValueError(
            f"at least {_BOOTSTRAP_MIN_POINTS} observations are required for bootstrap"
        )
    if not np.all(np.isfinite(time_arr)) or not np.all(np.isfinite(values_arr)):
        raise ValueError("time and values must contain only finite numbers")
    if np.any(values_arr <= 0.0):
        raise ValueError("values must be strictly positive")

    horizons_arr = np.asarray(
        horizons if horizons is not None else (), dtype=float
    )
    if horizons_arr.ndim != 1:
        raise ValueError("horizons must be a one-dimensional sequence")

    # Fit in the same time-origin frame the rest of the package uses, so that
    # K, r, and predicted values are interpretable in the user's units.
    origin = float(time_arr[0])
    time_norm = time_arr - origin
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
