from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ModelFit:
    """Stores the result of fitting a single growth model."""

    model_name: str
    parameters: Mapping[str, float]
    fitted_values: np.ndarray
    log_likelihood: float
    bic: float
    log_r_squared: float
    converged: bool
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        fitted_values = np.array(self.fitted_values, dtype=float, copy=True)
        fitted_values.setflags(write=False)
        object.__setattr__(self, "fitted_values", fitted_values)


@dataclass(frozen=True)
class SignalAgreement:
    """Boolean flags for the three non-BIC signals that point toward leveling off.

    Each flag is ``True`` when the corresponding signal is consistent with
    logistic (S-curve) growth rather than exponential or linear growth. The
    aggregate vote count is exposed via :attr:`levelling_off_votes`.
    """

    per_capita_slope_negative: bool
    residual_curvature_negative: bool
    logistic_has_best_forecast: bool

    @property
    def levelling_off_votes(self) -> int:
        return (
            int(self.per_capita_slope_negative)
            + int(self.residual_curvature_negative)
            + int(self.logistic_has_best_forecast)
        )


@dataclass(frozen=True)
class Diagnostics:
    """Supporting diagnostics for interpreting the primary model comparison."""

    per_capita_slope: float
    per_capita_intercept: float
    per_capita_slope_std_err: float
    per_capita_slope_t_stat: float
    per_capita_slope_p_value: float
    residual_curvature_score: float
    residual_curvature_std_err: float
    residual_curvature_t_stat: float
    residual_curvature_p_value: float
    forecast_mae_exponential: float
    forecast_mae_linear: float
    forecast_mae_logistic: float
    signal_agreement: SignalAgreement
    fit_warnings: Tuple[str, ...] = ()
    identifiability_warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Interval:
    """Percentile interval (low, median, high) at the bootstrap confidence level."""

    low: float
    median: float
    high: float


@dataclass(frozen=True)
class BootstrapIntervals:
    """Bootstrap uncertainty intervals for the logistic fit and any requested horizons.

    ``K`` and ``r`` are reported in the same units as the input ``values`` and
    ``time``. ``t0`` is reported in the *original* time coordinate (i.e. without
    the internal time-origin shift applied by :func:`analyze_growth`).
    ``predicted_intervals`` is one :class:`Interval` per requested horizon, in
    the order the horizons were supplied.
    """

    n_boot: int
    n_successful: int
    confidence: float
    K: Interval
    r: Interval
    t0: Interval
    horizons: Tuple[float, ...]
    predicted_intervals: Tuple[Interval, ...]


@dataclass(frozen=True)
class GrowthAnalysis:
    """End-to-end result returned by :func:`analyze_growth`."""

    p_exponential: float
    p_linear: float
    p_logistic: float
    preferred_model: str
    is_indeterminate: bool
    indeterminate_reason: Optional[str]
    exponential_fit: ModelFit
    linear_fit: ModelFit
    logistic_fit: ModelFit
    diagnostics: Diagnostics
    assumptions: Tuple[str, ...]
    logistic_intervals: Optional[BootstrapIntervals]

    def summary(self) -> str:
        """Return a short human-readable verdict suitable for ``print(result)``."""

        # Local import: ``_summary`` references ``GrowthAnalysis`` for type
        # checking, so importing it at module load would create a cycle.
        from ._summary import format_summary

        return format_summary(self)

    def __repr__(self) -> str:
        return self.summary()
