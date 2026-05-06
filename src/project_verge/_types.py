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
class Diagnostics:
    """Supporting diagnostics for interpreting the primary model comparison."""

    per_capita_slope: float
    per_capita_intercept: float
    residual_curvature_score: float
    forecast_mae_exponential: float
    forecast_mae_logistic: float
    fit_warnings: Tuple[str, ...] = ()
    identifiability_warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class GrowthAnalysis:
    """End-to-end result returned by :func:`analyze_growth`."""

    p_exponential: float
    p_logistic: float
    preferred_model: str
    is_indeterminate: bool
    indeterminate_reason: Optional[str]
    exponential_fit: ModelFit
    logistic_fit: ModelFit
    diagnostics: Diagnostics
    assumptions: Tuple[str, ...]
