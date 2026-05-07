"""Public package interface for Project Verge."""

from ._api import analyze_growth, fit_exponential, fit_logistic
from ._types import (
    BootstrapIntervals,
    Diagnostics,
    GrowthAnalysis,
    Interval,
    ModelFit,
)
from ._uncertainty import bootstrap_logistic_intervals

__all__ = [
    "BootstrapIntervals",
    "Diagnostics",
    "GrowthAnalysis",
    "Interval",
    "ModelFit",
    "analyze_growth",
    "bootstrap_logistic_intervals",
    "fit_exponential",
    "fit_logistic",
]

__version__ = "0.1.0"

