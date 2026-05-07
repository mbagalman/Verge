"""Public package interface for Project Verge."""

from ._api import analyze_growth, fit_exponential, fit_linear, fit_logistic
from ._types import (
    BootstrapIntervals,
    Diagnostics,
    GrowthAnalysis,
    Interval,
    ModelFit,
    Prediction,
    SignalAgreement,
    WeightIntervals,
)
from ._uncertainty import (
    bootstrap_logistic_intervals,
    bootstrap_model_weights,
    bootstrap_predictions,
)

__all__ = [
    "BootstrapIntervals",
    "Diagnostics",
    "GrowthAnalysis",
    "Interval",
    "ModelFit",
    "Prediction",
    "SignalAgreement",
    "WeightIntervals",
    "analyze_growth",
    "bootstrap_logistic_intervals",
    "bootstrap_model_weights",
    "bootstrap_predictions",
    "fit_exponential",
    "fit_linear",
    "fit_logistic",
]

__version__ = "0.1.0"

