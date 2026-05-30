"""Public package interface for GrowthShape."""

from ._api import (
    analyze_growth,
    fit_exponential,
    fit_linear,
    fit_logistic,
    fit_power_law,
)
from ._types import (
    AnalysisAssumptions,
    BootstrapIntervals,
    Diagnostics,
    ForecastDiagnostic,
    GrowthAnalysis,
    IndeterminateReason,
    Interval,
    ModelFit,
    ModelName,
    Prediction,
    PreferredModel,
    SignalAgreement,
    WeightIntervals,
)
from ._uncertainty import (
    bootstrap_logistic_intervals,
    bootstrap_model_weights,
    bootstrap_predictions,
)

__all__ = [
    "AnalysisAssumptions",
    "BootstrapIntervals",
    "Diagnostics",
    "ForecastDiagnostic",
    "GrowthAnalysis",
    "IndeterminateReason",
    "Interval",
    "ModelFit",
    "ModelName",
    "Prediction",
    "PreferredModel",
    "SignalAgreement",
    "WeightIntervals",
    "analyze_growth",
    "bootstrap_logistic_intervals",
    "bootstrap_model_weights",
    "bootstrap_predictions",
    "fit_exponential",
    "fit_linear",
    "fit_logistic",
    "fit_power_law",
]

__version__ = "0.1.0"

