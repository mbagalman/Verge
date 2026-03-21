"""Public package interface for Project Verge."""

from ._api import analyze_growth, fit_exponential, fit_logistic
from ._types import Diagnostics, GrowthAnalysis, ModelFit

__all__ = [
    "Diagnostics",
    "GrowthAnalysis",
    "ModelFit",
    "analyze_growth",
    "fit_exponential",
    "fit_logistic",
]

__version__ = "0.1.0"

