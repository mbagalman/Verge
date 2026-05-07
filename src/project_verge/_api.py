from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from ._diagnostics import build_diagnostics
from ._fit import fit_exponential_model, fit_logistic_model, prepare_inputs
from ._types import GrowthAnalysis, ModelFit
from ._uncertainty import bootstrap_logistic_intervals


def fit_exponential(time, values, *, min_points: int = 8) -> ModelFit:
    normalized_time, normalized_values = prepare_inputs(time, values, min_points=min_points)
    return fit_exponential_model(normalized_time, normalized_values, min_points=min_points)


def fit_logistic(time, values, *, min_points: int = 8) -> ModelFit:
    normalized_time, normalized_values = prepare_inputs(time, values, min_points=min_points)
    return fit_logistic_model(normalized_time, normalized_values, min_points=min_points)


def analyze_growth(
    time,
    values,
    *,
    prior_exponential: float = 0.5,
    prior_logistic: float = 0.5,
    min_points: int = 8,
    min_fit_quality: float = 0.85,
    horizons: Optional[Sequence[float]] = None,
    n_boot: int = 500,
    bootstrap_confidence: float = 0.90,
    bootstrap_seed: Optional[int] = None,
) -> GrowthAnalysis:
    normalized_time, normalized_values = prepare_inputs(time, values, min_points=min_points)
    _validate_priors(prior_exponential, prior_logistic)
    _validate_fit_quality(min_fit_quality)

    exponential_fit = fit_exponential_model(normalized_time, normalized_values, min_points=min_points)
    logistic_fit = fit_logistic_model(normalized_time, normalized_values, min_points=min_points)

    p_exponential, p_logistic = _posterior_model_weights(
        exponential_fit,
        logistic_fit,
        prior_exponential=prior_exponential,
        prior_logistic=prior_logistic,
    )

    diagnostics = build_diagnostics(
        normalized_time,
        normalized_values,
        exponential_fit=exponential_fit,
        logistic_fit=logistic_fit,
    )

    leading_model = "exponential" if p_exponential >= p_logistic else "logistic"
    winning_weight = max(p_exponential, p_logistic)
    both_fits_poor = (
        exponential_fit.log_r_squared < min_fit_quality
        and logistic_fit.log_r_squared < min_fit_quality
    )
    logistic_poorly_identified = len(diagnostics.identifiability_warnings) > 0

    # Reason precedence: an absolute fit-quality failure dominates relative
    # model comparison, which in turn dominates a logistic-only identifiability
    # caveat. Only one reason is reported even when several apply.
    if both_fits_poor:
        indeterminate_reason: Optional[str] = "neither_model_fits"
    elif winning_weight < 0.70:
        indeterminate_reason = "ambiguous_evidence"
    elif leading_model == "logistic" and logistic_poorly_identified:
        indeterminate_reason = "logistic_unidentifiable"
    else:
        indeterminate_reason = None

    is_indeterminate = indeterminate_reason is not None
    preferred_model = "indeterminate" if is_indeterminate else leading_model

    assumptions = (
        "The comparison is limited to exponential and logistic growth.",
        "Likelihood is computed under a shared log-normal observation model.",
        "Posterior probabilities are approximated from BIC with user-specified priors.",
        "Inputs are assumed to be positive, finite, and nondecreasing.",
    )

    # Bootstrap is only informative when the logistic verdict is part of the
    # answer the user cares about. When exponential wins decisively, the
    # logistic optimizer thrashes on every resample (K is unidentified) and
    # the resulting CI is meaningless decoration that costs many seconds.
    bootstrap_relevant = preferred_model == "logistic" or is_indeterminate
    if logistic_fit.converged and n_boot > 0 and bootstrap_relevant:
        logistic_intervals = bootstrap_logistic_intervals(
            time,
            values,
            n_boot=n_boot,
            horizons=horizons,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        )
    else:
        logistic_intervals = None

    return GrowthAnalysis(
        p_exponential=float(p_exponential),
        p_logistic=float(p_logistic),
        preferred_model=preferred_model,
        is_indeterminate=is_indeterminate,
        indeterminate_reason=indeterminate_reason,
        exponential_fit=exponential_fit,
        logistic_fit=logistic_fit,
        diagnostics=diagnostics,
        assumptions=assumptions,
        logistic_intervals=logistic_intervals,
    )


def _posterior_model_weights(
    exponential_fit: ModelFit,
    logistic_fit: ModelFit,
    *,
    prior_exponential: float,
    prior_logistic: float,
) -> Tuple[float, float]:
    bics = np.array([exponential_fit.bic, logistic_fit.bic], dtype=float)
    priors = np.array([prior_exponential, prior_logistic], dtype=float)

    finite_mask = np.isfinite(bics)
    if not np.any(finite_mask):
        raise RuntimeError("Neither model produced a valid fit.")

    min_bic = np.min(bics[finite_mask])
    log_weights = np.full(2, -np.inf, dtype=float)
    log_weights[finite_mask] = np.log(priors[finite_mask]) - 0.5 * (bics[finite_mask] - min_bic)
    normalization = np.logaddexp.reduce(log_weights[finite_mask])
    probabilities = np.zeros(2, dtype=float)
    probabilities[finite_mask] = np.exp(log_weights[finite_mask] - normalization)
    return float(probabilities[0]), float(probabilities[1])


def _validate_priors(prior_exponential: float, prior_logistic: float) -> None:
    if not math.isfinite(prior_exponential) or not math.isfinite(prior_logistic):
        raise ValueError("model priors must be finite")
    if prior_exponential <= 0.0 or prior_logistic <= 0.0:
        raise ValueError("model priors must be strictly positive")


def _validate_fit_quality(min_fit_quality: float) -> None:
    if not math.isfinite(min_fit_quality):
        raise ValueError("min_fit_quality must be finite")
    if min_fit_quality > 1.0:
        raise ValueError("min_fit_quality must be at most 1.0")
