from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from ._diagnostics import build_diagnostics
from ._fit import (
    fit_exponential_model,
    fit_linear_model,
    fit_logistic_model,
    prepare_inputs,
)
from ._types import GrowthAnalysis, ModelFit, SignalAgreement
from ._uncertainty import bootstrap_logistic_intervals, bootstrap_model_weights


def fit_exponential(time, values, *, min_points: int = 8) -> ModelFit:
    normalized_time, normalized_values = prepare_inputs(time, values, min_points=min_points)
    return fit_exponential_model(normalized_time, normalized_values, min_points=min_points)


def fit_linear(time, values, *, min_points: int = 8) -> ModelFit:
    normalized_time, normalized_values = prepare_inputs(time, values, min_points=min_points)
    return fit_linear_model(normalized_time, normalized_values, min_points=min_points)


def fit_logistic(time, values, *, min_points: int = 8) -> ModelFit:
    normalized_time, normalized_values = prepare_inputs(time, values, min_points=min_points)
    return fit_logistic_model(normalized_time, normalized_values, min_points=min_points)


def analyze_growth(
    time,
    values,
    *,
    prior_exponential: float = 0.5,
    prior_linear: float = 0.5,
    prior_logistic: float = 0.5,
    min_points: int = 8,
    min_fit_quality: float = 0.85,
    horizons: Optional[Sequence[float]] = None,
    n_boot: int = 500,
    bootstrap_confidence: float = 0.90,
    bootstrap_seed: Optional[int] = None,
) -> GrowthAnalysis:
    normalized_time, normalized_values = prepare_inputs(time, values, min_points=min_points)
    # ``prepare_inputs`` has validated that ``time`` is a finite, length-matched,
    # strictly-increasing 1-D sequence, so taking ``[0]`` after the fact is safe.
    time_origin = float(np.asarray(time, dtype=float)[0])
    input_time_array = np.asarray(time, dtype=float)
    input_values_array = np.asarray(values, dtype=float)
    _validate_priors(prior_exponential, prior_linear, prior_logistic)
    _validate_fit_quality(min_fit_quality)

    exponential_fit = fit_exponential_model(normalized_time, normalized_values, min_points=min_points)
    linear_fit = fit_linear_model(normalized_time, normalized_values, min_points=min_points)
    logistic_fit = fit_logistic_model(normalized_time, normalized_values, min_points=min_points)

    weights_by_name = _posterior_model_weights(
        {
            "exponential": (exponential_fit, prior_exponential),
            "linear": (linear_fit, prior_linear),
            "logistic": (logistic_fit, prior_logistic),
        }
    )
    p_exponential = weights_by_name["exponential"]
    p_linear = weights_by_name["linear"]
    p_logistic = weights_by_name["logistic"]

    diagnostics = build_diagnostics(
        normalized_time,
        normalized_values,
        exponential_fit=exponential_fit,
        linear_fit=linear_fit,
        logistic_fit=logistic_fit,
    )

    leading_model = max(weights_by_name, key=weights_by_name.__getitem__)
    winning_weight = weights_by_name[leading_model]
    all_fits_poor = (
        exponential_fit.log_r_squared < min_fit_quality
        and linear_fit.log_r_squared < min_fit_quality
        and logistic_fit.log_r_squared < min_fit_quality
    )
    logistic_poorly_identified = len(diagnostics.identifiability_warnings) > 0

    # Reason precedence: an absolute fit-quality failure dominates relative
    # model comparison, which in turn dominates a logistic-only identifiability
    # caveat, which in turn dominates a multi-signal disagreement. Only one
    # reason is reported even when several apply.
    if all_fits_poor:
        indeterminate_reason: Optional[str] = "neither_model_fits"
    elif winning_weight < 0.70:
        indeterminate_reason = "ambiguous_evidence"
    elif leading_model == "logistic" and logistic_poorly_identified:
        indeterminate_reason = "logistic_unidentifiable"
    elif _signals_disagree_with_logistic_verdict(leading_model, diagnostics.signal_agreement):
        indeterminate_reason = "signal_disagreement"
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
        weight_intervals = bootstrap_model_weights(
            time,
            values,
            prior_exponential=prior_exponential,
            prior_linear=prior_linear,
            prior_logistic=prior_logistic,
            n_boot=n_boot,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        )
    else:
        logistic_intervals = None
        weight_intervals = None

    return GrowthAnalysis(
        p_exponential=float(p_exponential),
        p_linear=float(p_linear),
        p_logistic=float(p_logistic),
        preferred_model=preferred_model,
        is_indeterminate=is_indeterminate,
        indeterminate_reason=indeterminate_reason,
        exponential_fit=exponential_fit,
        linear_fit=linear_fit,
        logistic_fit=logistic_fit,
        diagnostics=diagnostics,
        assumptions=assumptions,
        logistic_intervals=logistic_intervals,
        weight_intervals=weight_intervals,
        input_time=input_time_array,
        input_values=input_values_array,
        time_origin=time_origin,
    )


def _posterior_model_weights(
    fits_with_priors: dict,
) -> dict:
    """Normalize BIC-derived posterior weights across an arbitrary set of models.

    ``fits_with_priors`` is a mapping ``name -> (ModelFit, prior)``. Returns a
    mapping of the same names to normalized posterior weights summing to 1.
    Models whose BIC is non-finite (e.g. fit failures) receive zero weight.
    """

    names = list(fits_with_priors.keys())
    bics = np.array([fits_with_priors[name][0].bic for name in names], dtype=float)
    priors = np.array([fits_with_priors[name][1] for name in names], dtype=float)

    finite_mask = np.isfinite(bics)
    if not np.any(finite_mask):
        raise RuntimeError("None of the candidate models produced a valid fit.")

    min_bic = np.min(bics[finite_mask])
    log_weights = np.full(len(names), -np.inf, dtype=float)
    log_weights[finite_mask] = np.log(priors[finite_mask]) - 0.5 * (bics[finite_mask] - min_bic)
    normalization = np.logaddexp.reduce(log_weights[finite_mask])
    probabilities = np.zeros(len(names), dtype=float)
    probabilities[finite_mask] = np.exp(log_weights[finite_mask] - normalization)
    return {name: float(p) for name, p in zip(names, probabilities)}


def _signals_disagree_with_logistic_verdict(
    leading_model: str,
    agreement: SignalAgreement,
) -> bool:
    """Second-opinion check applied only to a BIC-derived logistic verdict.

    Per-capita-slope and log-residual-curvature can be significantly negative
    for clean linear data too (because ``log(a + b*t)`` is concave and
    ``b/y`` decreases with ``y``), so applying the same gate symmetrically
    against non-logistic verdicts would over-fire on linear cases. The
    asymmetry is deliberate: BIC's three-way comparison already weighs
    exponential vs linear vs logistic against each other, so the supporting
    signals only need to second-guess the logistic branch.
    """

    if leading_model != "logistic":
        return False
    return agreement.levelling_off_votes < 2


def _validate_priors(*priors: float) -> None:
    for prior in priors:
        if not math.isfinite(prior):
            raise ValueError("model priors must be finite")
        if prior <= 0.0:
            raise ValueError("model priors must be strictly positive")


def _validate_fit_quality(min_fit_quality: float) -> None:
    if not math.isfinite(min_fit_quality):
        raise ValueError("min_fit_quality must be finite")
    if min_fit_quality > 1.0:
        raise ValueError("min_fit_quality must be at most 1.0")
