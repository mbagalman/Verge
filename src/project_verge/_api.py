from __future__ import annotations

import math
from typing import Optional

import numpy as np
import numpy.typing as npt

from ._diagnostics import build_diagnostics
from ._fit import (
    fit_exponential_model,
    fit_linear_model,
    fit_logistic_model,
    fit_power_law_model,
    prepare_inputs,
    smooth_to_monotone,
)
from ._types import GrowthAnalysis, ModelFit, SignalAgreement, WeightIntervals
from ._uncertainty import bootstrap_logistic_intervals, bootstrap_model_weights


def fit_exponential(
    time, values, *, min_points: int = 8, min_relative_range: float = 0.01
) -> ModelFit:
    normalized_time, normalized_values = prepare_inputs(
        time, values, min_points=min_points, min_relative_range=min_relative_range
    )
    return fit_exponential_model(normalized_time, normalized_values, min_points=min_points)


def fit_linear(
    time, values, *, min_points: int = 8, min_relative_range: float = 0.01
) -> ModelFit:
    normalized_time, normalized_values = prepare_inputs(
        time, values, min_points=min_points, min_relative_range=min_relative_range
    )
    return fit_linear_model(normalized_time, normalized_values, min_points=min_points)


def fit_logistic(
    time,
    values,
    *,
    min_points: int = 8,
    n_starts: int = 1,
    min_relative_range: float = 0.01,
) -> ModelFit:
    normalized_time, normalized_values = prepare_inputs(
        time, values, min_points=min_points, min_relative_range=min_relative_range
    )
    return fit_logistic_model(
        normalized_time, normalized_values, min_points=min_points, n_starts=n_starts
    )


def fit_power_law(
    time, values, *, min_points: int = 8, min_relative_range: float = 0.01
) -> ModelFit:
    normalized_time, normalized_values = prepare_inputs(
        time, values, min_points=min_points, min_relative_range=min_relative_range
    )
    return fit_power_law_model(normalized_time, normalized_values, min_points=min_points)


def analyze_growth(
    time,
    values,
    *,
    prior_exponential: float = 0.5,
    prior_linear: float = 0.5,
    prior_logistic: float = 0.5,
    prior_power_law: float = 0.5,
    min_points: int = 8,
    min_fit_quality: float = 0.85,
    max_weight_ci_width: float = 0.40,
    criterion: str = "aicc",
    evidence_strength: str = "strong",
    allow_smoothing: bool = False,
    smoothing_window: int = 3,
    n_starts: int = 8,
    min_relative_range: float = 0.01,
    horizons: Optional[npt.ArrayLike] = None,
    n_boot: int = 500,
    bootstrap_confidence: float = 0.90,
    bootstrap_seed: Optional[int] = None,
) -> GrowthAnalysis:
    _validate_priors(prior_exponential, prior_linear, prior_logistic, prior_power_law)
    _validate_fit_quality(min_fit_quality)
    _validate_max_weight_ci_width(max_weight_ci_width)
    _validate_criterion(criterion)
    _validate_n_starts(n_starts)
    _validate_min_relative_range(min_relative_range)
    winning_weight_threshold = _evidence_strength_threshold(evidence_strength)

    # Optional pre-fit smoothing for noisy real-world data. The smoother runs
    # before prepare_inputs, so ``values`` and ``input_values_array`` below
    # both reflect the smoothed series; this is the array predict() and
    # plot() will see, since it's what was actually fit.
    transform_log = []
    values_for_fitting = values
    if allow_smoothing:
        smoothed = smooth_to_monotone(values, window=smoothing_window)
        values_for_fitting = smoothed
        transform_log.append(
            f"Pre-fit rolling-median smoothing (window={smoothing_window}) "
            f"with cumulative-max applied to enforce the nondecreasing contract."
        )

    normalized_time, normalized_values = prepare_inputs(
        time,
        values_for_fitting,
        min_points=min_points,
        min_relative_range=min_relative_range,
    )
    # ``prepare_inputs`` has validated that ``time`` is a finite, length-matched,
    # strictly-increasing 1-D sequence, so taking ``[0]`` after the fact is safe.
    time_origin = float(np.asarray(time, dtype=float)[0])
    input_time_array = np.asarray(time, dtype=float)
    input_values_array = np.asarray(values_for_fitting, dtype=float)

    exponential_fit = fit_exponential_model(normalized_time, normalized_values, min_points=min_points)
    linear_fit = fit_linear_model(normalized_time, normalized_values, min_points=min_points)
    logistic_fit = fit_logistic_model(
        normalized_time, normalized_values, min_points=min_points, n_starts=n_starts
    )
    power_law_fit = fit_power_law_model(normalized_time, normalized_values, min_points=min_points)

    weights_by_name = _posterior_model_weights(
        {
            "exponential": (exponential_fit, prior_exponential),
            "linear": (linear_fit, prior_linear),
            "logistic": (logistic_fit, prior_logistic),
            "power_law": (power_law_fit, prior_power_law),
        },
        criterion=criterion,
    )
    p_exponential = weights_by_name["exponential"]
    p_linear = weights_by_name["linear"]
    p_logistic = weights_by_name["logistic"]
    p_power_law = weights_by_name["power_law"]

    leading_model = max(weights_by_name, key=weights_by_name.__getitem__)
    winning_weight = weights_by_name[leading_model]
    leading_fit = {
        "exponential": exponential_fit,
        "linear": linear_fit,
        "logistic": logistic_fit,
        "power_law": power_law_fit,
    }[leading_model]

    diagnostics = build_diagnostics(
        normalized_time,
        normalized_values,
        exponential_fit=exponential_fit,
        linear_fit=linear_fit,
        logistic_fit=logistic_fit,
        leading_fit=leading_fit,
    )
    all_fits_poor = (
        exponential_fit.log_r_squared < min_fit_quality
        and linear_fit.log_r_squared < min_fit_quality
        and logistic_fit.log_r_squared < min_fit_quality
        and power_law_fit.log_r_squared < min_fit_quality
    )
    logistic_poorly_identified = len(diagnostics.identifiability_warnings) > 0

    # Reason precedence: an absolute fit-quality failure dominates everything;
    # a winning power-law shape (which has no clean verdict in v1) dominates
    # the relative ambiguity check; the logistic-only identifiability and
    # signal-disagreement gates apply only when a non-power-law model leads.
    # Only one reason is reported even when several apply.
    if all_fits_poor:
        indeterminate_reason: Optional[str] = "neither_model_fits"
    elif leading_model == "power_law":
        indeterminate_reason = "power_law_shape"
    elif winning_weight < winning_weight_threshold:
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
        # Bootstrap uses the smoothed values when smoothing was applied, so
        # the resampled fits live in the same coordinate system as the main
        # analysis. Otherwise the weight CIs would be computed on noisy raw
        # data while the headline fits sit on the smoothed series, and the
        # two would disagree.
        logistic_intervals = bootstrap_logistic_intervals(
            time,
            values_for_fitting,
            n_boot=n_boot,
            horizons=horizons,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        )
        weight_intervals = bootstrap_model_weights(
            time,
            values_for_fitting,
            prior_exponential=prior_exponential,
            prior_linear=prior_linear,
            prior_logistic=prior_logistic,
            prior_power_law=prior_power_law,
            criterion=criterion,
            n_boot=n_boot,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        )
    else:
        logistic_intervals = None
        weight_intervals = None

    # Fragile-verdict gate (T-28). Last in the indeterminate precedence chain
    # because it relies on bootstrap data that the earlier gates do not need.
    # Only downgrades verdicts that were otherwise decisive; once-indeterminate
    # results keep their structured reason.
    if not is_indeterminate and _verdict_is_fragile(
        leading_model, weight_intervals, max_weight_ci_width
    ):
        indeterminate_reason = "fragile_verdict"
        is_indeterminate = True
        preferred_model = "indeterminate"

    return GrowthAnalysis(
        p_exponential=float(p_exponential),
        p_linear=float(p_linear),
        p_logistic=float(p_logistic),
        p_power_law=float(p_power_law),
        preferred_model=preferred_model,
        is_indeterminate=is_indeterminate,
        indeterminate_reason=indeterminate_reason,
        exponential_fit=exponential_fit,
        linear_fit=linear_fit,
        logistic_fit=logistic_fit,
        power_law_fit=power_law_fit,
        diagnostics=diagnostics,
        assumptions=assumptions,
        logistic_intervals=logistic_intervals,
        weight_intervals=weight_intervals,
        input_time=input_time_array,
        input_values=input_values_array,
        time_origin=time_origin,
        transform_log=tuple(transform_log),
    )


def _posterior_model_weights(
    fits_with_priors: dict,
    *,
    criterion: str = "aicc",
) -> dict:
    """Normalize information-criterion-derived posterior weights across an
    arbitrary set of models.

    ``fits_with_priors`` is a mapping ``name -> (ModelFit, prior)``. Returns a
    mapping of the same names to normalized posterior weights summing to 1.
    Models whose criterion score is non-finite (e.g. fit failures, or AICc
    when ``n - k - 1 <= 0``) receive zero weight.
    """

    score_field = "bic" if criterion == "bic" else "aicc"
    names = list(fits_with_priors.keys())
    scores = np.array(
        [getattr(fits_with_priors[name][0], score_field) for name in names],
        dtype=float,
    )
    priors = np.array([fits_with_priors[name][1] for name in names], dtype=float)

    finite_mask = np.isfinite(scores)
    if not np.any(finite_mask):
        raise RuntimeError("None of the candidate models produced a valid fit.")

    min_score = np.min(scores[finite_mask])
    log_weights = np.full(len(names), -np.inf, dtype=float)
    log_weights[finite_mask] = np.log(priors[finite_mask]) - 0.5 * (scores[finite_mask] - min_score)
    normalization = np.logaddexp.reduce(log_weights[finite_mask])
    probabilities = np.zeros(len(names), dtype=float)
    probabilities[finite_mask] = np.exp(log_weights[finite_mask] - normalization)
    return {name: float(p) for name, p in zip(names, probabilities)}


def _verdict_is_fragile(
    leading_model: str,
    weight_intervals: Optional[WeightIntervals],
    max_width: float,
) -> bool:
    """Return True when the bootstrap CI on the leading model's weight is too wide.

    "Too wide" means the percentile interval spans more than ``max_width`` of
    the [0, 1] range -- a verdict whose own confidence number could swap
    materially under resampling. The gate is a safety net for cases that
    pass every earlier indeterminate check but still rest on unstable
    evidence (e.g. some random-walk-like inputs once T-15 admits noisy
    data); it only fires when ``weight_intervals`` is populated, since
    without bootstrap data there is no fragility signal to act on.
    """

    if weight_intervals is None or weight_intervals.n_successful == 0:
        return False
    interval_by_model = {
        "exponential": weight_intervals.p_exponential,
        "linear": weight_intervals.p_linear,
        "logistic": weight_intervals.p_logistic,
        "power_law": weight_intervals.p_power_law,
    }
    interval = interval_by_model.get(leading_model)
    if interval is None:
        return False
    if not math.isfinite(interval.low) or not math.isfinite(interval.high):
        return False
    return (interval.high - interval.low) > max_width


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


def _validate_max_weight_ci_width(max_weight_ci_width: float) -> None:
    if not math.isfinite(max_weight_ci_width):
        raise ValueError("max_weight_ci_width must be finite")
    # 0.0 is a valid maximum-strictness setting: any positive CI width on the
    # winning weight will trip the fragile_verdict gate.
    if max_weight_ci_width < 0.0 or max_weight_ci_width > 1.0:
        raise ValueError("max_weight_ci_width must be in the closed interval [0, 1]")


def _validate_criterion(criterion: str) -> None:
    if criterion not in ("aicc", "bic"):
        raise ValueError("criterion must be 'aicc' or 'bic'")


def _validate_n_starts(n_starts: int) -> None:
    if not isinstance(n_starts, (int, np.integer)) or isinstance(n_starts, bool):
        raise ValueError("n_starts must be an integer")
    if n_starts < 1:
        raise ValueError("n_starts must be >= 1")


def _validate_min_relative_range(min_relative_range: float) -> None:
    if not math.isfinite(min_relative_range):
        raise ValueError("min_relative_range must be finite")
    if min_relative_range < 0.0 or min_relative_range >= 1.0:
        raise ValueError(
            "min_relative_range must be in the half-open interval [0, 1)"
        )


# Winning-weight thresholds for the four-way model competition, mapped from
# Kass & Raftery's (1995) interpretive bands for log Bayes factors. The
# thresholds are rounded to clean two-decimal values; in the binary-comparison
# limit the exact mapping is w = exp(d/2) / (1 + exp(d/2)) where d is the IC
# gap to the next-best model -- so 0.75 ~ d=2.2, 0.95 ~ d=5.9, 0.99 ~ d=9.2.
_EVIDENCE_BAND_THRESHOLDS = {
    "positive": 0.75,
    "strong": 0.95,
    "decisive": 0.99,
}


def _evidence_strength_threshold(evidence_strength: str) -> float:
    if evidence_strength not in _EVIDENCE_BAND_THRESHOLDS:
        raise ValueError(
            "evidence_strength must be one of "
            + ", ".join(repr(name) for name in _EVIDENCE_BAND_THRESHOLDS)
        )
    return _EVIDENCE_BAND_THRESHOLDS[evidence_strength]
