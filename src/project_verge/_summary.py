"""Human-readable verdict formatting for :class:`GrowthAnalysis`."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ._types import GrowthAnalysis, Interval


_VERDICT_LABEL = {
    "exponential": "accelerating",
    "linear": "steady",
    "logistic": "leveling off",
    "indeterminate": "indeterminate",
}


_INDETERMINATE_NOTE = {
    "neither_model_fits": (
        "Neither exponential nor logistic explains this data well on the log scale."
    ),
    "ambiguous_evidence": (
        "Posterior weights between exponential and logistic are too close to call."
    ),
    "logistic_unidentifiable": (
        "The logistic carrying capacity is not identified by the observed window."
    ),
}


def format_summary(result: "GrowthAnalysis") -> str:
    """Return a short multi-line verdict suitable for ``print(result)``."""

    lines: List[str] = [_format_verdict_line(result)]

    if result.is_indeterminate:
        note = _INDETERMINATE_NOTE.get(result.indeterminate_reason or "")
        if note is not None:
            lines.append(note)
        if result.indeterminate_reason == "neither_model_fits":
            lines.append(
                "Log-space R^2: exponential "
                f"{result.exponential_fit.log_r_squared:.3f}, "
                f"logistic {result.logistic_fit.log_r_squared:.3f}."
            )

    if _should_show_logistic_intervals(result):
        intervals = result.logistic_intervals
        assert intervals is not None  # narrowed by _should_show_logistic_intervals
        lines.append(_format_K_line(intervals.K))
        lines.append(_format_t0_line(intervals.t0))

    lines.append(_format_diagnostic_line(result))
    return "\n".join(lines)


def _format_verdict_line(result: "GrowthAnalysis") -> str:
    label = _VERDICT_LABEL.get(result.preferred_model, result.preferred_model)
    if result.is_indeterminate:
        reason = result.indeterminate_reason or "unspecified"
        return f"Verdict: {label} (reason: {reason})."
    confidence = {
        "exponential": result.p_exponential,
        "linear": result.p_linear,
        "logistic": result.p_logistic,
    }.get(result.preferred_model, 0.0)
    return f"Verdict: {label} ({result.preferred_model}, {confidence:.2f} confidence)."


def _format_K_line(K: "Interval") -> str:
    return (
        f"Estimated ceiling K ~= {_fmt(K.median)} "
        f"[{_fmt(K.low)}, {_fmt(K.high)}]."
    )


def _format_t0_line(t0: "Interval") -> str:
    return (
        f"Estimated inflection time ~= {_fmt(t0.median)} "
        f"[{_fmt(t0.low)}, {_fmt(t0.high)}]."
    )


def _format_diagnostic_line(result: "GrowthAnalysis") -> str:
    diag = result.diagnostics
    parts = [f"Per-capita slope: {diag.per_capita_slope:+.4g}"]

    mae_by_model = {
        "exponential": diag.forecast_mae_exponential,
        "linear": diag.forecast_mae_linear,
        "logistic": diag.forecast_mae_logistic,
    }
    mae = mae_by_model.get(result.preferred_model)
    if mae is not None and math.isfinite(mae):
        parts.append(f"forecast log-MAE ({result.preferred_model}): {mae:.3g}")
    elif result.is_indeterminate:
        parts.append(
            "posterior weights: "
            f"exponential {result.p_exponential:.2f}, "
            f"linear {result.p_linear:.2f}, "
            f"logistic {result.p_logistic:.2f}"
        )

    return "; ".join(parts) + "."


def _should_show_logistic_intervals(result: "GrowthAnalysis") -> bool:
    intervals = result.logistic_intervals
    if intervals is None or intervals.n_successful == 0:
        return False
    # When neither model fits, the underlying logistic curve is itself
    # untrustworthy, so reporting K/t0 from it would be misleading.
    if result.indeterminate_reason == "neither_model_fits":
        return False
    return True


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.3g}"
