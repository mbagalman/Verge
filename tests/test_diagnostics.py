import math

import numpy as np

from growthshape import SignalAgreement, analyze_growth
from growthshape._api import _signals_disagree_with_logistic_verdict


def _exp_series(a=4.0, r=0.16, n=15, start=0.0, stop=10.0):
    time = np.linspace(start, stop, n)
    values = a * np.exp(r * (time - time[0]))
    return time, values


def _logistic_series(k=120.0, r=0.7, t0=6.0, n=18, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


def test_diagnostics_populates_slope_curvature_pvalues_and_signal_agreement():
    time, values = _logistic_series(k=100.0, r=0.7, t0=5.0, n=20)
    diag = analyze_growth(time, values, n_boot=0).diagnostics

    for value in (
        diag.per_capita_slope_std_err,
        diag.per_capita_slope_t_stat,
        diag.per_capita_slope_p_value,
        diag.residual_curvature_std_err,
        diag.residual_curvature_t_stat,
        diag.residual_curvature_p_value,
    ):
        assert math.isfinite(value)

    assert 0.0 <= diag.per_capita_slope_p_value <= 1.0
    assert 0.0 <= diag.residual_curvature_p_value <= 1.0
    assert isinstance(diag.signal_agreement, SignalAgreement)


def test_signal_agreement_votes_high_for_clean_logistic():
    time, values = _logistic_series(k=100.0, r=0.7, t0=5.0, n=20)
    diag = analyze_growth(time, values, n_boot=0).diagnostics

    assert diag.signal_agreement.levelling_off_votes >= 2
    assert diag.signal_agreement.per_capita_slope_negative
    assert diag.signal_agreement.residual_curvature_negative


def test_signal_agreement_votes_low_for_clean_exponential():
    time, values = _exp_series(a=3.0, r=0.18, n=20)
    diag = analyze_growth(time, values, n_boot=0).diagnostics

    assert diag.signal_agreement.levelling_off_votes <= 1


def test_signals_disagree_helper_is_asymmetric():
    # Helper unit test. The gate must not fire on non-logistic verdicts,
    # because per-capita slope and log-residual curvature are also negative
    # for clean linear data (log(a + b*t) is concave; b/y decreases with y).
    full = SignalAgreement(
        per_capita_slope_negative=True,
        residual_curvature_negative=True,
        logistic_has_best_forecast=True,
    )
    none = SignalAgreement(
        per_capita_slope_negative=False,
        residual_curvature_negative=False,
        logistic_has_best_forecast=False,
    )
    one = SignalAgreement(
        per_capita_slope_negative=True,
        residual_curvature_negative=False,
        logistic_has_best_forecast=False,
    )
    two = SignalAgreement(
        per_capita_slope_negative=True,
        residual_curvature_negative=True,
        logistic_has_best_forecast=False,
    )

    # Non-logistic verdicts: gate is silent regardless of supporting signals.
    assert _signals_disagree_with_logistic_verdict("exponential", none) is False
    assert _signals_disagree_with_logistic_verdict("exponential", full) is False
    assert _signals_disagree_with_logistic_verdict("linear", full) is False

    # Logistic verdict needs at least 2 of 3 supporting signals.
    assert _signals_disagree_with_logistic_verdict("logistic", none) is True
    assert _signals_disagree_with_logistic_verdict("logistic", one) is True
    assert _signals_disagree_with_logistic_verdict("logistic", two) is False
    assert _signals_disagree_with_logistic_verdict("logistic", full) is False


def test_clean_linear_passes_through_with_no_indeterminate_reason():
    # The asymmetric gate must not over-fire on linear data, even though
    # linear data has slightly negative log-curvature and decreasing
    # per-capita growth (signals that, in isolation, look logistic-ish).
    time = np.linspace(0.0, 10.0, 16)
    values = 5.0 + 2.0 * time

    result = analyze_growth(time, values, n_boot=0)

    assert result.preferred_model == "linear"
    assert result.indeterminate_reason is None
