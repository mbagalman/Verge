"""Non-converged-fit handling.

Regression coverage for the code-review P1 finding: ``_fit_model`` used to keep
the lowest-RSS ``least_squares`` result regardless of ``result.success`` and
score it with finite BIC/AICc, so a non-converged candidate could receive
posterior weight in :func:`analyze_growth`. The fix pins the information
criteria to ``+inf`` (and ``log_r_squared`` to ``-inf``) on the non-converged
fallback path so it can never win the posterior competition.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest
from scipy.optimize import least_squares as _real_least_squares

import growthshape._fit as fit_module
from growthshape import analyze_growth, fit_logistic


def _logistic_series(k=120.0, r=0.7, t0=6.0, n=22, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


def _force_least_squares_failure(monkeypatch):
    """Make every ``least_squares`` call return ``success=False``.

    Runs the real optimizer (so parameters and fitted-values are realistic),
    then returns a shallow copy with the ``success`` flag flipped to mimic
    a non-converged outcome. The optimizer message is preserved if
    informative, otherwise replaced with a clear marker so the test can
    assert on it.
    """

    def fake_least_squares(*args, **kwargs):
        real_result = _real_least_squares(*args, **kwargs)
        masked = types.SimpleNamespace(
            x=real_result.x,
            success=False,
            message="forced non-convergence for test",
        )
        return masked

    monkeypatch.setattr(fit_module, "least_squares", fake_least_squares)


def test_fit_logistic_non_converged_pins_scores_to_infinity(monkeypatch):
    _force_least_squares_failure(monkeypatch)
    time, values = _logistic_series()

    fit = fit_logistic(time, values, n_starts=4)

    assert fit.converged is False
    # Parameters and fitted values are still surfaced for inspection ...
    assert fit.parameters  # not empty
    assert np.all(np.isfinite(fit.fitted_values))
    # ... but the information criteria are pinned so this fit cannot win the
    # posterior competition or pass the min_fit_quality gate.
    assert math.isinf(fit.bic) and fit.bic > 0
    assert math.isinf(fit.aicc) and fit.aicc > 0
    assert math.isinf(fit.log_likelihood) and fit.log_likelihood < 0
    assert math.isinf(fit.log_r_squared) and fit.log_r_squared < 0
    # The optimizer message lands in warnings for traceability.
    assert any("forced non-convergence" in w for w in fit.warnings)


def test_analyze_growth_excludes_non_converged_logistic_from_posterior_weight(monkeypatch):
    # Patch ``least_squares`` only when the logistic fit is being computed by
    # gating on the residuals function's free variables. Simpler: patch every
    # call and use an exponential series so the logistic fit was never going
    # to win anyway -- then verify p_logistic is *exactly* zero, which is the
    # load-bearing property of the fix.
    time = np.linspace(0.0, 10.0, 16)
    true_a, true_r = 4.0, 0.16
    values = true_a * np.exp(true_r * time)

    _force_least_squares_failure(monkeypatch)

    result = analyze_growth(time, values, n_boot=0)

    # All three optimizer-driven fits (exponential / linear / logistic) come
    # back non-converged under the monkeypatch. Their IC scores are +inf so
    # they receive zero posterior weight. Power-law uses linear regression in
    # log-log space and does not go through least_squares, so it is the only
    # candidate with a finite score and ends up with weight 1.0.
    assert result.exponential_fit.converged is False
    assert result.linear_fit.converged is False
    assert result.logistic_fit.converged is False
    assert result.power_law_fit.converged is True

    assert result.p_exponential == pytest.approx(0.0, abs=1e-12)
    assert result.p_linear == pytest.approx(0.0, abs=1e-12)
    assert result.p_logistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_power_law == pytest.approx(1.0, abs=1e-12)


def test_multi_start_keeps_a_converged_result_over_a_lower_rss_failed_one(monkeypatch):
    # When some starts converge and others fail, the converged solution must
    # be returned even if a non-converged start happened to land at a lower
    # RSS during the iteration cap. Force the *first* call to least_squares
    # to come back with ``success=False`` while later calls succeed.
    call_state = {"n": 0}

    def selectively_failing_least_squares(*args, **kwargs):
        call_state["n"] += 1
        real = _real_least_squares(*args, **kwargs)
        if call_state["n"] == 1:
            return types.SimpleNamespace(
                x=real.x,
                success=False,
                message="forced non-convergence for test",
            )
        return real

    monkeypatch.setattr(fit_module, "least_squares", selectively_failing_least_squares)

    time, values = _logistic_series()
    fit = fit_logistic(time, values, n_starts=4)

    # At least one of the four starts ran the real optimizer to convergence;
    # we must surface that converged solution, not the forced-failure one.
    assert fit.converged is True
    assert math.isfinite(fit.bic)
    assert math.isfinite(fit.aicc)
    assert fit.parameters["K"] == pytest.approx(120.0, rel=5e-2)
