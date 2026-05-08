import numpy as np
import pytest

from project_verge import analyze_growth, fit_exponential, fit_logistic


def _exp_series(a=4.0, r=0.16, n=15, start=0.0, stop=10.0):
    time = np.linspace(start, stop, n)
    values = a * np.exp(r * (time - time[0]))
    return time, values


def _logistic_series(k=120.0, r=0.7, t0=6.0, n=22, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


def test_modelfit_exposes_aicc_alongside_bic():
    time, values = _exp_series()
    fit = fit_exponential(time, values)

    assert hasattr(fit, "aicc")
    assert hasattr(fit, "bic")
    # Both criteria are defined on the same log-likelihood, so they should be
    # finite for a converged fit on clean data.
    assert np.isfinite(fit.aicc)
    assert np.isfinite(fit.bic)


def test_aicc_penalizes_logistic_more_than_bic_at_minimum_n():
    # At n = 8 with a 4-parameter logistic, AICc's small-sample correction
    # adds 2k(k+1)/(n-k-1) = 2*4*5/3 = 13.33 over plain AIC. BIC adds
    # k*ln(n) = 4*ln(8) = 8.32. So AICc penalizes the logistic param count
    # more aggressively than BIC at the minimum-n boundary -- this is the
    # whole reason AICc exists.
    time = np.linspace(0.0, 7.0, 8)
    values = 4.0 * np.exp(0.16 * time)

    fit = fit_logistic(time, values)

    # AICc - BIC differential for n=8, k=4: 13.33 - 8.32 = 5.01.
    # We don't pin a tight number because log-likelihood differs across runs;
    # we just confirm the structural inequality holds.
    aicc_logistic_penalty = fit.aicc - (-2.0 * fit.log_likelihood)
    bic_logistic_penalty = fit.bic - (-2.0 * fit.log_likelihood)
    assert aicc_logistic_penalty > bic_logistic_penalty


def test_default_criterion_is_aicc():
    time, values = _exp_series()
    default_result = analyze_growth(time, values, n_boot=0)
    aicc_result = analyze_growth(time, values, n_boot=0, criterion="aicc")

    assert default_result.p_exponential == aicc_result.p_exponential
    assert default_result.p_linear == aicc_result.p_linear
    assert default_result.p_logistic == aicc_result.p_logistic


def test_criterion_bic_recovers_pre_t12_behavior_on_borderline_case():
    # On a partial-bend logistic where n is small, BIC and AICc should give
    # *different* weight distributions: AICc penalizes the logistic's extra
    # parameter more harshly than BIC at small n, so p_logistic under AICc
    # should be no larger than under BIC. (Both will still pick logistic
    # decisively on clean data; the differential is in how much weight
    # leaks to the simpler models.)
    time = np.linspace(0.0, 7.0, 10)
    values = 30.0 / (1.0 + np.exp(-0.5 * (time - 4.0)))

    aicc = analyze_growth(time, values, n_boot=0, criterion="aicc")
    bic = analyze_growth(time, values, n_boot=0, criterion="bic")

    assert aicc.p_logistic <= bic.p_logistic + 1e-12


def test_aicc_and_bic_pick_the_same_clean_winner():
    # On clean data the differential between AICc and BIC is dwarfed by the
    # huge BIC/AICc gap between the right and wrong models; both criteria
    # should commit to the same verdict.
    time, values = _logistic_series(n=22)

    aicc = analyze_growth(time, values, n_boot=0, criterion="aicc")
    bic = analyze_growth(time, values, n_boot=0, criterion="bic")

    assert aicc.preferred_model == bic.preferred_model == "logistic"


@pytest.mark.parametrize("bad_criterion", ["AIC", "AICC", "bic ", "", "ridge"])
def test_invalid_criterion_raises(bad_criterion):
    time, values = _exp_series()

    with pytest.raises(ValueError, match="criterion"):
        analyze_growth(time, values, n_boot=0, criterion=bad_criterion)


def test_aicc_falls_back_to_inf_when_n_minus_k_minus_1_is_nonpositive():
    # Reachable inside the bootstrap path with n=4 and a 4-parameter logistic
    # (3 curve params + noise scale): n - k - 1 = -1 < 0, so the AICc
    # correction is undefined and we mark the score as +inf.
    from project_verge._fit import _aicc

    aicc_undefined = _aicc(log_likelihood=-5.0, parameter_count=4, n=4)
    assert aicc_undefined == float("inf")

    # Sanity check: well-defined case.
    aicc_ok = _aicc(log_likelihood=-5.0, parameter_count=2, n=10)
    assert np.isfinite(aicc_ok)
