"""Tests for the structured AnalysisAssumptions field on GrowthAnalysis.

T-23 replaced the static prose tuple with a dataclass that records the
methodological choices Verge actually made for the call -- criterion,
evidence_strength, candidate models, observation model. These tests pin
the membership and the per-call dynamism.
"""

import numpy as np

from project_verge import AnalysisAssumptions, analyze_growth


def _logistic_series(k=120.0, r=0.7, t0=6.0, n=22, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


def test_result_assumptions_is_an_analysis_assumptions_dataclass():
    time, values = _logistic_series()
    result = analyze_growth(time, values, n_boot=0)

    assert isinstance(result.assumptions, AnalysisAssumptions)


def test_assumptions_records_actual_criterion_choice():
    time, values = _logistic_series()

    aicc = analyze_growth(time, values, n_boot=0, criterion="aicc")
    bic = analyze_growth(time, values, n_boot=0, criterion="bic")

    assert aicc.assumptions.criterion == "aicc"
    assert bic.assumptions.criterion == "bic"


def test_assumptions_records_actual_evidence_strength_choice():
    time, values = _logistic_series()

    positive = analyze_growth(time, values, n_boot=0, evidence_strength="positive")
    decisive = analyze_growth(time, values, n_boot=0, evidence_strength="decisive")

    assert positive.assumptions.evidence_strength == "positive"
    assert decisive.assumptions.evidence_strength == "decisive"


def test_assumptions_lists_all_four_candidate_models():
    time, values = _logistic_series()
    result = analyze_growth(time, values, n_boot=0)

    assert set(result.assumptions.candidate_models) == {
        "exponential",
        "linear",
        "logistic",
        "power_law",
    }


def test_assumptions_observation_model_is_log_normal_in_v1():
    # Documenting the v1 fixed value -- if we ever add a non-log-normal
    # noise model, this test needs an explicit update.
    time, values = _logistic_series()
    result = analyze_growth(time, values, n_boot=0)

    assert result.assumptions.observation_model == "log_normal"


def test_assumptions_is_immutable():
    # frozen=True dataclass; attempting to reassign a field should raise.
    import dataclasses
    import pytest

    time, values = _logistic_series()
    result = analyze_growth(time, values, n_boot=0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.assumptions.criterion = "bic"
