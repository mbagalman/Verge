"""Up-front scope checks: flat data and decreasing data.

The library's stated question is "do I have a reason to think this will keep
going up, or do I have a reason to think this will level off?" Two input
shapes are out of scope and are rejected before any fitting:

  - flat data (no growth signal): the question is ill-posed
  - decreasing data: GrowthShape analyzes growth, not decline

The flat-data threshold is configurable via ``min_relative_range`` on
``analyze_growth`` and the four ``fit_*`` wrappers. Default 0.01 (1% of
max value); pass 0 to disable.
"""

import numpy as np
import pytest

from growthshape import (
    analyze_growth,
    fit_exponential,
    fit_linear,
    fit_logistic,
    fit_power_law,
)


def _logistic_series(k=120.0, r=0.7, t0=6.0, n=22, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


# ---------------------------------------------------------------------------
# Flat-data rejection
# ---------------------------------------------------------------------------


def test_perfectly_flat_data_is_rejected():
    time = np.linspace(0.0, 10.0, 16)
    values = np.full(16, 100.0)

    with pytest.raises(ValueError, match="no growth signal"):
        analyze_growth(time, values, n_boot=0)


def test_near_flat_data_below_default_threshold_is_rejected():
    # range / max = 0.5 / 100.5 ~ 0.5% < 1% default -> reject
    time = np.linspace(0.0, 10.0, 16)
    values = np.linspace(100.0, 100.5, 16)

    with pytest.raises(ValueError, match="no growth signal"):
        analyze_growth(time, values, n_boot=0)


def test_near_flat_data_above_default_threshold_is_accepted():
    # range / max = 5 / 105 ~ 4.8% > 1% default -> accept
    time = np.linspace(0.0, 10.0, 16)
    values = np.linspace(100.0, 105.0, 16)

    result = analyze_growth(time, values, n_boot=0)

    assert result is not None  # didn't crash; verdict can be anything


def test_flat_data_check_can_be_disabled_with_min_relative_range_zero():
    time = np.linspace(0.0, 10.0, 16)
    values = np.linspace(100.0, 100.5, 16)  # would be rejected at default

    # Should not raise.
    result = analyze_growth(time, values, n_boot=0, min_relative_range=0.0)

    assert result is not None


def test_flat_data_check_threshold_can_be_relaxed():
    time = np.linspace(0.0, 10.0, 16)
    values = np.linspace(100.0, 100.5, 16)  # 0.5% range

    # 0.001 is below the 0.5% range, so the data should now be accepted.
    result = analyze_growth(time, values, n_boot=0, min_relative_range=0.001)
    assert result is not None

    # 0.05 is well above 0.5%, so still rejected.
    with pytest.raises(ValueError, match="no growth signal"):
        analyze_growth(time, values, n_boot=0, min_relative_range=0.05)


def test_flat_data_error_message_includes_observed_and_threshold_percentages():
    time = np.linspace(0.0, 10.0, 16)
    values = np.full(16, 100.0)

    with pytest.raises(ValueError) as excinfo:
        analyze_growth(time, values, n_boot=0)
    msg = str(excinfo.value)
    # Both the observed range and the configured threshold should appear,
    # in percent notation, so the user can see exactly what tripped.
    assert "0.0000%" in msg or "0.00%" in msg  # observed
    assert "1.00%" in msg  # default threshold
    assert "min_relative_range=0" in msg


def test_flat_data_check_applies_to_public_fit_wrappers():
    time = np.linspace(0.0, 10.0, 16)
    values = np.full(16, 100.0)

    with pytest.raises(ValueError, match="no growth signal"):
        fit_logistic(time, values)


# ---------------------------------------------------------------------------
# Decreasing-data rejection (message rewording)
# ---------------------------------------------------------------------------


def test_decreasing_data_error_explains_scope_not_just_contract():
    time = np.linspace(0.0, 10.0, 10)
    # One downward dip in an otherwise-increasing sequence.
    values = np.array([1.0, 2.0, 3.0, 2.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    with pytest.raises(ValueError) as excinfo:
        analyze_growth(time, values, n_boot=0)
    msg = str(excinfo.value)

    # Must communicate scope, not just "contract violation".
    assert "GrowthShape analyzes growth" in msg
    assert "outside its scope" in msg
    # Should also point users at the smoothing escape hatch.
    assert "allow_smoothing" in msg


def test_decreasing_data_message_does_not_appear_for_smoothed_input():
    # With smoothing on, a noisy nondecreasing series passes through.
    rng = np.random.default_rng(0)
    time = np.linspace(0.0, 12.0, 22)
    clean = 100.0 / (1.0 + np.exp(-0.7 * (time - 5.0)))
    noisy = clean * np.exp(rng.normal(0.0, 0.03, size=22))
    assert np.any(np.diff(noisy) < 0.0)  # confirm there are dips

    # Should not raise -- smoothing handles it.
    result = analyze_growth(time, noisy, n_boot=0, allow_smoothing=True)
    assert result is not None


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value", [float("nan"), float("inf"), -0.1, 1.0, 1.5]
)
def test_invalid_min_relative_range_raises(bad_value):
    time, values = _logistic_series(n=22)
    with pytest.raises(ValueError, match="min_relative_range"):
        analyze_growth(time, values, n_boot=0, min_relative_range=bad_value)


@pytest.mark.parametrize(
    "fit_func", [fit_exponential, fit_linear, fit_logistic, fit_power_law]
)
@pytest.mark.parametrize(
    "bad_value", [float("nan"), float("inf"), -0.1, 1.0, 1.5]
)
def test_public_fit_wrappers_reject_invalid_min_relative_range(fit_func, bad_value):
    time, values = _logistic_series(n=22)
    with pytest.raises(ValueError, match="min_relative_range"):
        fit_func(time, values, min_relative_range=bad_value)


def test_min_relative_range_zero_is_explicitly_allowed():
    time, values = _logistic_series(n=22)
    result = analyze_growth(time, values, n_boot=0, min_relative_range=0.0)
    assert result.preferred_model in {"exponential", "linear", "logistic", "indeterminate"}
