import numpy as np
import pytest

from growthshape import Interval, WeightIntervals, analyze_growth
from growthshape._api import _verdict_is_fragile


def _logistic_series(k=120.0, r=0.7, t0=6.0, n=22, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


def _weight_intervals(p_log_low, p_log_high, *, n_successful=200):
    """Build a WeightIntervals where only ``p_logistic`` matters for the test."""
    zero = Interval(low=0.0, median=0.0, high=0.0)
    return WeightIntervals(
        n_boot=n_successful,
        n_successful=n_successful,
        confidence=0.9,
        p_exponential=zero,
        p_linear=zero,
        p_logistic=Interval(
            low=p_log_low,
            median=(p_log_low + p_log_high) / 2,
            high=p_log_high,
        ),
        p_power_law=zero,
    )


def test_verdict_is_fragile_returns_true_when_logistic_ci_is_wider_than_threshold():
    intervals = _weight_intervals(0.30, 0.95)  # width 0.65
    assert _verdict_is_fragile("logistic", intervals, max_width=0.40) is True


def test_verdict_is_fragile_returns_false_when_ci_is_narrow():
    intervals = _weight_intervals(0.95, 1.00)  # width 0.05
    assert _verdict_is_fragile("logistic", intervals, max_width=0.40) is False


def test_verdict_is_fragile_returns_false_when_weight_intervals_is_none():
    assert _verdict_is_fragile("logistic", None, max_width=0.40) is False


def test_verdict_is_fragile_returns_false_when_no_successful_resamples():
    nan = float("nan")
    intervals = WeightIntervals(
        n_boot=200,
        n_successful=0,
        confidence=0.9,
        p_exponential=Interval(low=nan, median=nan, high=nan),
        p_linear=Interval(low=nan, median=nan, high=nan),
        p_logistic=Interval(low=nan, median=nan, high=nan),
        p_power_law=Interval(low=nan, median=nan, high=nan),
    )
    assert _verdict_is_fragile("logistic", intervals, max_width=0.40) is False


def test_clean_logistic_does_not_fire_fragile_verdict_at_default_threshold():
    # The default max_weight_ci_width=0.40 must not over-fire on clean data.
    time, values = _logistic_series(k=100.0, r=0.75, t0=5.0, n=22)

    result = analyze_growth(time, values, n_boot=150, bootstrap_seed=0)

    assert result.preferred_model == "logistic"
    assert result.indeterminate_reason is None


# An end-to-end "gate fires" test is intentionally NOT included here. v1's
# strict-monotone input contract produces bootstrap CIs of width exactly 0.0
# on clean logistic data (every resample fits perfectly), so the gate has
# nothing to act on regardless of how aggressively we lower
# ``max_weight_ci_width``. The gate is designed for noisier real-world inputs
# that the v1 contract does not yet allow; once T-15 (smoothing / noise
# tolerance) lands, the firing path becomes reachable end-to-end.


def test_fragile_verdict_does_not_override_other_indeterminate_reasons():
    # Polynomial input fires power_law_shape (T-27), which is earlier in the
    # precedence chain. Even with a punishing max_weight_ci_width=0.0, the
    # fragile_verdict gate must not steal the indeterminate reason.
    time = np.linspace(1.0, 12.0, 16)
    values = time ** 3

    result = analyze_growth(time, values, n_boot=100, max_weight_ci_width=0.0)

    assert result.is_indeterminate
    assert result.indeterminate_reason == "power_law_shape"


def test_fragile_verdict_skipped_when_n_boot_zero():
    # No bootstrap, no fragility signal -- the gate must not fire.
    time, values = _logistic_series(k=100.0, r=0.75, t0=5.0, n=22)

    result = analyze_growth(time, values, n_boot=0, max_weight_ci_width=0.0)

    assert result.preferred_model == "logistic"
    assert result.indeterminate_reason is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.5])
def test_invalid_max_weight_ci_width_raises(bad):
    time, values = _logistic_series()
    with pytest.raises(ValueError):
        analyze_growth(time, values, n_boot=0, max_weight_ci_width=bad)


def test_summary_note_text_includes_resampling_swap_phrase():
    # The summary's note for fragile_verdict points at the right honesty
    # signal -- weight could swap under resampling, so the headline
    # confidence is unreliable. We test this via the formatter's note table
    # because the end-to-end fire path is not reachable on v1 inputs (see
    # comment above).
    from growthshape._summary import _INDETERMINATE_NOTE

    note = _INDETERMINATE_NOTE["fragile_verdict"]
    assert "could swap under resampling" in note
    assert "headline confidence" in note
