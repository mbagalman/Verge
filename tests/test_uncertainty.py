import math

import numpy as np
import pytest

from project_verge import (
    BootstrapIntervals,
    analyze_growth,
    bootstrap_logistic_intervals,
)


def _logistic_series(k=120.0, r=0.7, t0=6.0, n=18, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


def test_bootstrap_returns_ordered_intervals_for_clean_logistic():
    time, values = _logistic_series(k=100.0, r=0.8, t0=5.0, n=20)

    result = bootstrap_logistic_intervals(time, values, n_boot=200, seed=0)

    assert isinstance(result, BootstrapIntervals)
    assert result.n_boot == 200
    assert result.n_successful > 100
    for interval in (result.K, result.r, result.t0):
        assert interval.low <= interval.median <= interval.high
    assert result.K.low <= 100.0 <= result.K.high
    # t0 is reported in original time, not normalized time.
    assert result.t0.low <= 5.0 <= result.t0.high


def test_bootstrap_predicted_intervals_match_horizons_length():
    time, values = _logistic_series(k=100.0, r=0.8, t0=5.0, n=20)
    horizons = [4.0, 8.0, 14.0]

    result = bootstrap_logistic_intervals(
        time, values, n_boot=100, horizons=horizons, seed=0
    )

    assert result.horizons == (4.0, 8.0, 14.0)
    assert len(result.predicted_intervals) == 3
    for interval in result.predicted_intervals:
        assert interval.low <= interval.median <= interval.high
    # In-sample horizon should bracket the noiseless logistic value.
    expected_at_8 = 100.0 / (1.0 + math.exp(-0.8 * (8.0 - 5.0)))
    assert result.predicted_intervals[1].low <= expected_at_8 <= result.predicted_intervals[1].high


def test_bootstrap_K_brackets_true_value_across_trials():
    # 90% percentile interval should contain the true K in at least 85% of
    # seeded trials. The threshold is intentionally loose because percentile
    # bootstrap is approximate, especially on small clean samples.
    rng = np.random.default_rng(0)
    n_trials = 30
    hits = 0
    for trial in range(n_trials):
        K_true = float(rng.uniform(60.0, 400.0))
        r_true = float(rng.uniform(0.4, 1.0))
        t0_true = float(rng.uniform(4.0, 8.0))
        time, values = _logistic_series(
            k=K_true, r=r_true, t0=t0_true, n=20, start=0.0, stop=12.0
        )
        result = bootstrap_logistic_intervals(
            time, values, n_boot=200, seed=trial, confidence=0.90
        )
        if result.K.low <= K_true <= result.K.high:
            hits += 1
    assert hits >= int(0.85 * n_trials)


def test_bootstrap_seed_makes_results_deterministic():
    time, values = _logistic_series(n=20)

    result_a = bootstrap_logistic_intervals(time, values, n_boot=100, seed=42)
    result_b = bootstrap_logistic_intervals(time, values, n_boot=100, seed=42)

    assert result_a.K == result_b.K
    assert result_a.r == result_b.r
    assert result_a.t0 == result_b.t0


@pytest.mark.parametrize("bad_confidence", [float("nan"), 0.0, 1.0, -0.1, 1.5])
def test_bootstrap_rejects_invalid_confidence(bad_confidence):
    time, values = _logistic_series()
    with pytest.raises(ValueError):
        bootstrap_logistic_intervals(
            time, values, n_boot=10, seed=0, confidence=bad_confidence
        )


def test_bootstrap_rejects_negative_n_boot():
    time, values = _logistic_series()
    with pytest.raises(ValueError):
        bootstrap_logistic_intervals(time, values, n_boot=-1, seed=0)


def test_bootstrap_returns_zero_iterations_when_n_boot_is_zero():
    time, values = _logistic_series()
    result = bootstrap_logistic_intervals(time, values, n_boot=0, seed=0)
    assert result.n_boot == 0
    assert result.n_successful == 0
    assert math.isnan(result.K.low)
    assert math.isnan(result.K.median)
    assert math.isnan(result.K.high)


def test_analyze_growth_attaches_logistic_intervals_when_logistic_converges():
    time, values = _logistic_series(k=100.0, r=0.75, t0=5.0, n=22)

    result = analyze_growth(time, values, n_boot=150, bootstrap_seed=0)

    assert result.logistic_intervals is not None
    assert result.logistic_intervals.n_successful > 0
    assert result.logistic_intervals.K.low <= 100.0 <= result.logistic_intervals.K.high


def test_analyze_growth_predicted_intervals_when_horizons_supplied():
    time, values = _logistic_series(k=100.0, r=0.75, t0=5.0, n=22)
    horizons = [10.0, 15.0]

    result = analyze_growth(
        time, values, n_boot=150, horizons=horizons, bootstrap_seed=0
    )

    intervals = result.logistic_intervals
    assert intervals is not None
    assert intervals.horizons == (10.0, 15.0)
    assert len(intervals.predicted_intervals) == 2


def test_analyze_growth_skips_bootstrap_when_n_boot_zero():
    time, values = _logistic_series()
    result = analyze_growth(time, values, n_boot=0)
    assert result.logistic_intervals is None


def test_analyze_growth_skips_bootstrap_when_exponential_wins():
    # When exponential wins decisively, the logistic optimizer is unidentified
    # and a bootstrap on it would be wasted work. The result should still be
    # well-formed, just without logistic intervals.
    time = np.linspace(0.0, 10.0, 15)
    values = 4.0 * np.exp(0.16 * time)

    result = analyze_growth(time, values)

    assert result.preferred_model == "exponential"
    assert result.logistic_intervals is None
