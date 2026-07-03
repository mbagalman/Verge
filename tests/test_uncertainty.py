import math

import numpy as np
import pytest

from growthshape import (
    BootstrapIntervals,
    WeightIntervals,
    analyze_growth,
    bootstrap_logistic_intervals,
    bootstrap_model_weights,
    bootstrap_predictions,
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
    assert result.weight_intervals is not None


def test_bootstrap_model_weights_returns_ordered_intervals():
    time, values = _logistic_series(k=100.0, r=0.8, t0=5.0, n=20)

    result = bootstrap_model_weights(time, values, n_boot=200, seed=0)

    assert isinstance(result, WeightIntervals)
    assert result.n_boot == 200
    assert result.n_successful > 100
    for interval in (result.p_exponential, result.p_linear, result.p_logistic):
        assert 0.0 <= interval.low <= interval.median <= interval.high <= 1.0
    # On clean logistic data the logistic-weight CI should sit well above 0.5.
    assert result.p_logistic.low > 0.5


def test_bootstrap_model_weights_seed_makes_results_deterministic():
    time, values = _logistic_series(n=20)

    a = bootstrap_model_weights(time, values, n_boot=100, seed=42)
    b = bootstrap_model_weights(time, values, n_boot=100, seed=42)

    assert a.p_exponential == b.p_exponential
    assert a.p_linear == b.p_linear
    assert a.p_logistic == b.p_logistic


def test_bootstrap_model_weights_smaller_n_gives_wider_logistic_ci():
    # Less data means the bootstrap pulls weights around more, so the
    # winning-weight interval should be wider on n=8 than on n=22 even on
    # clean data. (v1 does not yet accept noisy data, which is the textbook
    # "wider CI" signal -- see TICKETS T-15.)
    short_time, short_values = _logistic_series(
        k=100.0, r=0.8, t0=5.0, n=8, start=0.0, stop=10.0
    )
    long_time, long_values = _logistic_series(
        k=100.0, r=0.8, t0=5.0, n=22, start=0.0, stop=10.0
    )

    short = bootstrap_model_weights(short_time, short_values, n_boot=200, seed=0)
    long_ = bootstrap_model_weights(long_time, long_values, n_boot=200, seed=0)

    short_width = short.p_logistic.high - short.p_logistic.low
    long_width = long_.p_logistic.high - long_.p_logistic.low
    assert short_width >= long_width


def test_analyze_growth_attaches_weight_intervals_when_logistic_preferred():
    time, values = _logistic_series(k=100.0, r=0.75, t0=5.0, n=22)

    result = analyze_growth(time, values, n_boot=150, bootstrap_seed=0)

    assert result.weight_intervals is not None
    assert result.weight_intervals.n_successful > 0
    assert result.weight_intervals.p_logistic.low <= result.p_logistic <= result.weight_intervals.p_logistic.high


def test_analyze_growth_skips_weight_intervals_when_n_boot_zero():
    time, values = _logistic_series()
    result = analyze_growth(time, values, n_boot=0)
    assert result.weight_intervals is None


def test_analyze_growth_skips_weight_intervals_when_linear_wins():
    time = np.linspace(0.0, 10.0, 16)
    values = 5.0 + 2.0 * time

    result = analyze_growth(time, values)

    assert result.preferred_model == "linear"
    assert result.weight_intervals is not None


@pytest.mark.parametrize("bad_prior", [float("nan"), float("inf"), 0.0, -0.1])
def test_bootstrap_model_weights_rejects_invalid_priors(bad_prior):
    time, values = _logistic_series()
    with pytest.raises(ValueError):
        bootstrap_model_weights(time, values, n_boot=10, seed=0, prior_logistic=bad_prior)


def test_summary_includes_ci_when_weight_intervals_available():
    time, values = _logistic_series(k=100.0, r=0.75, t0=5.0, n=22)

    result = analyze_growth(time, values, n_boot=150, bootstrap_seed=0)
    summary = result.summary()

    # Format: "(logistic, 1.00 confidence; 90% CI [0.99, 1.00])."
    assert "90% CI [" in summary
    assert "logistic, " in summary


def test_summary_omits_ci_when_no_bootstrap_ran():
    time = np.linspace(0.0, 10.0, 15)
    values = 4.0 * np.exp(0.16 * time)

    result = analyze_growth(time, values, n_boot=0)
    summary = result.summary()

    assert "CI [" not in summary
    assert "still growing" not in summary  # T-05 rename
    assert "accelerating" in summary


# ---------------------------------------------------------------------------
# Strict-input-contract coverage: the public bootstrap helpers must enforce
# the same growth-only contract that analyze_growth applies via prepare_inputs
# (strictly increasing time, nondecreasing positive values), so that a direct
# caller cannot bypass the package contract and receive intervals the main
# analysis would refuse to produce.
# ---------------------------------------------------------------------------


def _call_bootstrap_logistic(time, values):
    return bootstrap_logistic_intervals(time, values, n_boot=10, seed=0)


def _call_bootstrap_weights(time, values):
    return bootstrap_model_weights(time, values, n_boot=10, seed=0)


def _call_bootstrap_predictions(time, values):
    return bootstrap_predictions(
        time,
        values,
        model_name="logistic",
        prediction_times=[5.0],
        n_boot=10,
        seed=0,
    )


_BOOTSTRAP_ENTRY_POINTS = [
    pytest.param(_call_bootstrap_logistic, id="bootstrap_logistic_intervals"),
    pytest.param(_call_bootstrap_weights, id="bootstrap_model_weights"),
    pytest.param(_call_bootstrap_predictions, id="bootstrap_predictions"),
]


@pytest.mark.parametrize("entry_point", _BOOTSTRAP_ENTRY_POINTS)
def test_bootstrap_entry_points_reject_unsorted_time(entry_point):
    # Permuted middle two points -- not strictly increasing. Without the
    # strict check the helpers used to silently pick time[0] as origin and
    # report a t0 coordinate that depended on the arbitrary first element.
    time = np.array([0.0, 2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    values = 10.0 * np.exp(0.2 * time)
    with pytest.raises(ValueError, match="strictly increasing"):
        entry_point(time, values)


@pytest.mark.parametrize("entry_point", _BOOTSTRAP_ENTRY_POINTS)
def test_bootstrap_entry_points_reject_duplicated_time(entry_point):
    # Strictly increasing forbids equal consecutive timestamps too.
    time = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    values = 10.0 * np.exp(0.2 * time)
    with pytest.raises(ValueError, match="strictly increasing"):
        entry_point(time, values)


@pytest.mark.parametrize("entry_point", _BOOTSTRAP_ENTRY_POINTS)
def test_bootstrap_entry_points_reject_decreasing_values(entry_point):
    # One dip in the middle -- decreasing in places. GrowthShape analyzes growth,
    # not decline, and the bootstrap entry points must surface that the
    # same way the main analysis path does.
    time = np.linspace(0.0, 7.0, 8)
    values = np.array([10.0, 12.0, 11.0, 13.0, 14.0, 15.0, 16.0, 17.0])
    with pytest.raises(ValueError, match="decreasing"):
        entry_point(time, values)


@pytest.mark.parametrize("entry_point", _BOOTSTRAP_ENTRY_POINTS)
def test_bootstrap_entry_points_accept_a_valid_input(entry_point):
    # Sanity check that the parametrized harness itself is not failing for
    # the wrong reason: a clean nondecreasing series goes through cleanly.
    time, values = _logistic_series(n=12)
    entry_point(time, values)
