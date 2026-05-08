import numpy as np
import pytest

from project_verge import analyze_growth, fit_logistic


def _logistic_series(k=120.0, r=0.7, t0=6.0, n=22, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


def test_multi_start_is_never_worse_than_single_start_on_clean_logistic():
    # Multi-start includes the heuristic init as one of its guesses, so by
    # construction multi.log_likelihood >= single.log_likelihood.
    time, values = _logistic_series(n=22)
    single = fit_logistic(time, values, n_starts=1)
    multi = fit_logistic(time, values, n_starts=8)

    assert multi.log_likelihood >= single.log_likelihood - 1e-9


def test_multi_start_strictly_beats_single_start_on_pathological_partial_S():
    # K=10000, r=1.0, t0=24 with time in [0, 8], n=12: t0 sits 16 units
    # past the observed window with a high growth rate. The heuristic's
    # midpoint is misled enough that single-start lands in a sub-optimal
    # local minimum; multi-start's K-and-t0 sweep finds the better basin.
    time = np.linspace(0.0, 8.0, 12)
    K_true, r_true, t0_true = 10000.0, 1.0, 24.0
    values = K_true / (1.0 + np.exp(-r_true * (time - t0_true)))

    single = fit_logistic(time, values, n_starts=1)
    multi = fit_logistic(time, values, n_starts=8)

    assert multi.log_likelihood > single.log_likelihood + 0.05


def test_logistic_bounds_fix_lets_very_small_values_fit():
    # Regression test for a bug surfaced while wiring T-16 multi-start:
    # _bounds_logistic floored max_value at 1.0, which placed the K lower
    # bound *above* the heuristic's K_init when observed values were well
    # below 1.0 -- so least_squares refused the start with "Initial guess
    # is outside of provided bounds". Pure exponential growth with very
    # small magnitudes used to fail; now both single- and multi-start
    # succeed.
    time = np.linspace(0.0, 8.0, 12)
    K, r, t0 = 50.0, 1.0, 14.0
    values = K / (1.0 + np.exp(-r * (time - t0)))
    assert float(np.max(values)) < 0.2  # well below the old floor

    single = fit_logistic(time, values, n_starts=1)

    assert single.converged
    assert single.parameters["K"] == pytest.approx(50.0, rel=1e-4)


def test_default_n_starts_in_analyze_growth_is_eight():
    time, values = _logistic_series(n=22)
    default_result = analyze_growth(time, values, n_boot=0)
    explicit_result = analyze_growth(time, values, n_boot=0, n_starts=8)

    assert default_result.preferred_model == explicit_result.preferred_model
    assert default_result.logistic_fit.log_likelihood == explicit_result.logistic_fit.log_likelihood


def test_n_starts_one_recovers_pre_t16_behaviour():
    # Setting n_starts=1 collapses to the single-start path, useful for
    # callers who want to opt out of multi-start (e.g. for performance in
    # an outer loop).
    time, values = _logistic_series(n=22)

    result = analyze_growth(time, values, n_boot=0, n_starts=1)

    assert result.preferred_model == "logistic"
    assert result.logistic_fit.converged


@pytest.mark.parametrize("bad", [0, -1, "8", 8.5, True])
def test_invalid_n_starts_raises(bad):
    time, values = _logistic_series()
    with pytest.raises(ValueError, match="n_starts"):
        analyze_growth(time, values, n_boot=0, n_starts=bad)


def test_multi_start_does_not_break_clean_decisive_cases():
    # Clean exponential / linear / logistic must keep their decisive
    # verdicts under multi-start; the heuristic init is the global
    # optimum on these and the additional starts can at most match it.
    t = np.linspace(0.0, 10.0, 15)
    assert analyze_growth(t, 4.0 * np.exp(0.16 * t), n_boot=0).preferred_model == "exponential"

    t = np.linspace(0.0, 10.0, 16)
    assert analyze_growth(t, 5.0 + 2.0 * t, n_boot=0).preferred_model == "linear"

    t = np.linspace(0.0, 12.0, 22)
    assert analyze_growth(t, 100.0 / (1.0 + np.exp(-0.75 * (t - 5.0))), n_boot=0).preferred_model == "logistic"
