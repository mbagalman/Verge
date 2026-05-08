import numpy as np
import pytest

from project_verge import analyze_growth
from project_verge._fit import smooth_to_monotone


def _logistic_series(k=120.0, r=0.7, t0=6.0, n=22, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


def _noisy_logistic(k, r, t0, n, start, stop, *, sigma=0.03, seed=0):
    rng = np.random.default_rng(seed)
    time = np.linspace(start, stop, n)
    clean = k / (1.0 + np.exp(-r * (time - t0)))
    # Multiplicative log-normal noise around the clean curve. With small
    # sigma the resulting series is *almost* monotone but has occasional
    # downward blips that fail the v1 strict-monotone gate.
    noise = rng.normal(0.0, sigma, size=len(clean))
    return time, clean * np.exp(noise)


def test_smooth_to_monotone_produces_nondecreasing_output():
    rng = np.random.default_rng(0)
    raw = 10.0 + np.cumsum(rng.standard_normal(50))  # random walk
    raw = np.clip(raw, 0.1, None)  # keep positive

    smoothed = smooth_to_monotone(raw, window=3)

    assert np.all(np.diff(smoothed) >= 0.0)
    assert len(smoothed) == len(raw)
    assert np.all(smoothed >= 0.1)


def test_smooth_to_monotone_is_idempotent_on_already_monotone_data():
    # Median of monotone data slightly biases endpoints but cumulative-max
    # plus a strictly-increasing input means the result equals the input
    # for points far from the boundary; the test confirms idempotence in
    # the "median pass changes nothing structurally" sense.
    time, values = _logistic_series(n=22)

    smoothed = smooth_to_monotone(values, window=3)

    assert np.all(np.diff(smoothed) >= 0.0)
    # Smoothed and original differ only at the boundaries by <1% on clean
    # logistic input.
    interior = slice(2, -2)
    assert np.allclose(smoothed[interior], values[interior], rtol=1e-6)


@pytest.mark.parametrize("bad_window", [0, -1, 2, 4, 100])
def test_smooth_to_monotone_rejects_invalid_window(bad_window):
    values = np.linspace(1.0, 10.0, 16)
    with pytest.raises(ValueError, match="positive odd integer"):
        smooth_to_monotone(values, window=bad_window)


def test_analyze_growth_still_rejects_non_monotone_by_default():
    # Default behaviour must be unchanged: no opt-in, no smoothing.
    time, noisy = _noisy_logistic(100.0, 0.7, 5.0, n=22, start=0.0, stop=12.0, sigma=0.03)
    # Sanity check: the noisy series violates monotonicity somewhere.
    assert np.any(np.diff(noisy) < 0.0)

    with pytest.raises(ValueError, match="nondecreasing"):
        analyze_growth(time, noisy, n_boot=0)


def test_analyze_growth_with_smoothing_accepts_noisy_logistic():
    time, noisy = _noisy_logistic(100.0, 0.7, 5.0, n=22, start=0.0, stop=12.0, sigma=0.03)

    result = analyze_growth(time, noisy, n_boot=0, allow_smoothing=True)

    # Underlying signal is logistic; smoothing should let Verge see it.
    # We don't assert the verdict committed (small-sample / borderline cases
    # can still hit indeterminate gates) -- we assert the call succeeded.
    assert result is not None
    assert result.preferred_model in {"logistic", "indeterminate"}


def test_transform_log_records_smoothing_action():
    time, noisy = _noisy_logistic(100.0, 0.7, 5.0, n=22, start=0.0, stop=12.0, sigma=0.03)

    result = analyze_growth(time, noisy, n_boot=0, allow_smoothing=True)

    assert len(result.transform_log) == 1
    assert "rolling-median smoothing" in result.transform_log[0]
    assert "window=3" in result.transform_log[0]
    assert "cumulative-max" in result.transform_log[0]


def test_transform_log_is_empty_when_smoothing_off():
    time, values = _logistic_series(n=22)
    result = analyze_growth(time, values, n_boot=0)

    assert result.transform_log == ()


def test_input_values_reflect_smoothed_series_when_smoothing_applied():
    time, noisy = _noisy_logistic(100.0, 0.7, 5.0, n=22, start=0.0, stop=12.0, sigma=0.03)

    result = analyze_growth(time, noisy, n_boot=0, allow_smoothing=True)

    # input_values is what predict() and plot() see; it must be the
    # smoothed series (which is what was actually fit), not the raw input.
    assert np.all(np.diff(result.input_values) >= 0.0)
    # And it differs from the raw input.
    assert not np.allclose(result.input_values, noisy)


def test_smoothing_with_custom_window_is_recorded_in_transform_log():
    time, noisy = _noisy_logistic(100.0, 0.7, 5.0, n=30, start=0.0, stop=12.0, sigma=0.05)

    result = analyze_growth(
        time, noisy, n_boot=0, allow_smoothing=True, smoothing_window=5
    )

    assert "window=5" in result.transform_log[0]


def test_smoothing_does_not_break_clean_monotone_input():
    # Sanity: clean logistic data with smoothing on still classifies
    # correctly. The smoother is a near-identity on already-monotone
    # data, so the verdict shouldn't change.
    time, values = _logistic_series(k=100.0, r=0.75, t0=5.0, n=22)

    no_smooth = analyze_growth(time, values, n_boot=0)
    with_smooth = analyze_growth(time, values, n_boot=0, allow_smoothing=True)

    assert no_smooth.preferred_model == with_smooth.preferred_model == "logistic"
