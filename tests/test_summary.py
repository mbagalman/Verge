import numpy as np

from project_verge import analyze_growth


def _exp_series(a=4.0, r=0.16, n=15, start=0.0, stop=10.0):
    time = np.linspace(start, stop, n)
    values = a * np.exp(r * (time - time[0]))
    return time, values


def _logistic_series(k=120.0, r=0.7, t0=6.0, n=18, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


def test_summary_for_clear_exponential_says_still_growing():
    time, values = _exp_series(a=4.0, r=0.16, n=15)

    result = analyze_growth(time, values, n_boot=0)
    summary = result.summary()

    assert summary.startswith("Verdict:")
    assert "still growing" in summary
    assert "exponential" in summary
    assert "Estimated ceiling" not in summary
    assert "Per-capita slope" in summary


def test_summary_for_clear_logistic_includes_K_and_inflection_intervals():
    time, values = _logistic_series(k=100.0, r=0.75, t0=5.0, n=22)

    result = analyze_growth(time, values, n_boot=200, bootstrap_seed=0)
    summary = result.summary()

    assert "leveling off" in summary
    assert "logistic" in summary
    assert "Estimated ceiling K" in summary
    assert "Estimated inflection time" in summary
    # The estimated ceiling should bracket the true K.
    intervals = result.logistic_intervals
    assert intervals is not None
    assert intervals.K.low <= 100.0 <= intervals.K.high


def test_summary_for_early_logistic_marks_indeterminate_with_reason():
    time, values = _logistic_series(k=200.0, r=0.35, t0=12.0, n=10, start=0.0, stop=5.0)

    result = analyze_growth(time, values, n_boot=100, bootstrap_seed=0)
    summary = result.summary()

    assert "indeterminate" in summary
    assert "logistic_unidentifiable" in summary
    assert "carrying capacity is not identified" in summary


def test_summary_for_polynomial_marks_neither_model_fits_and_omits_K():
    time = np.linspace(1.0, 12.0, 16)
    values = time ** 3

    result = analyze_growth(time, values, min_fit_quality=0.99, n_boot=0)
    summary = result.summary()

    assert "indeterminate" in summary
    assert "neither_model_fits" in summary
    assert "Neither exponential nor logistic" in summary
    assert "Log-space R^2" in summary
    # The underlying logistic fit is untrustworthy here, so K must not be
    # advertised in the summary even if the bootstrap had run.
    assert "Estimated ceiling" not in summary


def test_repr_returns_summary():
    time, values = _exp_series()

    result = analyze_growth(time, values, n_boot=0)

    assert repr(result) == result.summary()
