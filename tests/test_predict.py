import numpy as np
import pytest

from project_verge import Prediction, analyze_growth, bootstrap_predictions


def _exp_series(a=4.0, r=0.16, n=15, start=0.0, stop=10.0):
    time = np.linspace(start, stop, n)
    values = a * np.exp(r * (time - time[0]))
    return time, values


def _logistic_series(k=120.0, r=0.7, t0=6.0, n=22, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


def test_predict_returns_prediction_with_default_ci_for_logistic():
    time, values = _logistic_series(k=100.0, r=0.75, t0=5.0, n=22)
    result = analyze_growth(time, values, n_boot=150, bootstrap_seed=0)

    pred = result.predict(7.0)

    assert isinstance(pred, Prediction)
    assert pred.low <= pred.point <= pred.high


def test_predict_with_ci_none_returns_scalar_for_scalar_input():
    time, values = _exp_series()
    result = analyze_growth(time, values, n_boot=0)

    pred = result.predict(5.0, ci=None)

    assert isinstance(pred, float)
    expected = result.exponential_fit.parameters["a"] * np.exp(
        result.exponential_fit.parameters["r"] * (5.0 - result.time_origin)
    )
    assert pred == pytest.approx(expected, rel=1e-9)


def test_predict_with_ci_none_returns_ndarray_for_array_input():
    time, values = _exp_series()
    result = analyze_growth(time, values, n_boot=0)

    preds = result.predict(np.array([5.0, 8.0, 11.0]), ci=None)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (3,)
    assert np.all(np.diff(preds) > 0)  # exponential growth is monotone


def test_predict_with_default_ci_returns_list_of_predictions_for_array():
    time, values = _logistic_series(n=20)
    result = analyze_growth(time, values, n_boot=150, bootstrap_seed=0)

    preds = result.predict([4.0, 6.0, 10.0])

    assert isinstance(preds, list)
    assert len(preds) == 3
    for p in preds:
        assert isinstance(p, Prediction)
        assert p.low <= p.point <= p.high


def test_predict_raises_for_indeterminate_result():
    time, values = _logistic_series(k=200.0, r=0.35, t0=12.0, n=10, start=0.0, stop=5.0)
    result = analyze_growth(time, values, n_boot=0)
    assert result.is_indeterminate

    with pytest.raises(ValueError, match="indeterminate"):
        result.predict(6.0)


def test_predict_works_for_each_preferred_model():
    # exponential
    time, values = _exp_series(a=3.0, r=0.18, n=18)
    result = analyze_growth(time, values, n_boot=0)
    assert result.preferred_model == "exponential"
    assert isinstance(result.predict(11.0, ci=None), float)

    # linear
    time = np.linspace(0.0, 10.0, 16)
    values = 5.0 + 2.0 * time
    result = analyze_growth(time, values, n_boot=0)
    assert result.preferred_model == "linear"
    pred = result.predict(11.0, ci=None)
    assert isinstance(pred, float)
    # Linear extrapolation: 5 + 2*11 = 27
    assert pred == pytest.approx(27.0, rel=1e-3)

    # logistic
    time, values = _logistic_series(k=100.0, r=0.75, t0=5.0, n=22)
    result = analyze_growth(time, values, n_boot=100, bootstrap_seed=0)
    assert result.preferred_model == "logistic"
    assert isinstance(result.predict(15.0), Prediction)


def test_predict_respects_original_time_coordinate():
    # If the user fits on time in [2000, 2014], predict(2020) must use 2020
    # in the original coordinate, not 20 in normalized time.
    time = np.linspace(2000.0, 2014.0, 15)
    values = 2.5e6 * np.exp(0.09 * (time - time[0]))
    result = analyze_growth(time, values, n_boot=0)
    assert result.preferred_model == "exponential"

    # Predict at 2020 (six years past the last observation).
    pred_2020 = result.predict(2020.0, ci=None)

    # Manual extrapolation in original coordinates.
    a = result.exponential_fit.parameters["a"]
    r = result.exponential_fit.parameters["r"]
    expected = a * np.exp(r * (2020.0 - 2000.0))
    assert pred_2020 == pytest.approx(expected, rel=1e-6)


def test_predict_at_observed_time_recovers_value():
    # Sanity: predicting at one of the input times should recover roughly
    # the observed value on a clean fit.
    time = np.linspace(0.0, 10.0, 16)
    values = 5.0 + 2.0 * time
    result = analyze_growth(time, values, n_boot=0)

    pred = result.predict(time[5], ci=None)
    assert pred == pytest.approx(values[5], rel=1e-6)


def test_predict_bootstrap_is_deterministic_with_seed():
    time, values = _logistic_series(n=20)
    result = analyze_growth(time, values, n_boot=100, bootstrap_seed=0)

    a = result.predict(8.0, n_boot=100, seed=42)
    b = result.predict(8.0, n_boot=100, seed=42)
    assert a == b


def test_input_arrays_are_immutable_after_construction():
    time, values = _exp_series()
    result = analyze_growth(time, values, n_boot=0)

    with pytest.raises(ValueError):
        result.input_time[0] = -1.0
    with pytest.raises(ValueError):
        result.input_values[0] = -1.0


def test_bootstrap_predictions_helper_returns_intervals_per_time():
    time, values = _logistic_series(n=20)

    intervals = bootstrap_predictions(
        time,
        values,
        model_name="logistic",
        prediction_times=[4.0, 6.0, 10.0],
        n_boot=100,
        seed=0,
    )

    assert len(intervals) == 3
    for interval in intervals:
        assert interval.low <= interval.median <= interval.high


def test_bootstrap_predictions_rejects_unknown_model():
    time, values = _exp_series()
    with pytest.raises(ValueError, match="unsupported model_name"):
        bootstrap_predictions(
            time, values, model_name="quadratic", prediction_times=[1.0], n_boot=10, seed=0
        )
