import numpy as np
import pytest

# Skip the entire module if matplotlib is not installed (the [plot] extra).
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # non-interactive backend for headless test runs
import matplotlib.pyplot as plt  # noqa: E402

from project_verge import analyze_growth  # noqa: E402
from project_verge.plot import plot_growth_analysis  # noqa: E402


def _logistic_series(k=120.0, r=0.7, t0=6.0, n=18, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


def _exp_series(a=4.0, r=0.16, n=15, start=0.0, stop=10.0):
    time = np.linspace(start, stop, n)
    values = a * np.exp(r * (time - time[0]))
    return time, values


def test_plot_returns_axes_and_includes_data_label():
    time, values = _logistic_series(n=20)
    result = analyze_growth(time, values, n_boot=100, bootstrap_seed=0)

    fig, ax = plt.subplots()
    returned = plot_growth_analysis(result, ax=ax)
    try:
        assert returned is ax
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert "data" in labels
        assert any("preferred" in label for label in labels)
    finally:
        plt.close(fig)


def test_plot_includes_K_asymptote_when_logistic_preferred():
    time, values = _logistic_series(n=20)
    result = analyze_growth(time, values, n_boot=100, bootstrap_seed=0)
    assert result.preferred_model == "logistic"

    fig, ax = plt.subplots()
    try:
        plot_growth_analysis(result, ax=ax)
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert any(label.startswith("K") for label in labels)
    finally:
        plt.close(fig)


def test_plot_omits_K_asymptote_when_logistic_not_preferred():
    time, values = _exp_series()
    result = analyze_growth(time, values, n_boot=0)
    assert result.preferred_model == "exponential"

    fig, ax = plt.subplots()
    try:
        plot_growth_analysis(result, ax=ax)
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert not any(label.startswith("K") for label in labels)
    finally:
        plt.close(fig)


def test_plot_envelope_appears_for_logistic_with_bootstrap():
    time, values = _logistic_series(n=20)
    result = analyze_growth(time, values, n_boot=100, bootstrap_seed=0)
    assert result.weight_intervals is not None

    fig, ax = plt.subplots()
    try:
        plot_growth_analysis(result, ax=ax)
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert "90% envelope" in labels
    finally:
        plt.close(fig)


def test_plot_envelope_omitted_when_no_bootstrap_ran():
    # Exponential preferred -> bootstrap is gated off -> no envelope by default.
    time, values = _exp_series()
    result = analyze_growth(time, values, n_boot=0)

    fig, ax = plt.subplots()
    try:
        plot_growth_analysis(result, ax=ax)
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert "90% envelope" not in labels
    finally:
        plt.close(fig)


def test_plot_extrapolates_past_last_observation_by_default():
    time, values = _logistic_series(n=20, start=0.0, stop=10.0)
    result = analyze_growth(time, values, n_boot=100, bootstrap_seed=0)

    fig, ax = plt.subplots()
    try:
        plot_growth_analysis(result, ax=ax, extrapolate_fraction=0.5)
        # ax.dataLim covers all artists; right edge should be past the last
        # observation by roughly the requested fraction (0.5 * 10 = 5).
        right_edge = ax.dataLim.x1
        assert right_edge >= 14.0
    finally:
        plt.close(fig)


def test_plot_handles_indeterminate_without_envelope_or_preferred_label():
    time, values = _logistic_series(k=200.0, r=0.35, t0=12.0, n=10, start=0.0, stop=5.0)
    result = analyze_growth(time, values, n_boot=0)
    assert result.is_indeterminate

    fig, ax = plt.subplots()
    try:
        plot_growth_analysis(result, ax=ax)
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        # No "(preferred)" label when indeterminate.
        assert not any("preferred" in label for label in labels)
        # Title mentions the indeterminate reason.
        assert "indeterminate" in ax.get_title().lower()
    finally:
        plt.close(fig)


def test_plot_creates_its_own_axes_when_none_supplied():
    time, values = _logistic_series(n=20)
    result = analyze_growth(time, values, n_boot=100, bootstrap_seed=0)

    ax = plot_growth_analysis(result)
    try:
        assert ax is not None
        # Verify a figure was created.
        assert ax.get_figure() is not None
    finally:
        plt.close(ax.get_figure())
