"""Optional matplotlib visualization for :class:`GrowthAnalysis`.

This module is opt-in and requires matplotlib. Install with::

    pip install 'project-verge[plot]'
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - only for type checking
    from matplotlib.axes import Axes

    from ._types import GrowthAnalysis


_MODEL_COLOR = {
    "exponential": "tab:blue",
    "linear": "tab:green",
    "logistic": "tab:red",
}


_VERDICT_LABEL = {
    "exponential": "accelerating",
    "linear": "steady",
    "logistic": "leveling off",
    "indeterminate": "indeterminate",
}


def plot_growth_analysis(
    result: "GrowthAnalysis",
    ax: Optional["Axes"] = None,
    *,
    extrapolate_fraction: float = 0.2,
    show_alternatives: bool = True,
    envelope: Optional[bool] = None,
) -> "Axes":
    """Plot a :class:`GrowthAnalysis` result on a matplotlib ``Axes``.

    The figure includes the input data, each candidate model's fitted curve
    (the preferred model drawn boldly, the others as thin dashed lines when
    ``show_alternatives`` is True), the logistic carrying-capacity asymptote
    when logistic is preferred, and a bootstrap-derived 90% prediction
    envelope around the preferred model when uncertainty data is available.

    Parameters
    ----------
    result : GrowthAnalysis
        The result returned by :func:`analyze_growth`.
    ax : matplotlib Axes, optional
        Where to draw. A new figure / axes is created if not provided.
    extrapolate_fraction : float, default 0.2
        Fraction of the observed time span to extend past the last
        observation. ``0.0`` plots only the observed range.
    show_alternatives : bool, default True
        Draw the non-preferred model fits as faint dashed lines.
    envelope : bool, optional
        Whether to draw the 90% bootstrap prediction envelope. The default
        ``None`` enables the envelope when bootstrap data is already
        available on the result (i.e. when ``result.weight_intervals`` is
        populated and the verdict is decisive); pass ``True`` to force it
        on (incurs a fresh bootstrap inside :meth:`GrowthAnalysis.predict`)
        or ``False`` to suppress it.

    Returns
    -------
    matplotlib Axes
        The axes the plot was drawn on.

    Raises
    ------
    ImportError
        If matplotlib is not installed. Install via
        ``pip install 'project-verge[plot]'``.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "matplotlib is required for plot_growth_analysis. "
            "Install with: pip install 'project-verge[plot]'"
        ) from exc

    # Local imports keep ``project_verge.plot`` importable without matplotlib
    # and avoid a _types -> plot import cycle.
    from ._fit import exponential_curve, linear_curve, logistic_curve

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    time = result.input_time
    values = result.input_values
    span = float(time[-1] - time[0]) if len(time) > 1 else 1.0
    x_max = float(time[-1] + extrapolate_fraction * span)
    x_grid = np.linspace(float(time[0]), x_max, 250)
    x_grid_norm = x_grid - result.time_origin

    ax.scatter(time, values, color="black", s=30, zorder=5, label="data")

    fits_with_curves = (
        ("exponential", result.exponential_fit, _exponential_y(exponential_curve)),
        ("linear", result.linear_fit, _linear_y(linear_curve)),
        ("logistic", result.logistic_fit, _logistic_y(logistic_curve)),
    )

    for name, fit, curve_evaluator in fits_with_curves:
        if not fit.parameters:
            continue
        y_grid = curve_evaluator(x_grid_norm, fit.parameters)
        color = _MODEL_COLOR[name]
        if result.is_indeterminate:
            # No model is preferred; draw all three with equal prominence.
            ax.plot(x_grid, y_grid, color=color, linewidth=1.5, label=name)
        elif name == result.preferred_model:
            ax.plot(
                x_grid,
                y_grid,
                color=color,
                linewidth=2.5,
                label=f"{name} (preferred)",
            )
        elif show_alternatives:
            ax.plot(
                x_grid,
                y_grid,
                color=color,
                linewidth=1.0,
                linestyle="--",
                alpha=0.5,
                label=name,
            )

    if result.preferred_model == "logistic" and result.logistic_fit.parameters:
        K = result.logistic_fit.parameters["K"]
        ax.axhline(
            K,
            color=_MODEL_COLOR["logistic"],
            linestyle=":",
            alpha=0.6,
            label=f"K ≈ {K:.3g}",
        )

    if envelope is None:
        envelope = (
            result.weight_intervals is not None
            and not result.is_indeterminate
        )
    if envelope and not result.is_indeterminate:
        try:
            predictions = result.predict(x_grid, ci=0.9, n_boot=200, seed=0)
        except ValueError:
            predictions = None
        if predictions is not None:
            lows = np.array([p.low for p in predictions])
            highs = np.array([p.high for p in predictions])
            ax.fill_between(
                x_grid,
                lows,
                highs,
                color=_MODEL_COLOR[result.preferred_model],
                alpha=0.15,
                label="90% envelope",
            )

    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.set_title(_format_title(result))
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    return ax


def _exponential_y(curve):
    return lambda t, p: curve(t, p["a"], p["r"])


def _linear_y(curve):
    return lambda t, p: curve(t, p["a"], p["b"])


def _logistic_y(curve):
    return lambda t, p: curve(t, p["K"], p["r"], p["t0"])


def _format_title(result: "GrowthAnalysis") -> str:
    label = _VERDICT_LABEL.get(result.preferred_model, result.preferred_model)
    if result.is_indeterminate:
        return f"Verge: {label} (reason: {result.indeterminate_reason})"
    confidence = {
        "exponential": result.p_exponential,
        "linear": result.p_linear,
        "logistic": result.p_logistic,
    }.get(result.preferred_model, 0.0)
    return f"Verge: {label} ({result.preferred_model}, {confidence:.2f} confidence)"
