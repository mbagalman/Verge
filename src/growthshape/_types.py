from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Mapping, NamedTuple, Optional, Tuple, Union

import numpy as np


# Public type aliases. ``ModelName`` is the set of model identifiers any
# ModelFit can carry (the four candidate models that compete on BIC/AICc).
# ``PreferredModel`` is narrower: power-law is a diagnostic-only candidate
# and never becomes a verdict (the gate forces ``indeterminate`` first).
# ``IndeterminateReason`` enumerates the structured indeterminate gate
# triggers in their precedence order. Each catches typos at type-check
# time without changing runtime behaviour.
ModelName = Literal["exponential", "linear", "logistic", "power_law"]
PreferredModel = Literal["exponential", "linear", "logistic", "indeterminate"]
IndeterminateReason = Literal[
    "neither_model_fits",
    "power_law_shape",
    "ambiguous_evidence",
    "logistic_unidentifiable",
    "signal_disagreement",
    "fragile_verdict",
]


class Prediction(NamedTuple):
    """A point prediction with bootstrap percentile bounds.

    ``point`` is the analytical prediction from the preferred-model fit
    (i.e. what you would get by plugging the fit parameters into the
    curve formula). ``low`` and ``high`` are pair-bootstrap percentile
    bounds at the confidence level requested in
    :meth:`GrowthAnalysis.predict`.
    """

    low: float
    point: float
    high: float


class ForecastDiagnostic(NamedTuple):
    """Forward-chaining one-step forecast performance for a single model.

    ``median_log_error`` is the median of the absolute log-space errors over
    the rolling-window forecasts where the model converged. The previous
    aggregator used the *mean* of all errors, which collapsed to ``inf``
    whenever any single window failed to converge -- destroying signal
    from the rest. The median ignores non-converged windows by
    construction (their errors are not included), and the separate
    ``convergence_rate`` makes the unreliability auditable.

    ``convergence_rate`` is the fraction of attempted rolling windows
    where the model actually converged, in ``[0, 1]``. ``n_windows`` is
    the total number of rolling windows attempted, useful for
    distinguishing low convergence-rate at small n (few windows) from
    low convergence-rate at large n (genuine failure-prone fitting).

    When zero windows converge, ``median_log_error`` is NaN. Downstream
    consumers (e.g. the ``logistic_has_best_forecast`` signal) filter out
    NaN candidates.
    """

    median_log_error: float
    convergence_rate: float
    n_windows: int


@dataclass(frozen=True)
class ModelFit:
    """Stores the result of fitting a single growth model."""

    model_name: ModelName
    parameters: Mapping[str, float]
    fitted_values: np.ndarray
    log_likelihood: float
    bic: float
    aicc: float
    log_r_squared: float
    converged: bool
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        fitted_values = np.array(self.fitted_values, dtype=float, copy=True)
        fitted_values.setflags(write=False)
        object.__setattr__(self, "fitted_values", fitted_values)


@dataclass(frozen=True)
class SignalAgreement:
    """Boolean flags for the three non-BIC signals that point toward leveling off.

    Each flag is ``True`` when the corresponding signal is consistent with
    logistic (S-curve) growth rather than exponential or linear growth. The
    aggregate vote count is exposed via :attr:`levelling_off_votes`.
    """

    per_capita_slope_negative: bool
    residual_curvature_negative: bool
    logistic_has_best_forecast: bool

    @property
    def levelling_off_votes(self) -> int:
        return (
            int(self.per_capita_slope_negative)
            + int(self.residual_curvature_negative)
            + int(self.logistic_has_best_forecast)
        )


@dataclass(frozen=True)
class Diagnostics:
    """Supporting diagnostics for interpreting the primary model comparison."""

    per_capita_slope: float
    per_capita_intercept: float
    per_capita_slope_std_err: float
    per_capita_slope_t_stat: float
    per_capita_slope_p_value: float
    residual_curvature_score: float
    residual_curvature_std_err: float
    residual_curvature_t_stat: float
    residual_curvature_p_value: float
    forecast_exponential: ForecastDiagnostic
    forecast_linear: ForecastDiagnostic
    forecast_logistic: ForecastDiagnostic
    signal_agreement: SignalAgreement
    residual_normality_pvalue: float
    residual_autocorr_pvalue: float
    fit_warnings: Tuple[str, ...] = ()
    identifiability_warnings: Tuple[str, ...] = ()
    assumption_warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Interval:
    """Percentile interval (low, median, high) at the bootstrap confidence level."""

    low: float
    median: float
    high: float


@dataclass(frozen=True)
class BootstrapIntervals:
    """Bootstrap uncertainty intervals for the logistic fit and any requested horizons.

    ``K`` and ``r`` are reported in the same units as the input ``values`` and
    ``time``. ``t0`` is reported in the *original* time coordinate (i.e. without
    the internal time-origin shift applied by :func:`analyze_growth`).
    ``predicted_intervals`` is one :class:`Interval` per requested horizon, in
    the order the horizons were supplied.
    """

    n_boot: int
    n_successful: int
    confidence: float
    K: Interval
    r: Interval
    t0: Interval
    horizons: Tuple[float, ...]
    predicted_intervals: Tuple[Interval, ...]


@dataclass(frozen=True)
class WeightIntervals:
    """Bootstrap percentile intervals for the criterion-derived posterior weights.

    A narrow interval on the winning weight means the verdict's headline
    confidence is itself robust to resampling; a wide interval means the
    confidence is fragile and the verdict should be read with corresponding
    caution.
    """

    n_boot: int
    n_successful: int
    confidence: float
    p_exponential: Interval
    p_linear: Interval
    p_logistic: Interval
    p_power_law: Interval


@dataclass(frozen=True)
class AnalysisAssumptions:
    """Methodological choices baked into a single :func:`analyze_growth` call.

    These are the dials GrowthShape actually turned, recorded so a downstream
    consumer can reproduce or audit the analysis without having to remember
    which arguments were passed. The prose explanation of *why* each default
    was chosen lives in the README and the function docstring; this
    dataclass is the runtime record of *which* values were used.

    Fields that vary per call:

    * ``criterion`` -- the information criterion used for the four-way
      model competition: ``"aicc"`` (default) or ``"bic"``.
    * ``evidence_strength`` -- the named band the leading-weight threshold
      came from: ``"positive"``, ``"strong"`` (default), or ``"decisive"``.

    Fields fixed in v1 (recorded so future versions can change them
    without breaking forward consumers):

    * ``candidate_models`` -- the model identifiers that competed for
      posterior weight. Always all four in v1.
    * ``observation_model`` -- the noise model assumed when computing
      log-likelihoods. Always ``"log_normal"`` in v1.
    """

    criterion: str
    evidence_strength: str
    candidate_models: Tuple[str, ...]
    observation_model: str


@dataclass(frozen=True)
class GrowthAnalysis:
    """End-to-end result returned by :func:`analyze_growth`."""

    p_exponential: float
    p_linear: float
    p_logistic: float
    p_power_law: float
    preferred_model: PreferredModel
    is_indeterminate: bool
    indeterminate_reason: Optional[IndeterminateReason]
    exponential_fit: ModelFit
    linear_fit: ModelFit
    logistic_fit: ModelFit
    power_law_fit: ModelFit
    diagnostics: Diagnostics
    assumptions: AnalysisAssumptions
    logistic_intervals: Optional[BootstrapIntervals]
    weight_intervals: Optional[WeightIntervals]
    input_time: np.ndarray
    input_values: np.ndarray
    time_origin: float
    transform_log: Tuple[str, ...]

    def __post_init__(self) -> None:
        # Freeze the captured inputs so callers cannot mutate them out from
        # under the result. The arrays back :meth:`predict`'s on-demand
        # bootstrap, so silent mutation would corrupt downstream CIs.
        for field_name in ("input_time", "input_values"):
            arr = np.array(getattr(self, field_name), dtype=float, copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, field_name, arr)

    def summary(self) -> str:
        """Return a short human-readable verdict suitable for ``print(result)``."""

        # Local import: ``_summary`` references ``GrowthAnalysis`` for type
        # checking, so importing it at module load would create a cycle.
        from ._summary import format_summary

        return format_summary(self)

    def __repr__(self) -> str:
        return self.summary()

    def predict(
        self,
        time,
        *,
        ci: Optional[float] = 0.9,
        n_boot: int = 200,
        seed: Optional[int] = 0,
    ) -> Union[float, np.ndarray, Prediction, List[Prediction]]:
        """Predict the value of the preferred-model fit at one or more times.

        ``time`` is interpreted in the same coordinate as the original input
        (no time-origin shift required from the caller). With the default
        ``ci=0.9`` the result is a :class:`Prediction` ``(low, point, high)``
        whose bounds are pair-bootstrap percentile bounds at the requested
        confidence level. Pass ``ci=None`` to get just the point estimate.

        For an indeterminate verdict :meth:`predict` raises ``ValueError`` --
        the whole point of the indeterminate branch is that we don't know
        which model to predict from. Inspect ``exponential_fit`` /
        ``linear_fit`` / ``logistic_fit`` directly if you want a prediction
        from a specific candidate.

        Vectorized over ``time``: a scalar input returns a scalar (or
        :class:`Prediction`); an array input returns an ``ndarray``
        (or list of :class:`Prediction`).
        """

        if self.is_indeterminate:
            raise ValueError(
                f"cannot predict from an indeterminate result "
                f"(reason: {self.indeterminate_reason}). "
                f"Inspect exponential_fit / linear_fit / logistic_fit "
                f"to predict from a specific candidate model."
            )

        # Local imports avoid the _types <-> _fit / _uncertainty cycle.
        from ._fit import exponential_curve, linear_curve, logistic_curve
        from ._uncertainty import bootstrap_predictions

        time_arr = np.asarray(time, dtype=float)
        is_scalar = time_arr.ndim == 0
        times = np.atleast_1d(time_arr)
        times_norm = times - self.time_origin

        if self.preferred_model == "exponential":
            params = self.exponential_fit.parameters
            point = exponential_curve(times_norm, params["a"], params["r"])
        elif self.preferred_model == "linear":
            params = self.linear_fit.parameters
            point = linear_curve(times_norm, params["a"], params["b"])
        elif self.preferred_model == "logistic":
            params = self.logistic_fit.parameters
            point = logistic_curve(
                times_norm, params["K"], params["r"], params["t0"]
            )
        else:
            # power_law is diagnostic-only and never becomes preferred (the
            # gate forces ``indeterminate`` first), so reaching this branch
            # would mean an internal inconsistency.
            raise RuntimeError(
                f"unhandled preferred_model: {self.preferred_model!r}"
            )

        point_array = np.asarray(point, dtype=float)

        if ci is None:
            return float(point_array[0]) if is_scalar else point_array

        intervals = bootstrap_predictions(
            self.input_time,
            self.input_values,
            model_name=self.preferred_model,
            prediction_times=times,
            n_boot=n_boot,
            confidence=ci,
            seed=seed,
        )
        predictions = []
        for i, interval in enumerate(intervals):
            point = float(point_array[i])
            # The analytical point estimate and the bootstrap percentile
            # bounds come from separate computations, so the point can land
            # marginally outside the resampled interval (typically only by
            # floating-point noise when it sits on an edge). Widen the
            # reported bounds to always bracket the point so the interval is
            # guaranteed to contain its own best estimate.
            predictions.append(
                Prediction(
                    low=min(interval.low, point),
                    point=point,
                    high=max(interval.high, point),
                )
            )
        return predictions[0] if is_scalar else predictions
