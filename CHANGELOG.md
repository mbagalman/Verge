# Changelog

All notable changes to Project Verge are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the library is at the `0.x.y` series the public API is allowed to change between minor versions; once it stabilizes, breaking changes will require a major bump.

## [Unreleased]

The library has accumulated a substantial body of work since the v0.1.0 MVP — covering methodology improvements, an honest treatment of failure modes, and a much richer result surface. It has not yet been tagged as a release; the work below is queued for v0.2.0 ("Real question") whenever the release workflow lands ([T-26](TICKETS.md)).

### Added

- **Power-law shape detection** ([T-27](TICKETS.md)). A fourth diagnostic-only candidate (`y = a · (t + 1)^k`) competes for posterior weight; when it wins, the verdict is forced to `indeterminate (reason: power_law_shape)` rather than misclassifying polynomial growth as logistic.
- **AICc as the default information criterion** ([T-12](TICKETS.md)). Replaces BIC as the default for the four-way model competition, with an explicit `criterion` parameter to opt back into BIC. Defended in the README; matches the small-sample regression literature for `n` in the typical 8–30 range.
- **Bootstrap percentile intervals on the logistic K, r, t0, and prediction horizons** ([T-02](TICKETS.md)). Returned on `result.logistic_intervals`. The bootstrap is gated to run only when the logistic verdict is the focus.
- **Bootstrap percentile intervals on the model weights themselves** ([T-07](TICKETS.md)). `result.weight_intervals` exposes how much the headline confidence number could swap under resampling.
- **`GrowthAnalysis.predict(time, *, ci=0.9)`** ([T-08](TICKETS.md)). Vectorized prediction at one or more future times; returns `Prediction(low, point, high)` namedtuples by default. Raises for indeterminate verdicts.
- **`plot_growth_analysis()` helper** ([T-09](TICKETS.md)). Optional matplotlib visualization (install via the `[plot]` extra). Single-figure summary with data, all four fits, K asymptote when logistic preferred, and 90% prediction envelope when bootstrap data is available.
- **Noise-tolerant smoothing path** ([T-15](TICKETS.md)). `analyze_growth(allow_smoothing=True)` runs a rolling-median + cumulative-max pre-fit smoother that admits noisy real-world data. Recorded in `result.transform_log` for auditability.
- **Multi-start optimization for the logistic fit** ([T-16](TICKETS.md)). `n_starts=8` by default; sweeps K and t0 across plausible initial guesses. Bootstrap path stays single-start.
- **Log-normal assumption checks** ([T-13](TICKETS.md)). Shapiro-Wilk on log-residual normality and a hand-rolled Ljung-Box on autocorrelation. P-values on `Diagnostics.residual_normality_pvalue` / `residual_autocorr_pvalue`; warnings on `Diagnostics.assumption_warnings`.
- **Evidence-strength bands** ([T-14](TICKETS.md)). `evidence_strength` parameter mapped to Kass & Raftery's positive (≥0.75) / strong (≥0.95, default) / decisive (≥0.99) thresholds, replacing the previous magic 0.70 cutoff.
- **`fragile_verdict` indeterminate gate** ([T-28](TICKETS.md)). Auto-downgrades verdicts when the bootstrap weight CI on the leading model is wider than `max_weight_ci_width` (default 0.40). Catches noise-dominated cases that pass every earlier gate but rest on unstable evidence.
- **`signal_disagreement` indeterminate gate** ([T-06](TICKETS.md)). Logistic-only second-opinion check: per-capita-slope significance, log-residual-curvature significance, and forecast-MAE direction must all agree. Asymmetric by design.
- **Structured forecast diagnostic** ([T-18](TICKETS.md)). The three `forecast_*` fields on `Diagnostics` are now `ForecastDiagnostic` namedtuples carrying `median_log_error`, `convergence_rate`, and `n_windows`. The previous mean-of-errors aggregator collapsed to `inf` whenever any single rolling fit failed; the median ignores failed windows by construction.
- **Structured `AnalysisAssumptions`** ([T-23](TICKETS.md)). `result.assumptions` is now a dataclass recording the criterion, evidence-strength band, candidate models, and observation model — replacing the previous static prose tuple.
- **Up-front scope checks**. `min_relative_range` (default 0.01) rejects flat data with no growth signal up front; the decreasing-data error message reframes the rejection as a scope issue and points users at `allow_smoothing=True`.
- **`Literal` type aliases** ([T-20](TICKETS.md)). New `ModelName`, `PreferredModel`, and `IndeterminateReason` aliases replace `str` annotations on the corresponding fields. Catches typos at type-check time.
- **`numpy.typing.ArrayLike` annotations** ([T-21](TICKETS.md)) replace the previous `Sequence[float]` everywhere; matches numpy convention and admits the array shapes the runtime actually accepts.
- **Real-data example** ([T-11](TICKETS.md)). `examples/world_population.py` runs Verge on UN World Population Prospects estimates from 1750 through 2022, with both full-history and post-1950 windows, demonstrating how the verdict honestly depends on the analysis window.
- **Calibration evidence in docs** ([T-19](TICKETS.md)). `examples/calibration.py` runs 900 seeded synthetic trials and writes `docs/calibration.png`; the README's "How calibrated are these probabilities?" section walks through the methodology and reads the plot honestly.
- **Glossary** ([T-24](TICKETS.md)). `docs/glossary.md` covers ~30 statistics and methodology terms in plain English.
- **Failure-modes documentation** ([T-10](TICKETS.md)). README's "Failure modes" section catalogs the input patterns Verge does not handle gracefully and points at mitigations.

### Changed

- **Verdict surface is now four-way** ([T-05](TICKETS.md)). The previous binary "exponential vs logistic" frame became `accelerating` / `steady` / `leveling off` / `indeterminate` after a linear candidate was added. The previous `still growing` label was renamed to `accelerating` for parallelism.
- **`GrowthAnalysis.summary()` and `__repr__`** ([T-03](TICKETS.md)). `print(result)` now produces a human-readable verdict line (e.g. `"Verdict: leveling off (logistic, 1.00 confidence; 90% CI [1.00, 1.00])."`) plus a concise diagnostic block, instead of the raw dataclass repr.
- **README is reframed around the user's question** ([T-04](TICKETS.md)). The opening leads with "is this leveling off, or still going up?"; the Quick Start uses `summary()` and covers all four verdict types; "Interpreting the Verdict" replaces the previous "Interpreting the Probability" section.
- **`forecast_mae_*` fields renamed to `forecast_*`** ([T-18](TICKETS.md)) and changed type from `float` to `ForecastDiagnostic`.
- **`assumptions` field type changed** ([T-23](TICKETS.md)) from `Tuple[str, ...]` of static prose to `AnalysisAssumptions` dataclass.

### Fixed

- **Polynomial growth no longer silently misclassifies as logistic** ([T-01, T-27](TICKETS.md)). Previously, `y = t**3` quietly classified as `leveling off` because the logistic fit cleared the default fit-quality floor. The combination of T-01's `neither_model_fits` exit and T-27's `power_law_shape` gate now catches this honestly.
- **`_bounds_logistic` no longer floors `max_value` at 1.0** (surfaced during [T-16](TICKETS.md)). Previously, very-small-magnitude data (max < 1.0) caused the logistic K's lower bound to land *above* the heuristic K initial guess, and `least_squares` raised "Initial guess is outside of provided bounds."
- **Forecast aggregator no longer collapses to `inf` when one rolling window fails** ([T-18](TICKETS.md)). Median over converged windows replaces mean over all windows; convergence rate exposed separately.
- **Decreasing-input error message** rewritten to communicate scope, not just contract. Previously read as an arbitrary v1 limitation; now explains why the rejection happens and points at `allow_smoothing=True`.

### Refactored

- **`min_points` duplicate validation** ([T-22](TICKETS.md)). The redundant inner check in `_fit_model` and `fit_power_law_model` is now framed as an internal-invariant `RuntimeError`, distinct from the user-facing `ValueError` raised by `prepare_inputs`.

### Removed

- **Static prose `assumptions` tuple** ([T-23](TICKETS.md)). Replaced by structured `AnalysisAssumptions`.
- **Single-start-only `_fit_model`** ([T-16](TICKETS.md)). `_fit_model` now takes `initial_guesses: list` instead of `initial_guess: ndarray`; for non-multi-start callers, pass a single-element list (no behaviour change).

## [0.1.0] - 2026-03-21

### Added

- Initial MVP. Compares an exponential and a logistic fit on a positive nondecreasing time series; returns posterior model weights derived from BIC under a shared log-normal observation model. Supporting diagnostics: per-capita growth slope, log-residual curvature, forward-chaining one-step forecast error, logistic identifiability warnings. Indeterminate verdict for ambiguous cases.
