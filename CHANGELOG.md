# Changelog

All notable changes to GrowthShape are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the library is at the `0.x.y` series the public API is allowed to change between minor versions; once it stabilizes, breaking changes will require a major bump.

## [Unreleased]

(no changes yet)

## [0.1.0] - 2026-05-08

Initial release.

### Verdict surface

- Four-way verdict: `accelerating` (exponential winner), `steady` (linear winner), `leveling off` (logistic winner), or `indeterminate`.
- Six structured `indeterminate_reason` values describing precisely why a verdict was withheld: `neither_model_fits`, `power_law_shape`, `ambiguous_evidence`, `logistic_unidentifiable`, `signal_disagreement`, `fragile_verdict`. Ordered by precedence; documented in the README.
- Human-readable `GrowthAnalysis.summary()` (also bound to `__repr__`) producing a concise multi-line verdict suitable for `print(result)`.

### Modeling

- Four candidate models compete on the BIC/AICc score: exponential, linear, logistic, and power-law.
- Power-law is diagnostic-only; when it wins the chosen information-criterion competition the verdict is forced to `indeterminate (reason: power_law_shape)` because the user's "going up vs leveling off" question has no clean answer for power-law growth.
- `criterion="aicc"` (default) or `"bic"` selects the information criterion. AICc is the default because the small-sample correction matters at the typical input sizes GrowthShape sees (`n` between 8 and 30).
- Multi-start optimization on the logistic fit (`n_starts=8` by default), sweeping K and t0 across plausible initial guesses to avoid local minima.
- `evidence_strength` parameter mapped to Kass & Raftery's (1995) interpretive bands: `"positive"` (≥ 0.75), `"strong"` (≥ 0.95, default), `"decisive"` (≥ 0.99). Below the band threshold, the verdict is forced to `indeterminate (reason: ambiguous_evidence)`.

### Result API

- `analyze_growth(time, values, ...)` returns a `GrowthAnalysis` dataclass with the verdict, four `ModelFit` objects, the bootstrap intervals (when run), the diagnostics, the assumption record, and the captured input arrays.
- `GrowthAnalysis.predict(time, *, ci=0.9)` for vectorized future-value prediction; returns `Prediction(low, point, high)` namedtuples by default.
- Public `fit_exponential`, `fit_linear`, `fit_logistic`, `fit_power_law` wrappers for callers who want a single-model fit without the comparison machinery.
- All public types (`GrowthAnalysis`, `ModelFit`, `Diagnostics`, `BootstrapIntervals`, `WeightIntervals`, `ForecastDiagnostic`, `SignalAgreement`, `AnalysisAssumptions`, `Interval`, `Prediction`) plus `Literal` aliases (`ModelName`, `PreferredModel`, `IndeterminateReason`) exported from `growthshape` for type-narrowing in consumer code.

### Uncertainty quantification

- Bootstrap percentile intervals on the logistic carrying capacity `K`, growth rate `r`, inflection time `t0`, and predicted values at any user-supplied horizons (`result.logistic_intervals`).
- Bootstrap percentile intervals on the four-way model weights themselves (`result.weight_intervals`) — the verdict's own headline confidence comes with a confidence interval.
- A `fragile_verdict` indeterminate gate auto-downgrades verdicts when the weight CI on the leading model is wider than `max_weight_ci_width` (default 0.40), catching cases that pass every other gate but rest on unstable evidence.

### Diagnostics

- Per-capita growth slope with t-statistic and one-sided p-value.
- Log-residual curvature with t-statistic and one-sided p-value.
- Forward-chaining one-step forecast errors per candidate model, aggregated as `(median_log_error, convergence_rate, n_windows)` so a single failed rolling fit does not poison the whole metric.
- Shapiro-Wilk normality and (hand-rolled) Ljung-Box autocorrelation tests on the leading-model log-residuals; warnings on `Diagnostics.assumption_warnings` when either p-value drops below 0.05.
- A `signal_agreement` flag set with three boolean components and an aggregate vote count, used for the `signal_disagreement` indeterminate gate.

### Input handling

- Strict input contract by default: positive, finite, strictly-increasing time, nondecreasing values, at least 8 observations.
- Optional `allow_smoothing=True` runs a rolling-median + cumulative-max pre-fit smoother that admits noisy real-world data; the transformation is recorded in `result.transform_log`.
- Up-front scope checks reject flat data (`min_relative_range`, default 1%) and decreasing data with clear error messages explaining why those inputs are outside GrowthShape's scope.

### Visualization

- Optional `plot_growth_analysis()` helper installable via the `[plot]` extra (`pip install 'growthshape[plot]'`). Single-figure summary with input data, all four fits, the K asymptote when logistic preferred, and a 90% prediction envelope when bootstrap data is available.

### Documentation and examples

- README reframed around the user's question ("is this leveling off, or still going up?") with Quick Start, Predicting future values, Visualizing the result, Worked example: world population, How calibrated are these probabilities?, Failure modes, API, Input Contract, and Interpreting the Verdict sections.
- `docs/internal/methodology_notes.md` preserved pre-package design notebook walking through the diagnostic intuitions (kept under `docs/internal/` because the snippets are ad-hoc and pre-date the shipped API; the README's *Interpreting the Verdict* and *Failure modes* sections are the canonical methodology surfaces).
- `docs/glossary.md` with plain-English definitions for ~30 statistics terms in the API.
- `docs/calibration.png` reproducible from `examples/calibration.py` (900 seeded synthetic trials).
- `examples/demo_growth_analysis.py` for synthetic series, `examples/demo_plot.py` for the visualization helper, `examples/world_population.py` for a real-data worked example using UN World Population Prospects estimates from 1750 to 2022.
- 221 tests covering happy paths, noisy variants, wrong-model graceful behavior, calibration, real-data smoke tests, time-origin invariance, and per-feature unit tests.

---

The development that produced this release was ticket-driven; the per-ticket audit trail lives in [docs/internal/TICKETS.md](docs/internal/TICKETS.md), and the project decisions log lives in [docs/internal/PROJECT_PLAN.md](docs/internal/PROJECT_PLAN.md).
