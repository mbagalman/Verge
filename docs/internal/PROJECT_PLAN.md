# GrowthShape Plan and Tracker

## Project Goal

Build a public Python library that compares exponential and logistic growth for a positive, increasing time series and returns approximate model-evidence probabilities with supporting diagnostics.

## Frozen MVP Decisions

- Library-first release. No CLI in v1.
- Logistic is the only S-curve proxy in v1.
- The headline probability is a posterior model weight approximation derived from BIC under a shared log-scale observation model.
- The result should prefer `indeterminate` over false certainty when evidence is weak or logistic parameters are poorly identified.
- v1 input contract is intentionally narrow: univariate, finite, positive, nondecreasing data with strictly increasing time.

## Milestone Checklist

- [x] Create Python package scaffold with `src/` layout
- [x] Implement `fit_exponential`, `fit_logistic`, and `analyze_growth`
- [x] Add dataclass result types for fits, diagnostics, and full analysis output
- [x] Add validation for the v1 data contract
- [x] Add supporting diagnostics and indeterminate-state logic
- [x] Add tests for model recovery, classification, ambiguity, and validation
- [x] Add README, example script, license, and CI workflow
- [x] Publish repository to GitHub
- [x] Add package release workflow and first tagged release

## Current Status / Next Actions

- Core MVP implementation is in place in `src/growthshape/`.
- Tests cover the key statistical and validation paths.
- Immediate next step: review package metadata one more time, add a release workflow, and decide whether to publish to PyPI after one external review.

## Open Questions and Risks

- BIC-derived posterior weights are useful and lightweight, but they are still an approximation to full Bayesian model evidence.
- Logistic identifiability remains weak in very early-stage data even when the optimizer converges numerically.
- Real-world series with noise, seasonality, step changes, or short pullbacks are out of scope for v1.
- The current API intentionally avoids domain-specific priors; that may matter for practical forecasting use cases.

## Feature Backlog

- Add a CLI for CSV-driven workflows.
- Add richer sigmoidal families such as Gompertz or Richards curves.
- Support robust fitting for mildly non-monotone or noisy growth series.
- Add plotting helpers and a notebook-based tutorial.
- Add domain priors or constraint inputs when a carrying-capacity bound is known.

## Code Review Issues

- Resolved on 2026-03-21:
  - Added `min_points` to `fit_exponential` and `fit_logistic`.
  - Rejected non-finite priors explicitly.
  - Unified public wording on the shared log-normal observation model.
  - Documented the extra BIC parameter count for the observation-noise scale.
  - Relaxed the logistic lower bound on `K` and improved the midpoint initialization heuristic.
  - Kept the internal `min_points` guard, but documented why it remains for rolling-window callers.
  - Split `Diagnostics.fit_warnings` from `Diagnostics.identifiability_warnings`.
  - Reused shared curve helpers inside forward-chaining forecasts and documented the asymmetric minimum training windows.
  - Made `ModelFit.fitted_values` read-only by copying and freezing the array.
  - Added package metadata for PyPI discoverability.
  - Added `ruff` linting to CI and aligned README development instructions with the CI install flow.
- Assessed and not changed:
  - The per-capita regression uses time differences only, so shifting the series origin to zero does not change the reported slope or intercept. A clarifying code comment was added instead of changing the diagnostic.

### Round 2 — open issues (2026-03-21)

- Resolved on 2026-03-21:
  - Restored the logistic `K` lower bound to remain just above the observed maximum.
  - Documented `Diagnostics.fit_warnings` in the README and surfaced it in the Quick Start snippet.
  - Added negative tests to confirm custom `min_points` enforcement.
  - Verified that the published Git remote matches `https://github.com/mbagalman/growthshape`, so the current `pyproject.toml` URLs are correct.

## Decision Log

- 2026-03-21: Chose a library-first MVP instead of building a CLI first.
- 2026-03-21: Defined the headline probability as model evidence under explicit assumptions, not as a universal forecast truth score.
- 2026-03-21: Limited v1 support to exponential vs logistic comparison on positive, nondecreasing series.
- 2026-03-21: Accepted the first review round, with one per-capita diagnostic note treated as a documentation clarification rather than a behavioral bug.
