# Project Verge Plan and Tracker

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
- [ ] Publish repository to GitHub
- [ ] Add package release workflow and first tagged release

## Current Status / Next Actions

- Core MVP implementation is in place in `src/project_verge/`.
- Tests cover the key statistical and validation paths.
- Immediate next step: create the GitHub repository, review naming and packaging metadata, and decide whether to publish to PyPI after one external review.

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

- No open code review issues yet.

## Decision Log

- 2026-03-21: Chose a library-first MVP instead of building a CLI first.
- 2026-03-21: Defined the headline probability as model evidence under explicit assumptions, not as a universal forecast truth score.
- 2026-03-21: Limited v1 support to exponential vs logistic comparison on positive, nondecreasing series.
