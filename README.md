# Project Verge

Project Verge answers one focused question about a positive, growing time series:

> **Is there evidence this is leveling off, or evidence it's still going up?**

It compares three candidate trajectories — pure exponential growth, linear growth, and the early part of a logistic S-curve — using posterior model weights derived from BIC under a shared log-normal observation model, and returns one of four verdicts:

- **accelerating** — exponential growth fits best; the rate is increasing
- **steady** — linear growth fits best; growing at a constant rate, no acceleration and no asymptote in the data
- **leveling off** — an S-curve fits best, with bootstrap percentile intervals for the carrying capacity `K`, the inflection time `t0`, and any prediction horizons you ask for
- **indeterminate** — the data does not contain enough evidence to choose, with a structured reason explaining why

The "indeterminate" branch is intentional. A real-world series with only a handful of early data points often *cannot* be classified honestly, and Verge would rather say "I can't tell yet" than overclaim a confident verdict.

## What v1 answers

- The verdict (`accelerating` / `steady` / `leveling off` / `indeterminate`) and a one-line `summary()` you can `print()`.
- Posterior model weights `p_exponential`, `p_linear`, and `p_logistic` as approximate model-comparison evidence (BIC under a shared log-normal observation model).
- When the logistic verdict is the focus, a pair-bootstrap percentile interval for the carrying capacity `K`, the inflection time `t0`, and predicted values at any horizons you supply.
- A structured `indeterminate_reason` so callers can branch on *why* a verdict is being withheld:
  - `neither_model_fits` — none of the candidate models (exponential, linear, logistic) explains the data well on the log scale (e.g. polynomial growth)
  - `ambiguous_evidence` — no model is decisively preferred by BIC
  - `logistic_unidentifiable` — the logistic bend is not pinned down by the observed window
  - `signal_disagreement` — BIC prefers logistic but the supporting diagnostics (per-capita slope, log-residual curvature, forecast MAE) do not all agree
- A `Diagnostics.signal_agreement` flag set giving the three supporting signals individually, plus `levelling_off_votes` (0–3) for the aggregate.
- Supporting diagnostics: per-capita growth slope, log-residual curvature, and forward-chaining one-step forecast error for each candidate model.

## What v1 does not answer

- Whether a real-world process will *actually* saturate. Verge reports the evidence in the data under explicit modeling assumptions, not a guarantee about the future.
- Verdicts on noisy or non-monotone series. The v1 input contract is intentionally narrow; see *Input Contract* below.
- A choice between richer S-curve families (Gompertz, Richards, etc.) or richer sub-exponential families (power-law, Gompertz-of-log, etc.). v1 only compares pure exponential, plain linear, and the standard logistic.
- A CLI. v1 is library-only.

## Installation

```bash
pip install project-verge
```

For local development, install the package in editable mode before running tests or examples:

```bash
python3 -m pip install -e '.[dev]'
```

## Quick Start

```python
import numpy as np
from project_verge import analyze_growth

# 1) Series with clear S-curve shape — leveling off detected.
time = np.linspace(0.0, 12.0, 18)
values = 120.0 / (1.0 + np.exp(-0.7 * (time - 6.0)))
print(analyze_growth(time, values, n_boot=200, bootstrap_seed=0).summary())
print()

# 2) Pure exponential growth — no leveling-off signal, rate is increasing.
time = np.linspace(0.0, 12.0, 18)
values = 3.0 * np.exp(0.18 * time)
print(analyze_growth(time, values, n_boot=200, bootstrap_seed=0).summary())
print()

# 3) Linear growth — going up at a constant rate; neither accelerating
#    nor leveling off.
time = np.linspace(0.0, 10.0, 16)
values = 5.0 + 2.0 * time
print(analyze_growth(time, values, n_boot=200, bootstrap_seed=0).summary())
print()

# 4) Early-stage S-curve — the carrying capacity is not yet identified by
#    the observed window, so the honest verdict is indeterminate.
time = np.linspace(0.0, 5.0, 10)
values = 200.0 / (1.0 + np.exp(-0.35 * (time - 12.0)))
print(analyze_growth(time, values, n_boot=200, bootstrap_seed=0).summary())
```

Output:

```
Verdict: leveling off (logistic, 1.00 confidence).
Estimated ceiling K ~= 120 [120, 120].
Estimated inflection time ~= 6 [6, 6].
Per-capita slope: -0.0075; forecast log-MAE (logistic): 6.29e-15.

Verdict: accelerating (exponential, 1.00 confidence).
Per-capita slope: +1.055e-18; forecast log-MAE (exponential): 1.54e-16.

Verdict: steady (linear, 1.00 confidence).
Per-capita slope: -0.0141; forecast log-MAE (linear): 0.

Verdict: indeterminate (reason: logistic_unidentifiable).
The logistic carrying capacity is not identified by the observed window.
Estimated ceiling K ~= 200 [200, 200].
Estimated inflection time ~= 12 [12, 12].
Per-capita slope: -0.002307; posterior weights: exponential 0.00, linear 0.00, logistic 1.00.
```

The bootstrap intervals look implausibly tight here because the demo inputs are perfectly clean synthetic curves; on real noisy data they widen to reflect the true sampling uncertainty in the fit. For programmatic access rather than a printed summary, every value in the rendered output is also a typed attribute on `GrowthAnalysis` — `p_exponential`, `p_linear`, `p_logistic`, `preferred_model`, `is_indeterminate`, `indeterminate_reason`, `logistic_intervals.K`, etc.

## API

### `analyze_growth(time, values, *, prior_exponential=0.5, prior_linear=0.5, prior_logistic=0.5, min_points=8, min_fit_quality=0.85, horizons=None, n_boot=500, bootstrap_confidence=0.90, bootstrap_seed=None)`

Runs the full analysis and returns a `GrowthAnalysis` object.

`min_fit_quality` is the log-space R² floor that each candidate model is held to. When *both* models fall below it, the verdict is forced to `indeterminate` with `indeterminate_reason = "neither_model_fits"`, so a polynomial or other out-of-family series cannot quietly produce a confident-looking exponential-vs-logistic split.

`GrowthAnalysis.indeterminate_reason` is `None` when the verdict is decisive, and otherwise one of `"neither_model_fits"`, `"ambiguous_evidence"`, or `"logistic_unidentifiable"`.

`horizons`, `n_boot`, `bootstrap_confidence`, and `bootstrap_seed` control a pair-bootstrap that fills `GrowthAnalysis.logistic_intervals` with percentile intervals for the logistic `K`, `r`, and `t0`, plus one prediction interval per supplied horizon (in the original time coordinate). The bootstrap runs only when it is actually informative — when the logistic is the preferred model or the verdict is indeterminate — because the optimizer is unidentified on data that is clearly exponential and a bootstrap there would just be expensive decoration. Pass `n_boot=0` to skip the bootstrap entirely.

`GrowthAnalysis.diagnostics.fit_warnings` contains optimizer or fit-process warnings, while `GrowthAnalysis.diagnostics.identifiability_warnings` contains interpretation warnings specific to whether the logistic bend is actually identified by the observed window.

### `fit_exponential(time, values, *, min_points=8)`

Fits the exponential model and returns a `ModelFit`.

### `fit_linear(time, values, *, min_points=8)`

Fits the linear model `y = a + b*t` (in y-units, with residuals computed in log-space for consistency with the other models) and returns a `ModelFit`.

### `fit_logistic(time, values, *, min_points=8)`

Fits the logistic model and returns a `ModelFit`.

## Input Contract

Version 1 assumes:

- `time` and `values` are one-dimensional sequences of equal length
- `time` is strictly increasing
- `values` is strictly positive
- `values` is nondecreasing
- at least 8 observations are provided for `analyze_growth`

## Interpreting the Verdict

The verdict is one of four categorical labels — `accelerating`, `steady`, `leveling off`, or `indeterminate` — chosen from the leading model when a model is clearly preferred *and* the underlying fit passes a quality floor *and* (for logistic) the curve is identified by the observed window. If any of those checks fails, the verdict is forced to `indeterminate` with a structured `indeterminate_reason`.

The reported confidence is the posterior model weight of the winning model, approximated from BIC under the shared log-normal observation model. Those weights are:

- conditioned on exponential, linear, and logistic being the three candidates
- conditioned on the log-normal observation model
- approximate, because BIC is used as a tractable proxy for full Bayesian model evidence

They should be read as model-comparison evidence, not as a universal forecast probability that a real-world process must (or must not) plateau. The v1 model space is narrow on purpose; the [TICKETS](TICKETS.md) backlog tracks planned extensions to richer S-curve families.

Bootstrap intervals on `K`, `t0`, and prediction horizons are pair-bootstrap percentile intervals at the configured confidence level (default 90%). They quantify how much the logistic fit moves under resampling. A wide interval when the logistic is the preferred verdict is a load-bearing honesty signal — it means the data does not yet pin down where the plateau is — not a bug.

## Repository Layout

- `src/project_verge/`: library source
- `tests/`: unit tests
- `examples/demo_growth_analysis.py`: runnable demonstration
- `docs/methodology.md`: design notebook walking through the diagnostic intuitions
- `PROJECT_PLAN.md`: project plan and decision log
- `TICKETS.md`: prioritized backlog of methodology, code, and docs work

## Development

Run the test suite with:

```bash
python3 -m pytest
```

Run linting with:

```bash
python3 -m ruff check .
```
