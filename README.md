# Project Verge

Project Verge is a Python library for one narrow question:

Given a positive, increasing time series, is the observed growth better explained by a simple exponential curve or by the early part of a logistic S-curve?

The package returns an approximate probability for each model using posterior model weights derived from Bayesian Information Criterion (BIC) under a shared log-scale observation model.

## What v1 does

- Fits exponential and logistic growth curves to a univariate time series.
- Returns `p_exponential` and `p_logistic` as approximate model-evidence weights.
- Flags ambiguous cases as `indeterminate` instead of overclaiming.
- Reports supporting diagnostics:
  - per-capita growth slope versus level
  - log-residual curvature
  - forward-chaining one-step forecast error comparison
  - logistic identifiability warnings

## What v1 does not do

- It does not prove a system will saturate in the real world.
- It does not calibrate probabilities for every domain.
- It does not yet support non-monotone, highly noisy, or multivariate series.
- It does not include a CLI in the initial release.

## Installation

```bash
pip install project-verge
```

For local development:

```bash
python3 -m pip install -e .[dev]
```

## Quick Start

```python
import numpy as np
from project_verge import analyze_growth

time = np.linspace(0.0, 12.0, 18)
values = 120.0 / (1.0 + np.exp(-0.7 * (time - 6.0)))

result = analyze_growth(time, values)

print(result.p_exponential)
print(result.p_logistic)
print(result.preferred_model)
print(result.is_indeterminate)
print(result.diagnostics.identifiability_warnings)
```

## API

### `analyze_growth(time, values, *, prior_exponential=0.5, prior_logistic=0.5, min_points=8)`

Runs the full analysis and returns a `GrowthAnalysis` object.

### `fit_exponential(time, values)`

Fits the exponential model and returns a `ModelFit`.

### `fit_logistic(time, values)`

Fits the logistic model and returns a `ModelFit`.

## Input Contract

Version 1 assumes:

- `time` and `values` are one-dimensional sequences of equal length
- `time` is strictly increasing
- `values` is strictly positive
- `values` is nondecreasing
- at least 8 observations are provided for `analyze_growth`

## Interpreting the Probability

The headline probabilities are:

- conditioned on the exponential and logistic models being the two candidates
- conditioned on the log-normal observation model
- approximate, because BIC is used as a tractable model-evidence proxy

They should be read as model-comparison evidence, not as a universal forecast probability that a real-world process must plateau.

## Repository Layout

- `src/project_verge/`: library source
- `tests/`: unit tests
- `examples/demo_growth_analysis.py`: runnable demonstration
- `PROJECT_PLAN.md`: living tracker for milestones, review items, and backlog

## Development

Run the test suite with:

```bash
PYTHONPATH=src python3 -m pytest
```

