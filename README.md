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
- Posterior model weights `p_exponential`, `p_linear`, `p_logistic`, and `p_power_law` as approximate model-comparison evidence (BIC under a shared log-normal observation model). Power-law is a diagnostic-only candidate — it competes for BIC weight but never becomes the preferred verdict; when it wins the result is `indeterminate (reason: power_law_shape)`.
- When the logistic verdict is the focus, a pair-bootstrap percentile interval for the carrying capacity `K`, the inflection time `t0`, and predicted values at any horizons you supply.
- A structured `indeterminate_reason` so callers can branch on *why* a verdict is being withheld:
  - `neither_model_fits` — none of the candidate models (exponential, linear, logistic, power-law) explains the data well on the log scale
  - `power_law_shape` — the data is best described by a power-law shape (`y ≈ a · t^k`), which v1 has no clean verdict for; this catches polynomial growth that would otherwise misclassify as logistic
  - `ambiguous_evidence` — no model is decisively preferred by BIC
  - `logistic_unidentifiable` — the logistic bend is not pinned down by the observed window
  - `signal_disagreement` — BIC prefers logistic but the supporting diagnostics (per-capita slope, log-residual curvature, forecast MAE) do not all agree
  - `fragile_verdict` — BIC favors a single model but the bootstrap CI on its weight is wider than `max_weight_ci_width` (default 0.40), meaning the verdict could swap under resampling
- A `Diagnostics.signal_agreement` flag set giving the three supporting signals individually, plus `levelling_off_votes` (0–3) for the aggregate.
- A `weight_intervals` field on the result with bootstrap percentile intervals on `p_exponential`, `p_linear`, and `p_logistic` so the headline confidence number itself comes with a confidence interval.
- Automatic checks of the log-normal observation assumption: Shapiro-Wilk on the leading-model log-residuals and Ljung-Box for serial correlation, with `Diagnostics.residual_normality_pvalue`, `Diagnostics.residual_autocorr_pvalue`, and human-readable `Diagnostics.assumption_warnings` when either p-value drops below 0.05.
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
Verdict: leveling off (logistic, 1.00 confidence; 90% CI [1.00, 1.00]).
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
Per-capita slope: -0.002307; posterior weights: exponential 0.00 [0.00, 0.00], linear 0.00 [0.00, 0.00], logistic 1.00 [1.00, 1.00], power-law 0.00 [0.00, 0.00].
```

The bootstrap intervals look implausibly tight here because the demo inputs are perfectly clean synthetic curves; on real noisy data they widen to reflect the true sampling uncertainty in the fit. The verdict-line CI (e.g. `90% CI [1.00, 1.00]`) is a percentile interval on the winning posterior weight itself — a wide CI here means the headline confidence is fragile under resampling. The `accelerating` and `steady` cases show no CI because bootstrap is gated to run only when the logistic verdict is the focus or the result is indeterminate. For programmatic access rather than a printed summary, every value in the rendered output is also a typed attribute on `GrowthAnalysis` — `p_exponential`, `p_linear`, `p_logistic`, `preferred_model`, `is_indeterminate`, `indeterminate_reason`, `logistic_intervals.K`, `weight_intervals.p_logistic`, etc.

## Predicting future values

`GrowthAnalysis.predict(time)` returns a prediction at one or more future times in the *original* time coordinate (no manual time-origin shift required). With the default `ci=0.9` it returns a `Prediction(low, point, high)` namedtuple whose bounds are pair-bootstrap percentile bounds at the requested confidence level. Pass `ci=None` for just the point.

```python
result = analyze_growth(time, values, n_boot=200, bootstrap_seed=0)

result.predict(15.0)              # Prediction(low=119.78, point=119.78, high=119.78)
result.predict(15.0, ci=None)     # 119.78
result.predict([13.0, 15.0, 20.0])  # list of Prediction, one per horizon
```

For an indeterminate verdict `predict()` raises `ValueError` — the indeterminate branch exists precisely because no model is reliable enough to predict from. Inspect `result.exponential_fit`, `result.linear_fit`, or `result.logistic_fit` directly if you want a prediction from a specific candidate.

## Visualizing the result

Install the optional plotting extra:

```bash
pip install 'project-verge[plot]'
```

Then `plot_growth_analysis(result)` produces a single-figure summary — data, all three fitted curves (the preferred model bold, the others as faint dashed lines for comparison), the carrying-capacity asymptote `K` when logistic is preferred, a 90% bootstrap prediction envelope when uncertainty data is available, and a title that mirrors the verdict line.

```python
import matplotlib.pyplot as plt
from project_verge.plot import plot_growth_analysis

ax = plot_growth_analysis(result)
plt.tight_layout()
plt.show()
```

Pass an existing `ax` to compose with other axes; pass `extrapolate_fraction=0` to plot only the observed range; pass `envelope=False` to suppress the envelope (or `envelope=True` to force a fresh bootstrap when none is cached on the result). A runnable example lives at [examples/demo_plot.py](examples/demo_plot.py).

## Worked example: world population

[examples/world_population.py](examples/world_population.py) runs Verge on world-population estimates at milestone years from 1750 through 2022 ([data file](examples/data/un_population.csv), drawn from the UN World Population Prospects 2022 revision plus standard pre-1950 historical estimates). The script is deliberately set up to compare two analysis windows on the *same* dataset, because Verge's verdict is a function of what data you give it — not a universal claim about the future:

```
============================================================
Full history (1750-2022)
============================================================
Verdict: indeterminate (reason: ambiguous_evidence).
Posterior weights are too close to call between the candidate models.
Per-capita slope: +0.0008194; posterior weights: exponential 0.88 [0.88, 0.88], linear 0.00 [0.00, 0.00], logistic 0.12 [0.12, 0.12], power-law 0.00 [0.00, 0.00].

============================================================
Post-1950 only (demographic transition window)
============================================================
Verdict: indeterminate (reason: fragile_verdict).
The criterion favors a single model, but the bootstrap CI on its weight is wide enough that the verdict could swap under resampling -- treat the headline confidence as unreliable.
Estimated ceiling K ~= 12.98 [11.78, 14.2].
Estimated inflection time ~= 2004 [1997, 2012].
Per-capita slope: -0.002296; posterior weights: exponential 0.00 [0.00, 0.00], linear 0.00 [0.00, 0.85], logistic 1.00 [0.14, 1.00], power-law 0.00 [0.00, 0.00].
```

Both windows are flagged as `indeterminate` under defaults, for two distinct reasons that are themselves the lesson:

- Full history (n = 14): exponential leads with weight 0.88, which clears the `"positive"` band (0.75) but not the `"strong"` default (0.95). Verge declines to commit. The honest read is "exponential is the most likely candidate, but 0.88 is not a verdict, it is a lean." Pass `evidence_strength="positive"` and the verdict becomes `accelerating (exponential, 0.88 confidence)`; pass `"decisive"` and even AICc's preferred candidate would have to clear 0.99 (almost no real-data fit does).
- Post-1950 only (n = 9): the in-sample logistic fit is excellent (`K ≈ 13B`, inflection ≈ 2004), but with only 9 observations and a 4-parameter logistic the AICc small-sample penalty makes the bootstrap CI on the logistic weight wide ([0.14, 1.00]). T-28's `fragile_verdict` gate fires: "leveling off looks plausible but I cannot commit to it from 9 data points." Switching to `criterion="bic"` (gentler small-sample penalty) recovers the historical `leveling off (logistic, 1.00 confidence; 90% CI [0.67, 1.00])` verdict.

Both indeterminate reasons are working as designed. Add another decade of observations to the modern window and the dataset will likely flip to a decisive `leveling off`. Move the threshold from `"strong"` to `"positive"` and the historical-window verdict becomes `accelerating`. The point of the example is that those *are* knobs the user can turn — Verge's defaults are conservative on purpose so that the headline only commits when the evidence is genuinely strong.

Run it yourself with `python examples/world_population.py` to see the side-by-side plot.

## API

### `analyze_growth(time, values, *, prior_exponential=0.5, prior_linear=0.5, prior_logistic=0.5, prior_power_law=0.5, min_points=8, min_fit_quality=0.85, max_weight_ci_width=0.40, criterion="aicc", evidence_strength="strong", allow_smoothing=False, smoothing_window=3, horizons=None, n_boot=500, bootstrap_confidence=0.90, bootstrap_seed=None)`

Runs the full analysis and returns a `GrowthAnalysis` object.

`min_fit_quality` is the log-space R² floor that each candidate model is held to. When *both* models fall below it, the verdict is forced to `indeterminate` with `indeterminate_reason = "neither_model_fits"`, so a polynomial or other out-of-family series cannot quietly produce a confident-looking exponential-vs-logistic split.

`GrowthAnalysis.indeterminate_reason` is `None` when the verdict is decisive, and otherwise one of `"neither_model_fits"`, `"ambiguous_evidence"`, or `"logistic_unidentifiable"`.

`criterion` selects the information criterion used for the four-way model comparison: `"aicc"` (default) or `"bic"`. AICc applies the standard small-sample correction `+ 2k(k+1)/(n−k−1)` on top of AIC; for the typical input sizes Verge sees (n = 8–30) the correction is meaningful and matches the small-sample-regression literature's recommendation. The two criteria pick the same model on clean data with a clear winner; on borderline cases AICc tends to penalize the higher-parameter logistic / power-law candidates more strongly than BIC at small n.

`evidence_strength` controls how decisive the leading model has to be before Verge commits to a non-`indeterminate` verdict. The named bands are calibrated against Kass & Raftery's (1995) interpretive scale for log Bayes factors:

| `evidence_strength` | Winning weight | Approx. ΔIC gap | Kass & Raftery band |
| --- | --- | --- | --- |
| `"positive"` | ≥ 0.75 | ≥ ~2 | "positive" |
| `"strong"` (default) | ≥ 0.95 | ≥ ~6 | "strong" |
| `"decisive"` | ≥ 0.99 | ≥ ~10 | "decisive" |

Below the threshold the verdict is forced to `indeterminate (reason: ambiguous_evidence)`. The default `"strong"` is intentionally conservative: a leading weight of 0.85 is *suggestive* of accelerating / leveling off / steady, but it is not a verdict you should bet on, and Verge says so. Pass `evidence_strength="positive"` if you want the looser threshold; pass `"decisive"` if you want only very-high-confidence verdicts.

`allow_smoothing` opens Verge to noisy real-world data. With the default `False`, the v1 input contract requires strict nondecreasing values and any downward blip raises `ValueError`. Setting `allow_smoothing=True` runs a rolling-median smoother (window `smoothing_window`, default 3) followed by a cumulative-max pass that enforces the nondecreasing post-condition the rest of the library assumes. The smoothed series — not the raw input — is what gets stored on `result.input_values`, what `predict()` and `plot()` see, and what the bootstrap resamples; this keeps the entire analysis in one consistent coordinate. The transformation is recorded in `result.transform_log` so the action is auditable. Trade-off: a *genuine* downward move in the underlying process is mapped to a flat segment by `cumulative-max`, biasing the fit upward — for series where you expect occasional real dips, do your own pre-processing instead.

`horizons`, `n_boot`, `bootstrap_confidence`, and `bootstrap_seed` control a pair-bootstrap that fills `GrowthAnalysis.logistic_intervals` with percentile intervals for the logistic `K`, `r`, and `t0`, plus one prediction interval per supplied horizon (in the original time coordinate). The bootstrap runs only when it is actually informative — when the logistic is the preferred model or the verdict is indeterminate — because the optimizer is unidentified on data that is clearly exponential and a bootstrap there would just be expensive decoration. Pass `n_boot=0` to skip the bootstrap entirely.

`GrowthAnalysis.diagnostics.fit_warnings` contains optimizer or fit-process warnings; `GrowthAnalysis.diagnostics.identifiability_warnings` contains interpretation warnings specific to whether the logistic bend is actually identified by the observed window; `GrowthAnalysis.diagnostics.assumption_warnings` contains warnings from the log-normal residual checks (Shapiro-Wilk normality and Ljung-Box autocorrelation tests on the leading-model log-residuals). The assumption checks are automatically skipped when residuals are at the floating-point floor (clean synthetic inputs), so they do not over-fire on optimizer noise.

### `fit_exponential(time, values, *, min_points=8)`

Fits the exponential model and returns a `ModelFit`.

### `fit_linear(time, values, *, min_points=8)`

Fits the linear model `y = a + b*t` (in y-units, with residuals computed in log-space for consistency with the other models) and returns a `ModelFit`.

### `fit_logistic(time, values, *, min_points=8)`

Fits the logistic model and returns a `ModelFit`.

### `fit_power_law(time, values, *, min_points=8)`

Fits the power-law model `y = a * (t + 1)**k` via OLS in log-log space and returns a `ModelFit`. Power-law is a diagnostic-only candidate in v1; the headline verdict never becomes "power-law" (when it wins, the result is `indeterminate (reason: power_law_shape)`).

## Input Contract

Version 1 assumes:

- `time` and `values` are one-dimensional sequences of equal length
- `time` is strictly increasing
- `values` is strictly positive
- `values` is nondecreasing
- at least 8 observations are provided for `analyze_growth`

## Failure modes

Verge's input contract is intentionally narrow and its candidate model space is small. When inputs sit outside what v1 supports, the failure shows up in one of three ways: a `ValueError` from input validation, an `indeterminate` verdict with a structured `indeterminate_reason`, or — in a few cases worth being honest about — a confident-looking verdict on data the library cannot actually distinguish. This section catalogs the patterns most worth watching for.

### Polynomial or power-law growth

**Signature.** Series shaped like `y ∝ t^k` for some `k > 0` — cubic, square-root, anything that is not pure exponential, linear, or sigmoid.

**Library response.** Verge fits a power-law candidate (`y = a · (t + 1)^k`) alongside the three primary models. When BIC picks power-law as the leading shape, the verdict is forced to `indeterminate (reason: power_law_shape)`, since v1's verdict surface (still growing / steady / leveling off) has no clean answer for power-law growth. A cubic series like `y = t**3` returns `indeterminate (reason: power_law_shape)` at the default `min_fit_quality`.

**Mitigation.** None needed: at the default threshold the library now flags power-law growth honestly rather than misclassifying it as logistic. If you want the underlying power-law fit, it lives at `result.power_law_fit` and the weight at `result.p_power_law`.

### Step change or regime shift

**Signature.** A series with one or more abrupt changes — for example, flat for half the window followed by exponential growth.

**Library response.** Triggers `indeterminate (reason: neither_model_fits)` even at the default threshold, because none of the three candidates can capture the discontinuity: log-space R² lands around 0.55–0.77 for all of them.

**Mitigation.** None within v1. Subset the data to a single regime and re-analyze, or wait for change-point support.

### Very short series (`n < 8`)

**Signature.** Fewer than 8 observations.

**Library response.** Hard `ValueError("at least 8 observations are required")` from input validation.

**Mitigation.** Wait for more data, or pass `min_points=N` with a smaller `N` if you accept that statistical signals weaken below 8 points — the diagnostics t-tests have very low df and the bootstrap CIs become uninformative.

### Heavy noise that breaks monotonicity

**Signature.** Real-world noisy data where some adjacent observations have `y_{i+1} < y_i`.

**Library response.** By default, hard `ValueError("values must be nondecreasing for the v1 API")` from input validation. Pass `allow_smoothing=True` and Verge runs a rolling-median plus cumulative-max smoother to coerce the input to monotone before fitting.

**Mitigation.** Use `analyze_growth(time, values, allow_smoothing=True)`. The transformation is recorded in `result.transform_log` so the action is auditable. The smoother is parameter-free aside from `smoothing_window` (default 3, must be a positive odd integer); for very noisy data try `smoothing_window=5`. The trade-off is that genuine real-world *decreases* in the underlying process are flattened by `cumulative-max` — if your data has real dips you want preserved, do your own pre-processing.

### Non-positive values

**Signature.** Any `y_i <= 0`.

**Library response.** Hard `ValueError("values must be strictly positive")`. The shared log-normal observation model needs strictly positive `y`.

**Mitigation.** Shift or clip the data so all values are positive, or transform the underlying problem so the quantity of interest is naturally positive.

### Random-walk-like or unstructured series

**Signature.** Data with no underlying growth model at all but happens to be nondecreasing — for example, `y = 1 + cumsum(|N(0, 1)|)` with `n = 20`.

**Library response.** Two automatic gates catch this in v1:

- **Power-law shape detection.** Cumulative-noise series have shape statistics that are well-approximated by a power-law fit, so most random-walk-like seeds now classify as `indeterminate (reason: power_law_shape)` — no false confidence on the headline.
- **Fragile-verdict gate.** When a verdict survives the other indeterminate checks but its bootstrap weight CI is wider than `max_weight_ci_width` (default 0.40), the verdict is downgraded to `indeterminate (reason: fragile_verdict)`. This catches cases where BIC picks a model decisively but resampling shows the choice is unstable.

**Mitigation.** None needed for typical inputs at default thresholds; the two gates handle this automatically. Tune `max_weight_ci_width` lower (toward 0.0) for stricter rejection of fragile verdicts, or higher (toward 1.0) to disable the gate. Cross-checking with domain knowledge is still wise — Verge can only see the data it is given.

### Already-saturated series

**Signature.** All observations are at or near a plateau, with only minor early growth visible.

**Library response.** Usually classifies as `leveling off` with high confidence — this is the **correct** answer, not a failure. `result.logistic_intervals.K` will sit just above the observed maximum.

**Mitigation.** None needed; this is a working case. If the early growth phase is too brief to identify the bend, you may instead see `indeterminate (reason: logistic_unidentifiable)`, which means the data really is too uninformative to pin down where the plateau is.

---

## Interpreting the Verdict

The verdict is one of four categorical labels — `accelerating`, `steady`, `leveling off`, or `indeterminate` — chosen from the leading model when a model is clearly preferred *and* the underlying fit passes a quality floor *and* (for logistic) the curve is identified by the observed window. If any of those checks fails, the verdict is forced to `indeterminate` with a structured `indeterminate_reason`.

The reported confidence is the posterior model weight of the winning model, approximated from BIC under the shared log-normal observation model. Those weights are:

- conditioned on exponential, linear, logistic, and power-law being the four candidates (power-law is diagnostic-only — when it wins, the verdict is `indeterminate (reason: power_law_shape)` rather than a new fourth verdict)
- conditioned on the log-normal observation model
- approximate, because BIC is used as a tractable proxy for full Bayesian model evidence

They should be read as model-comparison evidence, not as a universal forecast probability that a real-world process must (or must not) plateau. The v1 model space is narrow on purpose; the [TICKETS](TICKETS.md) backlog tracks planned extensions to richer S-curve families.

Bootstrap intervals on `K`, `t0`, and prediction horizons are pair-bootstrap percentile intervals at the configured confidence level (default 90%). They quantify how much the logistic fit moves under resampling. A wide interval when the logistic is the preferred verdict is a load-bearing honesty signal — it means the data does not yet pin down where the plateau is — not a bug.

## Repository Layout

- `src/project_verge/`: library source
- `tests/`: unit tests
- `examples/demo_growth_analysis.py`: runnable demonstration on synthetic series
- `examples/demo_plot.py`: matplotlib visualization demo
- `examples/world_population.py` (+ `examples/data/un_population.csv`): real-data worked example on world population
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
