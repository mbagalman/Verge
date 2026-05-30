# Glossary

Plain-English definitions for the statistics and methodology terms GrowthShape surfaces. Aimed at smart users who aren't statisticians — the goal is "enough to understand what the library is saying," not a textbook treatment.

## Verdict and gating

### Verdict
The headline answer GrowthShape returns: `accelerating`, `steady`, `leveling off`, or `indeterminate`. Maps directly to the user-facing question "is this still going up, or is it leveling off?" *Accelerating* corresponds to a winning exponential fit; *steady* to linear; *leveling off* to logistic; *indeterminate* means GrowthShape declined to commit (see *Indeterminate*).

### Indeterminate
GrowthShape's "I can't tell" verdict. Triggered when one of several conditions makes a confident answer dishonest — see [README's "Interpreting the Verdict"](../README.md#interpreting-the-verdict) for the precedence chain. The structured `result.indeterminate_reason` says exactly which condition fired.

### Power-law shape
The structured indeterminate reason that fires when a `y = a · t^k` fit wins the BIC competition. Power-law growth has no clean answer to "going up vs leveling off" (depending on `k`, it can be either), so GrowthShape declines rather than misclassifying. Catches polynomial inputs that would otherwise misclassify as logistic.

### Fragile verdict
The structured indeterminate reason that fires when BIC favors a single model decisively but the bootstrap CI on its weight is wider than `max_weight_ci_width` (default 0.40). Means the headline confidence could swap under resampling — treat the leading model as a lean, not a verdict.

### Signal disagreement
The structured indeterminate reason that fires when BIC prefers logistic but two or more of the three supporting diagnostics (per-capita slope, log-residual curvature, forecast MAE) do not agree. Catches cases where the criterion's preference rests on shaky evidence.

## Information criteria and posterior weights

### BIC (Bayesian Information Criterion)
A model-comparison score: `BIC = -2·log_likelihood + k·ln(n)`, where `k` is the parameter count and `n` is the observation count. Lower is better. The `k·ln(n)` term penalizes models with more parameters more aggressively as `n` grows. Used as a tractable proxy for full Bayesian model evidence.

### AICc (Corrected Akaike Information Criterion)
A small-sample-corrected variant of AIC: `AICc = -2·log_likelihood + 2k + 2k(k+1)/(n−k−1)`. The correction matters when `n` is small relative to `k`, which is exactly GrowthShape's typical input size (`n` between 8 and 30 with `k` up to 4). GrowthShape defaults to AICc for this reason; pass `criterion="bic"` for the gentler penalty.

### Posterior weight
The normalized exponentiated negative-half-criterion score. With four candidate models and equal priors, `p_model = exp(−criterion_model / 2) / Σ exp(−criterion_j / 2)`, mapped onto `[0, 1]` so weights sum to one. The "0.95 confidence" in a verdict line is the winning model's posterior weight. Reading: a higher weight means BIC/AICc puts more relative evidence on this model than on the other three; it is *not* a frequentist probability that the underlying truth is this model.

### Evidence strength bands
Named thresholds on the winning posterior weight, calibrated against Kass & Raftery (1995):
- **positive** ≥ 0.75 (≈ ΔIC ≥ 2)
- **strong** ≥ 0.95 (≈ ΔIC ≥ 6) — GrowthShape's default
- **decisive** ≥ 0.99 (≈ ΔIC ≥ 10)

Below the configured threshold, the verdict is forced to `indeterminate (reason: ambiguous_evidence)`.

## Models and parameters

### Carrying capacity (K)
The asymptote of a logistic curve `y = K / (1 + exp(-r·(t − t₀)))`. Conceptually the long-run ceiling the underlying process is approaching. When GrowthShape says "leveling off," `K` is its estimate of where the leveling-off plateau sits.

### Inflection time (t₀)
The time at which a logistic curve's growth rate is highest — the curve's midpoint, where `y = K/2`. Reported in the original time coordinate of the input data.

### Identifiability
Whether the data actually constrains a model's parameters. The logistic's `K` is "unidentifiable" when the observed window only covers an early portion of the bend — many different `K` values fit the data nearly equally well, so the bootstrap interval on `K` would span orders of magnitude. GrowthShape's `logistic_unidentifiable` indeterminate reason catches this: a logistic fit can converge numerically while still being meaningless.

### Power-law candidate
The fourth candidate model (added in T-27): `y = a · (t + 1)^k`, fit in log-log space. Diagnostic-only — never becomes the preferred verdict; if it wins the BIC competition the gate forces `indeterminate (reason: power_law_shape)`.

## Observation model and assumption checks

### Log-normal observation model
The assumption that residuals are normally distributed *on the log scale* — equivalently, observed `y` is the curve value times an independent log-normal multiplicative noise term. Captures the empirical pattern that growth-process variance scales with the level. GrowthShape's BIC/AICc and bootstrap intervals assume this; if the assumption is violated, both are biased.

### Shapiro-Wilk test
A null-hypothesis test for whether a sample is drawn from a normal distribution. `p < 0.05` rejects normality. GrowthShape runs this on the leading-model log-residuals; failure surfaces as an `assumption_warning`.

### Ljung-Box test
A test for autocorrelation in a residual series at lags up to `h`. `p < 0.05` rejects "no autocorrelation" — meaning the residuals show a systematic pattern the model didn't capture (typical of misfit). GrowthShape runs this on the leading-model log-residuals; failure surfaces as an `assumption_warning`.

## Diagnostics

### Per-capita growth slope
The OLS slope of per-period growth-rate on level: `(Δy / Δt) / y` regressed against `y`. Negative for logistic data (growth rate decreases as `y` approaches `K`); near zero for exponential; mildly negative for linear (because `b/y` decreases with `y`).

### Log-residual curvature
The `t²` coefficient from a quadratic fit of `log(y)` against scaled time. Negative when log-`y` is concave-down (a saturation signature); near zero for pure exponential.

### Forward-chaining forecast
A rolling-window cross-validation: fit the model on the first `k` observations, predict observation `k+1`, record the absolute log-error; advance `k`; repeat. GrowthShape reports the *median* of converged-window errors plus the *convergence rate* on `Diagnostics.forecast_*`. The median ignores failed-to-converge windows (which previously poisoned the mean to `inf`).

### Convergence rate
For a forward-chaining forecast: the fraction of rolling windows where the model fit converged, in `[0, 1]`. Below 100% means the median is computed over a subset; the absolute count is `n_windows`.

## Uncertainty quantification

### Pair bootstrap
A resampling procedure: draw `n` `(time, value)` pairs with replacement from the original `n`-point input, refit, and record the parameters. Repeat `n_boot` times to get an empirical distribution. The percentile method takes the 5th and 95th percentiles as a 90% confidence interval.

### Weight intervals
Pair-bootstrap percentile intervals on the BIC/AICc-derived posterior weights themselves. Tells you whether the headline "0.93 confidence" would survive resampling. A wide weight CI (default fragility threshold 0.40) trips the `fragile_verdict` gate.

### Prediction interval
A `(low, point, high)` triple for the predicted value at a given future time. The bounds are pair-bootstrap percentiles at the requested CI level (default 90%); the `point` is the analytical prediction from the preferred-model fit, not the bootstrap median.

## Optimization

### Multi-start optimization
Running the nonlinear optimizer from multiple diverse initial guesses and keeping the lowest-RSS solution. Used in the logistic fit only; the other candidates have well-behaved likelihood surfaces. Default `n_starts=8`. Matters for pathological cases where the inflection sits well outside the observed window — single-start can land in a worse local minimum.

### Cumulative-max smoothing
The pre-fit transformation GrowthShape applies when `allow_smoothing=True`: a rolling median (window 3 by default) followed by `np.maximum.accumulate` to enforce the nondecreasing post-condition. The trade-off: genuine downward moves in the underlying process become flat segments. Recorded in `result.transform_log` for auditability.

## Result fields and shapes

### `GrowthAnalysis`
The result returned by `analyze_growth`. Has the verdict (`preferred_model`, `is_indeterminate`, `indeterminate_reason`), the four `ModelFit` objects, the `Diagnostics`, the bootstrap intervals (when applicable), the `AnalysisAssumptions`, and the input arrays plus time-origin (so `predict()` is self-contained).

### `ModelFit`
The result of fitting a single model: parameters, fitted values, log-likelihood, BIC, AICc, log-space R², convergence flag, optimizer warnings.

### `Diagnostics`
The supporting evidence collected once per `analyze_growth` call: per-capita slope (with t-stat and p-value), log-residual curvature (with t-stat and p-value), forward-chaining forecast diagnostics for each model, the `signal_agreement` flag set, the assumption-test p-values, and the warning tuples.

### `AnalysisAssumptions`
A structured record of the methodological choices GrowthShape actually used for the call: the information criterion, the evidence-strength band, the candidate model identifiers, and the observation model. Lets a downstream consumer reproduce or audit without re-deriving from passed arguments.

### `transform_log`
A tuple of strings on `GrowthAnalysis` describing pre-fit transformations applied to the input. Currently only used by `allow_smoothing` (T-15); designed to be extensible for future transforms.
