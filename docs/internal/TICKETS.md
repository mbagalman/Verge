# Project Verge — Tickets

Working backlog of improvements toward the goal: *a focused, statistically rigorous, intellectually honest tool that answers "is this going to keep going up, or level off?"*

Tickets are grouped by priority. Within a tier, ordering is rough but generally reflects dependency or value-per-effort.

**Priority legend**
- **P0** — required before v0.1.0; addresses silent-failure or honesty gaps
- **P1** — high-value features that materially advance the stated goal
- **P2** — robustness and quality; ship before v0.2.0
- **P3** — polish and process

**Effort legend** — S (<1 day), M (1–3 days), L (>3 days)

**Category** — M = Methodology, C = Code, D = Docs/UX

---

## Summary

| ID | Title | Priority | Category | Effort | Depends on | Status |
|----|-------|----------|----------|--------|------------|--------|
| T-01 | Add a "neither model fits" exit | P0 | M+C | M | — | **done** (`a99d601`) |
| T-02 | Bootstrap CI on K, t0, and predicted value | P0 | M+C | M | — | **done** (`eb37fe1`) |
| T-03 | `GrowthAnalysis.summary()` with human-readable verdict | P0 | C | S | T-02 (soft) | **done** (`397a3d5`) |
| T-04 | Reframe README around the real question + show indeterminate case | P0 | D | S | T-03 | **done** (`c769031`) |
| T-05 | Add a third "linear-in-log" / sub-exponential baseline | P1 | M | M | T-01 | **done** (`0907752`) |
| T-06 | Wire diagnostics (slope sig., curvature sig., forecast MAE) into verdict | P1 | M+C | M | — | **done** (`f0d76df`) |
| T-07 | Bootstrap CI on the model weights themselves | P1 | M+C | S | T-02 | **done** (`6271730`) |
| T-08 | `GrowthAnalysis.predict(time, *, ci=0.9)` | P1 | C | S | T-02 | **done** (`ea50a77`) |
| T-09 | `plot_growth_analysis()` helper | P1 | C+D | S | T-02 (soft) | **done** (`3b4d7f8`) |
| T-10 | Failure-modes section in README | P1 | D | S | T-01 | **done** (`c9c528f`) |
| T-11 | Real-data example (UN population) | P1 | D | S | — | **done** (`d81238d`) |
| T-27 | Power-law shape detection (catches polynomial misclassification) | P1 | M | M | T-01 | **done** (`9e63106`) |
| T-28 | Auto-downgrade verdict on wide weight CI (catches random-walk fragility) | P1 | M+C | S | T-07 | **done** (`ecab902`) |
| T-12 | AICc instead of (or alongside) BIC | P2 | M | S | — | **done** (`d3194f3`) |
| T-13 | Log-normal assumption checks (Shapiro-Wilk, Ljung-Box) | P2 | M | S | — | **done** (`274eb97`) |
| T-14 | Tie indeterminate threshold to documented evidence bands | P2 | M | S | — | **done** (`3cdde75`) |
| T-15 | Smoothing / noise-tolerance path for non-monotone data | P2 | M+C | M | — | **done** (`99d4ff7`) |
| T-16 | Multi-start optimization for the logistic fit | P2 | C | S | — | **done** (`fcc8342`) |
| T-17 | Strengthen tests (noise, wrong-model, calibration) | P2 | C | M | T-11 (soft) | **done** (`eff41d2`) |
| T-18 | Better `forecast_mae` aggregator (median + frac_converged) | P2 | C | S | — | **done** (`0509163`) |
| T-19 | Calibration evidence in docs | P2 | D | M | T-17 | **done** (`904907e`) |
| T-20 | `Literal`/`Enum` types for `model_name` and `preferred_model` | P3 | C | S | — | **done** (`993bc56`) |
| T-21 | `npt.ArrayLike` type hints | P3 | C | S | — | **done** (`e860e42`) |
| T-22 | Resolve `min_points` duplicate validation | P3 | C | S | — | **done** (`37ad665`) |
| T-23 | Make `assumptions` field structured (or remove) | P3 | C | S | T-15 | **done** (`ab5695a`) |
| T-24 | Glossary in docs | P3 | D | S | — | **done** (`ffd7716`) |
| T-25 | `CHANGELOG.md` | P3 | D | S | — | **done** (`7f4bc98`) |
| T-26 | Defer release workflow until P0/P1 land | P3 | Process | S | — | open |

---

## P0 — Required before v0.1.0

### T-01: Add a "neither model fits" exit
**Category:** Methodology + Code · **Effort:** M · **Status:** Done in commit `a99d601`

**Problem.** [_api.py:88-101](../../src/project_verge/_api.py#L88-L101) normalizes BIC weights to sum to 1.0 even when both fits are terrible. A polynomial, linear, or power-law input still produces a confident-looking exponential-vs-logistic verdict.

**Proposal.** Compute an absolute fit-quality metric (e.g. log-space R², or residual standard error vs a constant predictor). If both models fall below threshold, force `is_indeterminate = True` with a structured reason `"neither_model_fits"`.

**Acceptance criteria.**
- Helper computes log-space R² (or chosen metric) for both fits
- `analyze_growth` gates the verdict on at least one fit clearing the threshold (default configurable, e.g. R² > 0.85)
- `preferred_model = "indeterminate"` when neither passes
- New field on `GrowthAnalysis` carrying the indeterminate reason: `"ambiguous_evidence" | "neither_model_fits" | "logistic_unidentifiable"`
- Test: polynomial growth series → indeterminate with `"neither_model_fits"`
- Test: clean exponential → exponential, not indeterminate

**Files.** [_api.py](../../src/project_verge/_api.py), [_diagnostics.py](../../src/project_verge/_diagnostics.py), [_types.py](../../src/project_verge/_types.py), [tests/test_analysis.py](../../tests/test_analysis.py)

---

### T-02: Bootstrap CI on K, t0, and predicted value
**Category:** Methodology + Code · **Effort:** M · **Status:** Done in commit `eb37fe1`

**Problem.** When logistic wins, [_diagnostics.py:97-123](../../src/project_verge/_diagnostics.py#L97-L123) only sanity-checks K's plausibility. The user's real question — *how soon* and *where* will it level off? — needs uncertainty intervals.

**Proposal.** Add a pair-bootstrap helper: resample (time, values) with replacement, refit logistic, report 5/50/95 percentiles for K, t0, and `predict(t)` at user-supplied horizons.

**Acceptance criteria.**
- New module `src/project_verge/_uncertainty.py` exposing `bootstrap_logistic_intervals(time, values, n_boot=500, horizons=None)`
- `analyze_growth` accepts optional `horizons` parameter; when supplied, returns `predicted_intervals` on the result
- CI on K and t0 always available when logistic fit converged
- `n_boot` configurable; default 500; capped to keep wall time <2s on n ≤ 200
- Test: known logistic series → 90% CI brackets true K in ≥85% of seeded trials

**Implementation note.** The "always available when logistic converged" criterion was relaxed: the bootstrap is *gated* to run only when the logistic verdict is actually informative — i.e. when `preferred_model == "logistic"` or the result is indeterminate. On clean exponential data the logistic optimizer is unidentified (K runs against its bound), and a 500-iteration bootstrap on those resamples added ~125 s to a single test. The gate keeps the default usable; pass `n_boot=0` to skip explicitly, or call `bootstrap_logistic_intervals` directly to bootstrap without the gate.

**Files.** new `src/project_verge/_uncertainty.py`, [_api.py](../../src/project_verge/_api.py), [_types.py](../../src/project_verge/_types.py), new `tests/test_uncertainty.py`

---

### T-03: `GrowthAnalysis.summary()` with human-readable verdict
**Category:** Code · **Effort:** S · **Depends on:** T-02 (soft — degrades gracefully) · **Status:** Done in commit `397a3d5`

**Problem.** `print(result)` dumps the raw dataclass. The most-used surface should be a verdict sentence, not field names.

**Proposal.** Add `summary()` returning a short multi-line string. Include verdict, confidence, K interval (if logistic), and the indeterminate reason where applicable.

**Acceptance criteria.**
- `GrowthAnalysis.summary()` returns formatted string. Example:
  > Verdict: **leveling off** (logistic, 0.87 confidence).
  > Estimated ceiling K ≈ 102 [88, 121]. Inflection ≈ year 6.4 [5.9, 7.1].
  > Per-capita slope: −0.012 (significant). Forecast MAE: log 0.04.
- `__repr__` defers to `summary()` (or a one-line variant)
- Snapshot test of `summary()` on the three demo cases in [examples/demo_growth_analysis.py](../../examples/demo_growth_analysis.py)

**Files.** [_types.py](../../src/project_verge/_types.py), possibly new `_summary.py`, [tests/test_analysis.py](../../tests/test_analysis.py)

---

### T-04: Reframe README around the real question + show the indeterminate case
**Category:** Docs · **Effort:** S · **Depends on:** T-03 · **Status:** Done in commit `c769031`

**Problem.** [README.md](../../README.md) leads with "compares exponential vs logistic." User's question is "will this level off?" Quick Start also doesn't show the indeterminate case, which is the most honest output.

**Proposal.** Rewrite the opening paragraph and Quick Start. Lead with the user-facing question. Restructure Quick Start to print `result.summary()` and cover three cases: clearly growing, clearly leveling off, indeterminate.

**Acceptance criteria.**
- README opens with the user's question, not the methodology
- Quick Start uses `summary()` output
- Quick Start covers all three verdict types
- "What v1 does / does not do" preserved but reframed in user-question terms

**Files.** [README.md](../../README.md)

---

## P1 — High-value features

### T-05: Add a third "linear-in-log" / sub-exponential baseline
**Category:** Methodology · **Effort:** M · **Depends on:** T-01 · **Status:** Done in commit `0907752`

**Problem.** Binary frame forces every series into exp or logistic. A power-law or linear-in-y series is silently misclassified. Three-way frame (*accelerating / steady / leveling-off*) maps onto the user's question much more directly.

**Proposal.** Fit a third "no-curvature on log scale" baseline (linear-in-log, i.e. power-law in y vs t — or pick linear-in-y; rationale documented either way). Include it in the BIC/AICc comparison.

**Acceptance criteria.**
- `fit_baseline_model` exists with documented model form
- `analyze_growth` returns three weights, e.g. `p_accelerating`, `p_steady`, `p_levelling_off`
- `preferred_model` accepts the new third value
- Verdict surface in `summary()` updated to use the user-facing labels
- Tests for known power-law and linear series

**Implementation note.** Picked **linear-in-y** (`y = a + b*t`) over the power-law option. Rationale: linear maps cleanly to one verdict (`steady`), where power-law spans both accelerating (`k > 1`) and decelerating (`0 < k < 1`) regimes and would muddy the verdict mapping; linear matches exponential on parameter count (2) so the BIC penalty is symmetric; and linear avoids the awkward `t = 0` handling power-law would need. Polynomial / sub-exponential growth that linear cannot capture still falls through to T-01's `neither_model_fits` exit, which is the honest answer for those out-of-family cases. The verdict surface is now four-way: `accelerating` (exponential preferred), `steady` (linear preferred), `leveling off` (logistic preferred), `indeterminate`. The previous label `still growing` was renamed to `accelerating` so the three growing verdicts are parallel.

**Files.** [_fit.py](../../src/project_verge/_fit.py), [_api.py](../../src/project_verge/_api.py), [_types.py](../../src/project_verge/_types.py), tests

---

### T-06: Wire diagnostics into the verdict
**Category:** Methodology + Code · **Effort:** M · **Status:** Done in commit `f0d76df`

**Problem.** `per_capita_slope`, `residual_curvature_score`, and forecast MAE are computed in [_diagnostics.py](../../src/project_verge/_diagnostics.py) but never feed the verdict or the indeterminate decision. The user's question is best answered by signal *agreement*, not BIC alone.

**Proposal.** Add t-tests for the per-capita slope (H1: slope < 0) and the curvature coefficient (H1: t² coef < 0). Combine BIC + slope-significance + curvature-significance + forecast-MAE-direction into a vote. Require multi-signal agreement before declaring a non-indeterminate verdict.

**Acceptance criteria.**
- Per-capita regression returns slope, std err, t-stat, p-value
- Residual curvature returns coefficient, std err, t-stat, p-value
- New field on `Diagnostics`: `signal_agreement` (count or vector of signals favoring leveling-off)
- Indeterminate gate considers signal agreement, not just `|p_log − p_exp|`
- Test: case where BIC favors logistic but slope and forecast MAE disagree → indeterminate

**Files.** [_diagnostics.py](../../src/project_verge/_diagnostics.py), [_api.py](../../src/project_verge/_api.py), [_types.py](../../src/project_verge/_types.py), tests

**Implementation note.** The signal-agreement gate is **asymmetric** by design — it only second-guesses the logistic verdict, not exponential or linear. Per-capita slope and log-residual curvature are *also* significantly negative for clean linear data (`b/y` decreases with `y`; `log(a + b*t)` is concave), so a symmetric "signals must agree with leading_model" gate would over-fire on every clean linear case. BIC's three-way comparison from T-05 already weighs exponential vs linear vs logistic against each other; the supporting signals only need to second-guess the logistic branch. An end-to-end "BIC says logistic but signals disagree" test is not added because clean monotone synthetic logistic data always fires all three signals at p < 1e-6 — the gate is designed for noisier real-world inputs that the v1 input contract does not yet allow. Once T-15 (smoothing / noise tolerance) lands the gate will see real use; for now the helper unit test plus the linear-passthrough test verify correctness on the inputs supported.

---

### T-07: Bootstrap CI on the model weights themselves
**Category:** Methodology + Code · **Effort:** S · **Depends on:** T-02 · **Status:** Done in commit `6271730`

**Problem.** "p_logistic = 0.93" sounds decisive, but the 90% CI from a residual or pair bootstrap might be (0.55, 0.99).

**Proposal.** Reuse the bootstrap infrastructure from T-02. For each resample, refit both models, recompute weights. Report 5/50/95 percentiles.

**Acceptance criteria.**
- `bootstrap_weights` helper added
- `p_exponential` / `p_logistic` exposed as point + interval (or sibling fields `p_exponential_low`, `p_exponential_high`)
- `summary()` shows the interval
- Test: noisy data → wide CI; clean data → narrow CI

**Files.** `_uncertainty.py`, [_api.py](../../src/project_verge/_api.py), [_types.py](../../src/project_verge/_types.py)

---

### T-08: `GrowthAnalysis.predict(time, *, ci=0.9)`
**Category:** Code · **Effort:** S · **Depends on:** T-02 · **Status:** Done in commit `ea50a77`

**Problem.** User has no ergonomic way to ask "predicted value at horizon X." Today they must grab `fit.parameters` and re-run the curve function manually.

**Proposal.** `predict(time)` returns the point estimate from the preferred-model fit. With `ci`, returns `(low, point, high)` using the bootstrap from T-02.

**Acceptance criteria.**
- `predict(time)` works for all `preferred_model` values
- For `indeterminate`, document the chosen behavior (raise informative error, or return ensemble mean with explicit caveat — pick one)
- `ci=0.9` by default
- Vectorized over `time`

**Files.** [_types.py](../../src/project_verge/_types.py), [_api.py](../../src/project_verge/_api.py)

---

### T-09: `plot_growth_analysis()` helper
**Category:** Code + Docs · **Effort:** S · **Depends on:** T-02 (soft — for envelope) · **Status:** Done in commit `3b4d7f8`

**Problem.** First thing any user wants is to see fits overlaid on data.

**Proposal.** New `project_verge.plot` module exposing `plot_growth_analysis(result, ax=None)`. matplotlib added as an optional dependency.

**Acceptance criteria.**
- New `src/project_verge/plot.py`
- matplotlib added to `[project.optional-dependencies]` as `plot = ["matplotlib>=3.7"]`
- Single-figure summary: data points, both fits, forecast envelope (when available), K asymptote when logistic preferred
- Returns `Axes`
- Example script uses it
- README snippet

**Files.** new `src/project_verge/plot.py`, [examples/demo_growth_analysis.py](../../examples/demo_growth_analysis.py), [README.md](../../README.md), [pyproject.toml](../../pyproject.toml)

---

### T-10: Failure-modes section in README
**Category:** Docs · **Effort:** S · **Depends on:** T-01 · **Status:** Done in commit `c9c528f`

**Problem.** User can't tell what'll happen if their data violates assumptions.

**Proposal.** Add a "Failure modes" section. For each problematic input pattern, state what will happen and how the library will signal it.

**Patterns to cover.**
- Polynomial / power-law growth
- Step change or regime shift
- Very short series (n < ~12)
- Heavy noise
- Monotone violations
- Unit-root / random-walk series
- Already-saturated series (growth complete)

**Acceptance criteria.**
- Section exists in README
- Each pattern lists: example signature, library response (warning text or indeterminate reason), suggested mitigation

**Files.** [README.md](../../README.md)

---

### T-11: Real-data example
**Category:** Docs · **Effort:** S · **Status:** Done in commit `d81238d`

**Problem.** Examples use synthetic curves only. Trust comes from real-data demos.

**Proposal.** Commit a small CSV (UN World Population Prospects subset, or another well-known public source) and an `examples/` script that reproduces the population analysis from [docs/internal/methodology_notes.md](methodology_notes.md).

**Acceptance criteria.**
- `examples/data/un_population.csv` with provenance comment in the script
- `examples/world_population.py` runs end-to-end and prints `summary()`
- README references it as a hero example

**Files.** new `examples/data/un_population.csv`, new `examples/world_population.py`, [README.md](../../README.md)

---

### T-27: Power-law shape detection (catches polynomial misclassification)
**Category:** Methodology · **Effort:** M · **Depends on:** T-01 · **Status:** Done in commit `9e63106`

**Problem.** Surfaced by writing the [README's "Failure modes / Polynomial or power-law growth"](../../README.md) section in T-10. At the default `min_fit_quality=0.85`, polynomial growth (`y = t**3`, etc.) silently classifies as `leveling off`: the logistic fit clears the floor at log-space R² ≈ 0.96 because `log(t**k)` is concave-down on linear `t` — the same shape signature that logistic late-stage data has. Power-law shapes have nowhere honest to land in v1's three-model space, so the library misclassifies confidently. The current mitigation (lift `min_fit_quality=0.99`) works but pushes calibration onto every user.

**Proposal.** Add a power-law fit (linear regression of `log(y)` against `log(t - time_origin + ε)`, recovering `y = a * (t - origin)**k`) as a fourth diagnostic-only candidate. Compete it on BIC alongside exponential / linear / logistic. If power-law wins decisively, force `indeterminate` with new reason `"power_law_shape"`; do **not** extend the four-way verdict surface, because the user's question ("is this leveling off, or going up?") doesn't have a clean answer for power-law growth and forcing one into "accelerating" or "steady" would mis-translate.

**Acceptance criteria.**
- New `fit_power_law_model` with documented model form, fit in log-log space
- Power-law BIC weight included in `_posterior_model_weights` competition (now four-way)
- New `indeterminate_reason = "power_law_shape"` with precedence between `neither_model_fits` and `ambiguous_evidence`
- Plain-language note added to `_summary.py`
- Test: cubic and square-root series → indeterminate with `"power_law_shape"` at the *default* `min_fit_quality`
- Test: clean logistic / exponential / linear → still classify decisively (power-law does not steal weight)
- README "Failure modes / Polynomial" mitigation rewritten — no manual threshold tuning needed

**Files.** [_fit.py](../../src/project_verge/_fit.py), [_api.py](../../src/project_verge/_api.py), [_types.py](../../src/project_verge/_types.py), [_summary.py](../../src/project_verge/_summary.py), tests, [README.md](../../README.md)

---

### T-28: Auto-downgrade verdict to indeterminate when weight CI is too wide
**Category:** Methodology + Code · **Effort:** S · **Depends on:** T-07 · **Status:** Done in commit `ecab902`

**Problem.** Surfaced by writing the [README's "Failure modes / Random-walk-like or unstructured series"](../../README.md) section in T-10. Random-walk-like data (nondecreasing cumulative noise) produces a confident-looking verdict line — `Verdict: leveling off (logistic, 0.78 confidence; 90% CI [0.35, 1.00])`. T-07's weight CI is wide, signaling fragility, but the headline still reads "0.78 confidence." A user not paying close attention to the CI suffix can be misled by the headline. The CI is the load-bearing honesty signal but it is too easy to skip past.

**Proposal.** Add a new indeterminate gate that fires when the bootstrap weight CI on the winning model spans more than a configurable threshold (default ~0.40 — clean cases hit ≈ 0, noisy real-data ≈ 0.05–0.20, random walks ≈ 0.65). Maps to new `indeterminate_reason = "fragile_verdict"`. Place in the precedence chain after `signal_disagreement` (it is a fallback for cases that pass all earlier gates but still produce unstable verdicts under resampling).

**Acceptance criteria.**
- New parameter `max_weight_ci_width: float = 0.40` on `analyze_growth`, with input validation
- New `indeterminate_reason` value `"fragile_verdict"` plus plain-language note in `_summary.py`
- Gate is **skipped** when `weight_intervals is None` (bootstrap didn't run, no signal to act on)
- Default threshold chosen via empirical calibration: clean cases must not be flagged, random walks must be flagged
- Test: seeded random-walk synthetic series → indeterminate with `"fragile_verdict"`
- Test: clean logistic / linear / exponential → not flagged
- README "Failure modes / Random-walk-like" rewritten — the gate is now automatic; users no longer need to read the CI suffix to catch this

**Files.** [_api.py](../../src/project_verge/_api.py), [_types.py](../../src/project_verge/_types.py), [_summary.py](../../src/project_verge/_summary.py), tests, [README.md](../../README.md)

---

## P2 — Robustness and quality

### T-12: AICc instead of (or alongside) BIC
**Category:** Methodology · **Effort:** S · **Status:** Done in commit `d3194f3`

**Problem.** With n typically 8–30, BIC's asymptotic regularity is shaky and it over-penalizes the logistic. AICc has explicit small-sample correction.

**Proposal.** Compute AICc = AIC + 2k(k+1)/(n−k−1). Either replace BIC for the headline weights, or add a `criterion` parameter and let the caller pick. Document why.

**Acceptance criteria.**
- `aicc` field on `ModelFit`
- `analyze_growth(criterion="aicc" | "bic")`, default `"aicc"`
- README explains the choice
- Test: AICc weights differ from BIC weights as expected on small n

**Files.** [_fit.py](../../src/project_verge/_fit.py), [_api.py](../../src/project_verge/_api.py), [_types.py](../../src/project_verge/_types.py), tests

---

### T-13: Log-normal assumption checks
**Category:** Methodology · **Effort:** S · **Status:** Done in commit `274eb97`

**Problem.** Log-normal observation model is asserted but never checked. Heteroscedasticity and autocorrelation silently break BIC/AICc.

**Proposal.** After fitting the preferred model, run Shapiro-Wilk on log-residuals and Ljung-Box on residual autocorrelation. Surface as warnings on `Diagnostics`.

**Acceptance criteria.**
- New diagnostic fields: `residual_normality_pvalue`, `residual_autocorr_pvalue`
- Warnings appended to `fit_warnings` (or new `assumption_warnings`) when p < 0.05
- Tests on known violators

**Files.** [_diagnostics.py](../../src/project_verge/_diagnostics.py), [_types.py](../../src/project_verge/_types.py), tests

---

### T-14: Tie indeterminate threshold to documented evidence bands
**Category:** Methodology · **Effort:** S · **Status:** Done in commit `3cdde75`

**Problem.** The 0.70 cutoff in [_api.py:53](../../src/project_verge/_api.py#L53) is a magic number that corresponds to roughly ΔBIC ≈ 1.7 — Kass & Raftery's "barely worth mentioning" band.

**Proposal.** Replace the magic number with an `evidence_strength` parameter mapping to documented bands (`"positive"` ΔBIC > 2, `"strong"` > 6, `"decisive"` > 10). Default to `"strong"`.

**Acceptance criteria.**
- New `evidence_strength` parameter on `analyze_growth`
- Mapping documented in docstring and README
- Existing tests adjusted (or pinned to `"positive"` for back-compat)

**Files.** [_api.py](../../src/project_verge/_api.py), [README.md](../../README.md)

---

### T-15: Smoothing / noise-tolerance path for non-monotone data
**Category:** Methodology + Code · **Effort:** M · **Status:** Done in commit `99d4ff7`

**Problem.** [_fit.py:39-40](../../src/project_verge/_fit.py#L39-L40) rejects every real-world series. Most monthly time series have noise.

**Proposal.** Add `allow_smoothing` parameter; when True, apply a small rolling-median or LOWESS smoother before validation, log the action in the result. Keep default strict.

**Open question.** Which smoother? Suggest rolling-median first (parameter-free, handles outliers) and revisit.

**Acceptance criteria.**
- `analyze_growth(time, values, allow_smoothing=True)` accepts noisy non-monotone input
- Smoothing applied + recorded in `assumptions` (see T-23) or a new `transform_log` field
- Default behavior unchanged
- Tests on known noisy real-world data (T-11 fixture works)

**Files.** [_fit.py](../../src/project_verge/_fit.py), [_api.py](../../src/project_verge/_api.py), [_types.py](../../src/project_verge/_types.py), tests

---

### T-16: Multi-start optimization for the logistic fit
**Category:** Code · **Effort:** S · **Status:** Done in commit `fcc8342`

**Problem.** [_fit.py:168-178](../../src/project_verge/_fit.py#L168-L178) uses a single heuristic initial guess; on noisy or partial-S data, the optimizer can land in local minima.

**Proposal.** Run 5–10 starts varying K and t0 across plausible ranges; keep the best RSS.

**Acceptance criteria.**
- `n_starts` parameter (default 8)
- Tests on noisy partial-S series where single-start currently produces worse RSS

**Files.** [_fit.py](../../src/project_verge/_fit.py), tests

---

### T-17: Strengthen tests
**Category:** Code · **Effort:** M · **Depends on:** T-11 (soft, for fixture) · **Status:** Done in commit `eff41d2`

**Problem.** [tests/test_analysis.py](../../tests/test_analysis.py) covers happy paths only. No noise tests, no wrong-model robustness tests, no calibration test.

**Proposal.**
- Noisy variants of existing happy-path tests at multiple SNRs (seeded)
- Wrong-model series (polynomial, Gompertz, linear) → confirm graceful indeterminate
- Calibration test: many synthetic logistic series, verify `p_logistic > 0.9` for at least X% (X chosen empirically)
- Real-data smoke test (T-11 fixture)
- Property-based test: `analyze_growth` is invariant to time-origin shift

**Acceptance criteria.**
- New test files / sections cover the above
- Failing cases surface as warnings or indeterminate, never as exceptions

**Files.** [tests/](../../tests/)

---

### T-18: Better `forecast_mae` aggregator
**Category:** Code · **Effort:** S · **Status:** Done in commit `0509163`

**Problem.** [_diagnostics.py:59-94](../../src/project_verge/_diagnostics.py#L59-L94) returns `inf` if any single rolling fit fails to converge — destroying signal from the converged forecasts.

**Proposal.** Use median; also report `fraction_converged`. Consider a tuple or a dataclass field.

**Acceptance criteria.**
- `forecast_mae_*` fields restructured (e.g. `forecast_median_log_error_*` and `forecast_convergence_rate_*`)
- Tests: synthetic series where one window fails — overall metric remains finite and informative

**Files.** [_diagnostics.py](../../src/project_verge/_diagnostics.py), [_types.py](../../src/project_verge/_types.py), tests

---

### T-19: Calibration evidence in docs
**Category:** Docs · **Effort:** M · **Depends on:** T-17 · **Status:** Done in commit `904907e`

**Problem.** Headline number is a probability with no calibration evidence.

**Proposal.** Generate a calibration plot from many synthetic series at known truth (script in `examples/` or a notebook). Include the plot in README.

**Acceptance criteria.**
- Reproducible script that generates calibration data
- Plot committed and referenced in README
- README has a "How calibrated are these probabilities?" section

**Files.** new `examples/calibration.py`, new `docs/calibration.png`, [README.md](../../README.md)

---

## P3 — Polish and process

### T-20: `Literal` / `Enum` types for `model_name` and `preferred_model`
**Category:** Code · **Effort:** S · **Status:** Done in commit `993bc56`

Replace `str` annotations on [_types.py:13](../../src/project_verge/_types.py#L13) and [_types.py:46](../../src/project_verge/_types.py#L46) with `typing.Literal[...]`. Catches typos at type-check time.

**Files.** [_types.py](../../src/project_verge/_types.py)

---

### T-21: `npt.ArrayLike` type hints
**Category:** Code · **Effort:** S · **Status:** Done in commit `e860e42`

`Sequence[float]` in [_fit.py:17-18](../../src/project_verge/_fit.py#L17-L18) admits strings and isn't the numpy convention. Switch to `numpy.typing.ArrayLike`.

**Files.** [_fit.py](../../src/project_verge/_fit.py)

---

### T-22: Resolve `min_points` duplicate validation
**Category:** Code · **Effort:** S · **Status:** Done in commit `37ad665`

`min_points` is validated in both `prepare_inputs` and `_fit_model`. The reason is sound (rolling-window callers) but the surface is confusing. Either add a clarifying comment, or factor so only one path validates.

**Files.** [_fit.py](../../src/project_verge/_fit.py)

---

### T-23: Make `assumptions` field structured (or remove)
**Category:** Code · **Effort:** S · **Depends on:** T-15 · **Status:** Done in commit `ab5695a`

[_api.py:62-67](../../src/project_verge/_api.py#L62-L67) returns a fixed prose tuple regardless of what actually happened during the analysis. Either remove (it's docstring material) or convert to structured flags: `{"used_lognormal": True, "n_observations": 18, "smoothing_applied": False, "criterion": "aicc"}`.

**Files.** [_types.py](../../src/project_verge/_types.py), [_api.py](../../src/project_verge/_api.py)

---

### T-24: Glossary in docs
**Category:** Docs · **Effort:** S · **Status:** Done in commit `ffd7716`

Plain-English definitions for: BIC / AICc, log-normal observation model, identifiability, carrying capacity, posterior weight, indeterminate. Either a new `GLOSSARY.md` or a section in README.

**Files.** new `GLOSSARY.md` or [README.md](../../README.md)

---

### T-25: `CHANGELOG.md`
**Category:** Docs · **Effort:** S · **Status:** Done in commit `7f4bc98`

Standard Keep-a-Changelog format. Pre-populate with the v0.1.0 entry once that ships.

**Files.** new `CHANGELOG.md`

---

### T-26: Defer release workflow until P0/P1 land
**Category:** Process · **Effort:** S

[PROJECT_PLAN.md](PROJECT_PLAN.md) currently lists "release workflow + first tagged release" as the next milestone. Several P0/P1 tickets here change the public API (new fields, new return values, possibly a 3-way verdict). Either tag pre-releases (`v0.1.0a1`, etc.) or hold the tag until at least T-01 through T-04 are merged.

**Files.** [PROJECT_PLAN.md](PROJECT_PLAN.md)

---

## Suggested release groupings

- **v0.1.0 — "Honest shape"**: T-01, T-02, T-03, T-04, T-26
- **v0.2.0 — "Real question"**: T-05, T-06, T-07, T-08, T-09, T-10, T-11, T-27, T-28
- **v0.3.0 — "Robustness"**: T-12, T-13, T-14, T-15, T-16, T-17, T-18, T-19
- **Polish (any time)**: T-20, T-21, T-22, T-23, T-24, T-25
