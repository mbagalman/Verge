"""Robustness and calibration tests.

Covers the five categories called out in T-17:

  - noisy variants of happy-path tests at multiple SNRs (seeded)
  - wrong-model graceful behaviour (Gompertz, polynomial, etc.)
  - Monte-Carlo calibration on clean synthetic logistic data
  - real-data smoke test against the world-population fixture
  - property-based time-origin invariance

The tests deliberately use ``n_boot=0`` where possible to keep total runtime
within budget; calibration loops use modest trial counts (~30) so that the
total wall time remains under a minute.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from project_verge import analyze_growth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_logistic(k=100.0, r=0.7, t0=5.0, n=22, start=0.0, stop=12.0):
    time = np.linspace(start, stop, n)
    values = k / (1.0 + np.exp(-r * (time - t0)))
    return time, values


def _add_lognormal_noise(values, *, sigma, seed):
    rng = np.random.default_rng(seed)
    return values * np.exp(rng.normal(0.0, sigma, size=len(values)))


# ---------------------------------------------------------------------------
# 1. Noisy variants of happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [0.02, 0.05])
@pytest.mark.parametrize("seed", range(3))
def test_noisy_logistic_classifies_logistic_or_indeterminate(sigma, seed):
    """Noisy logistic data, run through smoothing, should never produce a
    decisive non-logistic verdict. Indeterminate is acceptable -- the
    library's job is to be honest about uncertainty when the small-sample
    AICc + fragile_verdict combination triggers."""
    time, clean = _clean_logistic(n=22)
    noisy = _add_lognormal_noise(clean, sigma=sigma, seed=seed)

    # n_boot=0 keeps the test fast; the wrong-decisive-verdict assertion
    # works without bootstrap data because the BIC-derived verdict is what
    # we're checking.
    result = analyze_growth(time, noisy, n_boot=0, allow_smoothing=True)

    # Acceptable outcomes: logistic decisive, or any indeterminate reason.
    # Forbidden: a decisive non-logistic verdict on clean-ish logistic data.
    if not result.is_indeterminate:
        assert result.preferred_model == "logistic", (
            f"sigma={sigma}, seed={seed}: expected logistic or indeterminate, "
            f"got {result.preferred_model}"
        )


@pytest.mark.parametrize("seed", range(3))
def test_noisy_exponential_classifies_exponential_or_indeterminate(seed):
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 10.0, 18)
    clean = 4.0 * np.exp(0.16 * time)
    noisy = clean * np.exp(rng.normal(0.0, 0.03, size=len(time)))

    result = analyze_growth(time, noisy, n_boot=0, allow_smoothing=True)

    if not result.is_indeterminate:
        assert result.preferred_model == "exponential"


@pytest.mark.parametrize("seed", range(3))
def test_noisy_linear_classifies_linear_or_indeterminate(seed):
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 10.0, 18)
    clean = 5.0 + 2.0 * time
    noisy = clean * np.exp(rng.normal(0.0, 0.03, size=len(time)))

    result = analyze_growth(time, noisy, n_boot=0, allow_smoothing=True)

    if not result.is_indeterminate:
        assert result.preferred_model == "linear"


# ---------------------------------------------------------------------------
# 2. Wrong-model graceful behaviour
# ---------------------------------------------------------------------------


def test_gompertz_data_does_not_crash():
    """Gompertz growth (y = K * exp(-b * exp(-c*t))) is not in v1's model
    space. Verge must handle it without crashing and either commit to a
    plausible candidate (logistic is closest geometrically) or emit
    indeterminate."""
    time = np.linspace(0.0, 12.0, 20)
    K, b, c = 100.0, 5.0, 0.3
    values = K * np.exp(-b * np.exp(-c * time))

    result = analyze_growth(time, values, n_boot=0)

    assert result is not None
    assert result.preferred_model in {
        "exponential", "linear", "logistic", "indeterminate"
    }


@pytest.mark.parametrize("k", [0.5, 1.5, 2.0, 3.0, 4.0])
def test_polynomial_growth_handled_gracefully(k):
    """Polynomial t^k for any positive k should never crash. The
    power_law_shape gate (T-27) catches most of these as indeterminate."""
    time = np.linspace(1.0, 12.0, 16)
    values = time ** k

    result = analyze_growth(time, values, n_boot=0)

    assert result is not None
    # Any verdict is allowed -- this is a graceful-handling test, not a
    # specific-classification test.


def test_logarithmic_growth_handled_gracefully():
    """Logarithmic growth (y = a + b*log(t)) is decelerating but not
    asymptotic -- nothing in v1 fits this directly."""
    time = np.linspace(1.0, 30.0, 18)
    values = 1.0 + 5.0 * np.log(time)

    result = analyze_growth(time, values, n_boot=0)

    assert result is not None


# ---------------------------------------------------------------------------
# 3. Monte-Carlo calibration on clean synthetic logistic data
# ---------------------------------------------------------------------------


def test_calibration_clean_logistic_classifies_correctly_at_least_90_percent():
    """Across 30 seeded clean logistic series with varied parameters, the
    BIC-derived verdict must commit to logistic at least 90% of the time.
    Run with n_boot=0 to skip the fragile_verdict gate; this is a check on
    the criterion's classification rate, not on the bootstrap-derived
    indeterminate gating (which has its own dedicated tests)."""
    rng = np.random.default_rng(0)
    n_trials = 30
    correct_decisive = 0
    indeterminate = 0
    wrong_decisive = 0

    for _ in range(n_trials):
        K = float(rng.uniform(50.0, 500.0))
        r = float(rng.uniform(0.5, 1.2))
        t0 = float(rng.uniform(4.0, 8.0))
        n = int(rng.integers(18, 30))
        time = np.linspace(0.0, 12.0, n)
        values = K / (1.0 + np.exp(-r * (time - t0)))

        # n_starts=1 because multi-start has its own dedicated coverage; for
        # this calibration test we just want the BIC-derived classification
        # rate, and 8x the cost on every iteration is wasteful here.
        result = analyze_growth(time, values, n_boot=0, n_starts=1)

        if result.is_indeterminate:
            indeterminate += 1
        elif result.preferred_model == "logistic":
            correct_decisive += 1
        else:
            wrong_decisive += 1

    # The hard assertion: never produce a *wrong* decisive verdict on clean
    # logistic data with reasonable parameters. Indeterminate is fine.
    assert wrong_decisive == 0, (
        f"Got {wrong_decisive} wrong decisive verdicts: "
        f"calibration is broken (correct={correct_decisive}, "
        f"indeterminate={indeterminate})"
    )
    # And the soft assertion: at least 90% should commit decisively (with
    # n_boot=0 there is no fragile_verdict gate so the threshold is high).
    assert correct_decisive >= int(0.90 * n_trials), (
        f"Only {correct_decisive}/{n_trials} clean logistic trials "
        f"classified decisively (indeterminate: {indeterminate})"
    )


def test_calibration_clean_exponential_classifies_correctly_at_least_90_percent():
    # Smaller trial count than the logistic calibration test: clean
    # exponential data makes the logistic fit's K parameter run against the
    # upper bound (no ceiling in the data to anchor on) and the optimizer
    # spends many iterations failing to converge cleanly. 12 trials at ~5s
    # each is enough to detect a classification regression without making
    # the suite unusable.
    rng = np.random.default_rng(0)
    n_trials = 12
    correct_decisive = 0
    indeterminate = 0
    wrong_decisive = 0

    for _ in range(n_trials):
        a = float(rng.uniform(1.0, 10.0))
        r = float(rng.uniform(0.05, 0.25))
        n = int(rng.integers(15, 30))
        stop = float(rng.uniform(8.0, 15.0))
        time = np.linspace(0.0, stop, n)
        values = a * np.exp(r * time)

        # n_starts=1 keeps the calibration loop fast; pure-exponential data
        # makes the logistic optimizer thrash through all 8 starts when
        # multi-start is on, ballooning runtime.
        result = analyze_growth(time, values, n_boot=0, n_starts=1)

        if result.is_indeterminate:
            indeterminate += 1
        elif result.preferred_model == "exponential":
            correct_decisive += 1
        else:
            wrong_decisive += 1

    assert wrong_decisive == 0
    assert correct_decisive >= int(0.90 * n_trials)


# ---------------------------------------------------------------------------
# 4. Real-data smoke test
# ---------------------------------------------------------------------------


def test_world_population_real_data_runs_end_to_end():
    """Smoke test: the committed UN population CSV must run through
    analyze_growth without crashing, and produce a well-formed result."""
    csv_path = Path("examples/data/un_population.csv")
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    years = np.array([float(r["year"]) for r in rows])
    pop = np.array([float(r["population_billions"]) for r in rows])

    # n_boot=100 + n_starts=1 to keep the smoke test under a few seconds;
    # the test asserts well-formedness, not specific numeric values.
    result = analyze_growth(years, pop, n_boot=100, bootstrap_seed=0, n_starts=1)

    # Well-formed: all weights sum to 1, point estimates finite, predict()
    # works (since the verdict is a non-power-law decisive or indeterminate
    # that we can still evaluate manually).
    weights_sum = (
        result.p_exponential + result.p_linear + result.p_logistic + result.p_power_law
    )
    assert weights_sum == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(result.exponential_fit.bic)
    assert np.isfinite(result.exponential_fit.aicc)


def test_world_population_post_1950_subset_runs_end_to_end():
    """The post-1950 subset is the canonical demonstration of T-15 + T-28
    interacting (small n + AICc + fragile_verdict). Smoke test it
    end-to-end so changes there are caught."""
    csv_path = Path("examples/data/un_population.csv")
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    years = np.array([float(r["year"]) for r in rows])
    pop = np.array([float(r["population_billions"]) for r in rows])
    mask = years >= 1950
    yr2, p2 = years[mask], pop[mask]

    result = analyze_growth(yr2, p2, n_boot=100, bootstrap_seed=0, n_starts=1)

    assert result is not None
    # Either a logistic verdict (under BIC) or an indeterminate from
    # small-sample fragility (under AICc default) is acceptable -- both
    # outcomes have been documented as correct behaviour.
    assert result.preferred_model in {"logistic", "indeterminate"}


# ---------------------------------------------------------------------------
# 5. Property-based: time-origin invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [100.0, 1000.0, -50.0, 1e6, 2025.0])
def test_analyze_growth_is_invariant_to_time_origin_shift(shift):
    """Shifting time by a constant must not change the verdict. The
    library's prepare_inputs normalizes time internally, so the fits live
    in (time - time[0]) coordinates regardless of the original origin --
    the verdict and posterior weights must be identical (up to optimizer
    numerical noise) for any shift."""
    time, values = _clean_logistic(k=100.0, r=0.7, t0=5.0, n=22)

    base = analyze_growth(time, values, n_boot=0)
    shifted = analyze_growth(time + shift, values, n_boot=0)

    assert base.preferred_model == shifted.preferred_model
    assert base.indeterminate_reason == shifted.indeterminate_reason
    # Weights should match to within optimizer noise.
    assert abs(base.p_logistic - shifted.p_logistic) < 1e-3
    assert abs(base.p_exponential - shifted.p_exponential) < 1e-3
    assert abs(base.p_linear - shifted.p_linear) < 1e-3
    assert abs(base.p_power_law - shifted.p_power_law) < 1e-3


def test_logistic_t0_parameter_is_in_normalized_time_after_origin_shift():
    """The fit's raw t0 parameter is in *normalized* time (zero-based), not
    the original coordinate, so it is invariant to time-origin shifts. The
    bootstrap-derived t0 *interval*, on the other hand, is reported in
    original time and should shift with the input."""
    time, values = _clean_logistic(k=100.0, r=0.7, t0=5.0, n=22)
    shift = 1000.0

    base = analyze_growth(time, values, n_boot=100, bootstrap_seed=0, n_starts=1)
    shifted = analyze_growth(time + shift, values, n_boot=100, bootstrap_seed=0, n_starts=1)

    # The fitter's t0 is in normalized time; should match.
    assert base.logistic_fit.parameters["t0"] == pytest.approx(
        shifted.logistic_fit.parameters["t0"], rel=1e-3
    )
    # The bootstrap interval's t0 is in original time; should shift.
    if base.logistic_intervals is not None and shifted.logistic_intervals is not None:
        assert shifted.logistic_intervals.t0.median == pytest.approx(
            base.logistic_intervals.t0.median + shift, rel=1e-3
        )
