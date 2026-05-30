import numpy as np
import pytest

from growthshape import analyze_growth
from growthshape._api import _evidence_strength_threshold


def _exp_series(a=4.0, r=0.16, n=15, start=0.0, stop=10.0):
    time = np.linspace(start, stop, n)
    values = a * np.exp(r * (time - time[0]))
    return time, values


def test_evidence_strength_thresholds_match_documented_bands():
    # The thresholds are calibrated against Kass & Raftery's positive /
    # strong / decisive bands. The exact values are rounded to two decimals
    # for clean communication; what matters is the ordering.
    positive = _evidence_strength_threshold("positive")
    strong = _evidence_strength_threshold("strong")
    decisive = _evidence_strength_threshold("decisive")

    assert 0.7 <= positive < strong < decisive < 1.0
    assert positive == 0.75
    assert strong == 0.95
    assert decisive == 0.99


def test_default_evidence_strength_is_strong():
    time, values = _exp_series()
    default_result = analyze_growth(time, values, n_boot=0)
    strong_result = analyze_growth(time, values, n_boot=0, evidence_strength="strong")

    assert default_result.preferred_model == strong_result.preferred_model
    assert default_result.indeterminate_reason == strong_result.indeterminate_reason
    assert default_result.p_exponential == strong_result.p_exponential


def test_borderline_case_is_decisive_under_positive_but_indeterminate_under_strong():
    # World-population full-history (1750-2022) gives p_exponential ~ 0.88
    # under the AICc default. That clears the "positive" band (0.75) but
    # not the "strong" band (0.95), giving the canonical demonstration of
    # the parameter.
    import csv
    from pathlib import Path

    csv_path = Path("examples/data/un_population.csv")
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    years = np.array([float(r["year"]) for r in rows])
    pop = np.array([float(r["population_billions"]) for r in rows])

    positive = analyze_growth(years, pop, n_boot=0, evidence_strength="positive")
    strong = analyze_growth(years, pop, n_boot=0, evidence_strength="strong")

    assert positive.preferred_model == "exponential"
    assert positive.indeterminate_reason is None

    assert strong.preferred_model == "indeterminate"
    assert strong.indeterminate_reason == "ambiguous_evidence"


def test_decisive_band_rejects_world_population_under_either_criterion():
    # The decisive band (0.99) rejects anything below very-high confidence,
    # so the world-population full history -- which sits around 0.88 to 0.79
    # depending on the criterion -- never clears it.
    import csv
    from pathlib import Path

    csv_path = Path("examples/data/un_population.csv")
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    years = np.array([float(r["year"]) for r in rows])
    pop = np.array([float(r["population_billions"]) for r in rows])

    for criterion in ("aicc", "bic"):
        result = analyze_growth(
            years, pop, n_boot=0, evidence_strength="decisive", criterion=criterion
        )
        assert result.preferred_model == "indeterminate"
        assert result.indeterminate_reason == "ambiguous_evidence"


def test_clean_inputs_pass_strong_threshold_decisively():
    # Sanity check: clean exponential / linear / logistic data hits weight
    # ~ 1.00, well above strong (0.95) and even decisive (0.99). These
    # canonical happy-path cases must not flip to indeterminate under the
    # tightened default.
    time, values = _exp_series(n=18)
    assert analyze_growth(time, values, n_boot=0).preferred_model == "exponential"

    time = np.linspace(0.0, 10.0, 18)
    values = 5.0 + 2.0 * time
    assert analyze_growth(time, values, n_boot=0).preferred_model == "linear"

    time = np.linspace(0.0, 12.0, 22)
    values = 100.0 / (1.0 + np.exp(-0.75 * (time - 5.0)))
    assert analyze_growth(time, values, n_boot=0).preferred_model == "logistic"


@pytest.mark.parametrize("bad", ["weak", "extreme", "Strong", "", "0.95"])
def test_invalid_evidence_strength_raises(bad):
    time, values = _exp_series()
    with pytest.raises(ValueError, match="evidence_strength"):
        analyze_growth(time, values, n_boot=0, evidence_strength=bad)


def test_summary_omits_K_when_logistic_is_not_the_leading_model():
    # The world-population full history fires ambiguous_evidence with
    # logistic in 12% weight (not leading). The summary must NOT advertise
    # the wild K from an unidentified logistic fit -- that K interval has
    # no relationship to a real saturation ceiling.
    import csv
    from pathlib import Path

    csv_path = Path("examples/data/un_population.csv")
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    years = np.array([float(r["year"]) for r in rows])
    pop = np.array([float(r["population_billions"]) for r in rows])

    result = analyze_growth(years, pop, n_boot=200, bootstrap_seed=0)
    summary = result.summary()

    assert result.preferred_model == "indeterminate"
    assert result.indeterminate_reason == "ambiguous_evidence"
    assert "Estimated ceiling" not in summary
