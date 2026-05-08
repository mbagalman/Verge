"""Generate calibration evidence for Verge's posterior model weights.

Runs many seeded synthetic trials drawn from each of the in-family model
classes (exponential, linear, logistic), records the BIC/AICc-derived
posterior weight on the *winning* model alongside the ground truth, and
plots empirical accuracy versus predicted confidence with the y=x
"perfect calibration" reference.

The plot is saved to docs/calibration.png and is what the README's
"How calibrated are these probabilities?" section refers to. The script
is reproducible -- seed-controlled and committed alongside the plot --
so reviewers can re-run and verify.

Usage::

    python examples/calibration.py                       # save to docs/calibration.png
    python examples/calibration.py --output other.png    # save elsewhere
    python examples/calibration.py --show                # show interactively

Methodology choices:
  - 200 trials per truth class (logistic / exponential / linear), seeded
  - n in [12, 30], parameters sampled from realistic ranges
  - multiplicative log-normal noise at sigma drawn from [0, 0.10]
  - allow_smoothing=True so the noisy series is admitted, n_starts=1 and
    n_boot=0 to keep the experiment under a few minutes
  - "predicted" is argmax(p_exponential, p_linear, p_logistic, p_power_law)
  - "correct" is (predicted == truth_class)

The calibration target is the y=x diagonal: at 0.95 predicted confidence,
empirical accuracy should also sit near 0.95. Deviations highlight where
the criterion-derived probability is mis-calibrated -- usually under
smoothing-related shape distortions on linear inputs.
"""

import argparse
from pathlib import Path

import numpy as np

from project_verge import analyze_growth


HERE = Path(__file__).parent
DEFAULT_OUTPUT = HERE.parent / "docs" / "calibration.png"

TRUTH_CLASSES = ("exponential", "linear", "logistic")
N_TRIALS_PER_CLASS = 300
SIGMA_RANGE = (0.0, 0.10)
N_RANGE = (12, 30)
SEED = 0


def _generate_one_trial(rng, truth_class):
    n = int(rng.integers(*N_RANGE))
    sigma = float(rng.uniform(*SIGMA_RANGE))

    if truth_class == "logistic":
        K = float(rng.uniform(50.0, 500.0))
        r = float(rng.uniform(0.5, 1.2))
        t0 = float(rng.uniform(4.0, 8.0))
        time = np.linspace(0.0, 12.0, n)
        clean = K / (1.0 + np.exp(-r * (time - t0)))
    elif truth_class == "exponential":
        a = float(rng.uniform(1.0, 10.0))
        r_param = float(rng.uniform(0.05, 0.25))
        stop = float(rng.uniform(8.0, 14.0))
        time = np.linspace(0.0, stop, n)
        clean = a * np.exp(r_param * time)
    else:  # linear
        a = float(rng.uniform(1.0, 10.0))
        b = float(rng.uniform(0.5, 5.0))
        stop = float(rng.uniform(8.0, 14.0))
        time = np.linspace(0.0, stop, n)
        clean = a + b * time

    noisy = clean * np.exp(rng.normal(0.0, sigma, size=n))
    result = analyze_growth(
        time, noisy, n_boot=0, allow_smoothing=True, n_starts=1
    )
    weights = {
        "exponential": result.p_exponential,
        "linear": result.p_linear,
        "logistic": result.p_logistic,
        "power_law": result.p_power_law,
    }
    predicted = max(weights, key=weights.__getitem__)
    return predicted, weights[predicted]


def run_trials(seed=SEED):
    """Return a dict mapping truth class -> list of (predicted, max_weight)."""
    rng = np.random.default_rng(seed)
    results = {cls: [] for cls in TRUTH_CLASSES}
    for cls in TRUTH_CLASSES:
        for _ in range(N_TRIALS_PER_CLASS):
            predicted, max_weight = _generate_one_trial(rng, cls)
            results[cls].append((predicted, max_weight))
    return results


def calibration_curve(weights, correct, n_bins=10, weight_range=(0.4, 1.0)):
    """Bin (weight, correct) pairs and return (bin_centers, empirical_accuracy, counts)."""
    weights = np.asarray(weights, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(weight_range[0], weight_range[1], n_bins + 1)
    bin_idx = np.digitize(weights, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    empirical = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = bin_idx == i
        counts[i] = int(mask.sum())
        if counts[i] > 0:
            empirical[i] = float(np.mean(correct[mask]))
    return centers, empirical, counts


def summarize(results):
    """Print a per-truth-class summary table to stdout."""
    print(f"{'truth':>13} {'n':>5} {'correct':>9} {'pct':>6}")
    print("-" * 38)
    overall_correct = 0
    overall_n = 0
    for cls, trials in results.items():
        n = len(trials)
        correct = sum(1 for predicted, _ in trials if predicted == cls)
        pct = 100.0 * correct / n if n else 0.0
        print(f"{cls:>13} {n:>5} {correct:>9} {pct:>5.1f}%")
        overall_correct += correct
        overall_n += n
    print("-" * 38)
    print(
        f"{'overall':>13} {overall_n:>5} {overall_correct:>9} "
        f"{100.0 * overall_correct / overall_n:>5.1f}%"
    )


def plot(results, output_path, show=False):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    # Aggregated calibration curve across all truth classes.
    all_weights = []
    all_correct = []
    for cls, trials in results.items():
        for predicted, weight in trials:
            all_weights.append(weight)
            all_correct.append(predicted == cls)
    centers, empirical, counts = calibration_curve(all_weights, all_correct)

    # Plot the aggregated curve, dropping bins with zero observations.
    valid = counts > 0
    ax.plot(
        centers[valid],
        empirical[valid],
        marker="o",
        linewidth=2.0,
        color="tab:blue",
        label="aggregated (all truth classes)",
    )

    # Annotate bin counts.
    for c, e, n in zip(centers[valid], empirical[valid], counts[valid]):
        ax.annotate(
            f"n={n}",
            xy=(c, e),
            xytext=(0, 8),
            textcoords="offset points",
            fontsize=8,
            ha="center",
            color="gray",
        )

    # Per-truth-class curves, lighter lines.
    palette = {
        "exponential": "tab:orange",
        "linear": "tab:green",
        "logistic": "tab:red",
    }
    for cls, trials in results.items():
        weights = [w for _, w in trials]
        correct = [predicted == cls for predicted, _ in trials]
        c, e, n = calibration_curve(weights, correct)
        valid = n > 0
        ax.plot(
            c[valid],
            e[valid],
            marker="s",
            linewidth=1.0,
            linestyle="--",
            alpha=0.5,
            color=palette[cls],
            label=f"truth = {cls} (n={sum(n)})",
        )

    # y = x reference.
    ax.plot([0.4, 1.0], [0.4, 1.0], color="gray", linewidth=1.0, label="perfect calibration (y = x)")

    ax.set_xlabel("Predicted confidence (max posterior weight)")
    ax.set_ylabel("Empirical accuracy (predicted == truth)")
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        f"Verge calibration ({sum(len(t) for t in results.values())} trials, "
        f"seed={SEED})"
    )
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    if show:
        plt.show()
    fig.savefig(output_path, dpi=120)
    print(f"\nSaved calibration plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output path for the calibration plot (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--show", action="store_true", help="show plot interactively")
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="RNG seed for reproducibility (default: 0)",
    )
    args = parser.parse_args()

    print(
        f"Running {N_TRIALS_PER_CLASS} trials per truth class "
        f"({len(TRUTH_CLASSES)} classes, seed={args.seed})..."
    )
    results = run_trials(seed=args.seed)
    summarize(results)
    plot(results, args.output, show=args.show)


if __name__ == "__main__":
    main()
