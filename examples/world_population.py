"""GrowthShape applied to historical world population.

Data: estimated world population at well-known milestone years from 1750 to
2022. Values for 1950 onward are drawn from the UN World Population Prospects
2022 revision; pre-1950 estimates come from the HYDE 3.2 historical-population
database and standard secondary sources. Values are rounded to the milestone
year at which world population is conventionally said to have reached the
given billion mark, so they are precise enough to characterize the *shape* of
demographic growth but not for fine-grained forecasting.

Run interactively to see the verdict, the 2050 prediction, and a plot::

    python examples/world_population.py

Save the plot instead of showing it::

    python examples/world_population.py --save world_population.png
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from growthshape import analyze_growth


HERE = Path(__file__).parent
DATA_PATH = HERE / "data" / "un_population.csv"


def load_world_population():
    years = []
    populations = []
    with open(DATA_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            years.append(float(row["year"]))
            populations.append(float(row["population_billions"]))
    return np.array(years), np.array(populations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", metavar="PATH", help="save figure to PATH instead of showing")
    args = parser.parse_args()

    years, pop = load_world_population()
    print(
        f"Loaded {len(years)} world-population estimates "
        f"({int(years[0])} to {int(years[-1])})."
    )
    print()

    # The point of running this on two windows is pedagogical: GrowthShape's
    # verdict is a function of *what data you give it*, not a universal
    # claim about the future. The full-history view is dominated by
    # 200+ years of acceleration and gives "accelerating"; the
    # post-1950 view sees the demographic transition and gives
    # "leveling off" with a carrying-capacity estimate.
    print("=" * 60)
    print("Full history (1750-2022)")
    print("=" * 60)
    full = analyze_growth(
        years, pop, n_boot=300, bootstrap_seed=0, horizons=[2050.0, 2100.0]
    )
    print(full.summary())
    if not full.is_indeterminate:
        for horizon in (2050.0, 2100.0):
            pred = full.predict(horizon)
            print(
                f"Prediction for {int(horizon)}: {pred.point:.2f}B "
                f"(90% CI [{pred.low:.2f}, {pred.high:.2f}]B)"
            )

    print()
    print("=" * 60)
    print("Post-1950 only (demographic transition window)")
    print("=" * 60)
    mask = years >= 1950
    modern = analyze_growth(
        years[mask], pop[mask], n_boot=300, bootstrap_seed=0, horizons=[2050.0, 2100.0]
    )
    print(modern.summary())
    if not modern.is_indeterminate:
        for horizon in (2050.0, 2100.0):
            pred = modern.predict(horizon)
            print(
                f"Prediction for {int(horizon)}: {pred.point:.2f}B "
                f"(90% CI [{pred.low:.2f}, {pred.high:.2f}]B)"
            )

    try:
        import matplotlib.pyplot as plt

        from growthshape.plot import plot_growth_analysis
    except ImportError:
        print()
        print("(matplotlib not installed; skipping plot. "
              "Install with: pip install 'growthshape[plot]')")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_growth_analysis(full, ax=axes[0], extrapolate_fraction=0.4)
    axes[0].set_title("Full history (1750-2022)\n" + axes[0].get_title())
    plot_growth_analysis(modern, ax=axes[1], extrapolate_fraction=2.5)
    axes[1].set_title("Post-1950 only\n" + axes[1].get_title())
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f"\nSaved plot to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
