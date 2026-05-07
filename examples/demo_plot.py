"""Demo: plot a Verge growth analysis.

Requires the optional plotting extra:

    pip install 'project-verge[plot]'

Run interactively to pop up a window:

    python examples/demo_plot.py

Or save to a file:

    python examples/demo_plot.py --save out.png
"""

import argparse

import numpy as np

from project_verge import analyze_growth
from project_verge.plot import plot_growth_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", metavar="PATH", help="save figure to PATH instead of showing")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    time = np.linspace(0.0, 12.0, 18)
    values = 120.0 / (1.0 + np.exp(-0.7 * (time - 6.0)))
    result = analyze_growth(time, values, n_boot=200, bootstrap_seed=0)

    print(result.summary())

    plot_growth_analysis(result)
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=120)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
