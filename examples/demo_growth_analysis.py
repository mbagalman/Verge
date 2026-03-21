"""Runnable demo for Project Verge."""

import numpy as np

from project_verge import analyze_growth


def run_case(name, time, values):
    result = analyze_growth(time, values)
    print(f"{name}:")
    print(f"  p_exponential={result.p_exponential:.3f}")
    print(f"  p_logistic={result.p_logistic:.3f}")
    print(f"  preferred_model={result.preferred_model}")
    print(f"  is_indeterminate={result.is_indeterminate}")
    print(f"  identifiability_warnings={result.diagnostics.identifiability_warnings}")
    print()


def main():
    time = np.linspace(0.0, 12.0, 18)

    exponential_values = 3.0 * np.exp(0.18 * time)
    logistic_values = 100.0 / (1.0 + np.exp(-0.8 * (time - 6.0)))
    early_logistic_time = np.linspace(0.0, 5.0, 10)
    early_logistic_values = 200.0 / (1.0 + np.exp(-0.35 * (early_logistic_time - 12.0)))

    run_case("Clear exponential", time, exponential_values)
    run_case("Clear logistic", time, logistic_values)
    run_case("Early logistic / ambiguous", early_logistic_time, early_logistic_values)


if __name__ == "__main__":
    main()

