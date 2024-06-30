"""Compare coverage of the explanations generated by Anchor and R-LIME."""

import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

EXAMPLES_DIR_NAME = "src/rlime-examples/examples"

sample_num = 100


def get_weights(idx: int) -> list[float] | None:
    """Get the weights of LIME for the given sample."""
    csv_name = EXAMPLES_DIR_NAME + f"/lime-{idx:04d}.csv"
    try:
        with Path(csv_name).open() as f:
            reader = csv.reader(f)
            return [float(x) for x in next(reader)]
    except FileNotFoundError:
        return None


def get_rlime_weights(idx: int, tau: int) -> list[float] | None:
    """Get the weights of R-LIME for the given sample."""
    csv_name = EXAMPLES_DIR_NAME + f"/newlime-{idx:04d}-{tau:02d}.csv"
    try:
        with Path(csv_name).open() as f:
            reader = csv.reader(f)
            return [float(x) for x in next(reader)]
    except FileNotFoundError:
        return None


def get_anchor_cov(idx: int, tau: int) -> float | None:
    """Get the coverage of Anchor for the given sample."""
    csv_name = EXAMPLES_DIR_NAME + f"/anchor-{idx:04d}-{tau:02d}.csv"
    try:
        with Path(csv_name).open() as f:
            reader = csv.reader(f)
            next(reader)  # Skip the weights.
            return float(next(reader)[1])  # Skip the accuracy.
    except FileNotFoundError:
        return None


def get_rlime_cov(idx: int, tau: int) -> float | None:
    """Get the coverage of R-LIME for the given sample."""
    csv_name = EXAMPLES_DIR_NAME + f"/newlime-{idx:04d}-{tau:02d}.csv"
    try:
        with Path(csv_name).open() as f:
            reader = csv.reader(f)
            next(reader)  # Skip the weights.
            next(reader)  # Skip the rule.
            return float(next(reader)[1])  # Skip the accuracy.
    except FileNotFoundError:
        return None


def main() -> None:
    """Compare coverages of the explanations for Anchor and R-LIME.

    Read coverages from the CSV files in "g-ohara/rlime-examples" repo,
    and print the mean and standard deviation of the coverages.
    """
    anchor_stats: list[tuple[float, float]] = []
    rlime_stats: list[tuple[float, float]] = []

    tau_list = [65, 70, 75, 80, 85, 90]
    for tau in tau_list:
        anchor_covs = [get_anchor_cov(idx, tau) for idx in range(sample_num)]
        anchor_covs = [x for x in anchor_covs if x is not None]
        rlime_covs = [get_rlime_cov(idx, tau) for idx in range(sample_num)]
        rlime_covs = [x for x in rlime_covs if x is not None]

        print("----")
        print(f"tau = {tau}")
        print("Anchor:")
        print(f" Mean: {mean(anchor_covs)}")
        print(f" Std: {stdev(anchor_covs)}")
        print("R-LIME:")
        print(f" Mean: {mean(rlime_covs)}")
        print(f" Std: {stdev(rlime_covs)}")

        anchor_stats.append((mean(anchor_covs), stdev(anchor_covs)))
        rlime_stats.append((mean(rlime_covs), stdev(rlime_covs)))

    plt.figure()
    plt.rcParams["font.size"] = 15
    for stats in [anchor_stats, rlime_stats]:
        plt.errorbar(
            [tau / 100 for tau in tau_list],
            [x[0] for x in stats],
            yerr=[x[1] for x in stats],
            capsize=5,
            fmt="--o",
        )

    plt.xticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    plt.xlabel("Threshold of Accuracy")
    plt.ylabel("Coverage")
    plt.legend(["Anchor", "R-LIME"])
    plt.tight_layout()
    plt.savefig("comp_cov.eps")
    plt.close()


if __name__ == "__main__":
    main()