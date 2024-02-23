"""Module providing a function creating a box plot of the data in a file."""

import csv

import matplotlib.pyplot as plt


def main() -> None:
    """Main function of the module."""

    plt.figure()

    accs = []
    for tau_percent in [70, 80, 90]:
        file_name = f"{tau_percent}.csv"
        lime_acc, rlime_acc = read_data(file_name)
        accs.append(lime_acc)
        accs.append(rlime_acc)

    bp = plt.boxplot(
        accs,
        labels=["LIME", "R-LIME"] * 3,
        widths=0.7,
        positions=[1, 2, 3.5, 4.5, 6, 7],
        showfliers=False,
        patch_artist=True,
    )
    colors = ["#377eb8", "#4daf4a"] * 3
    for patch, med, color in zip(bp["boxes"], bp["medians"], colors):
        patch.set_facecolor(color)
        med.set_color("black")

    for i, tau_percent in enumerate([70, 80, 90]):
        plt.text(1.0 + 2.5 * i, 0.36, r"$\tau=$" + f"0.{tau_percent}")

    plt.hlines(
        [0.7, 0.8, 0.9],
        [0, 2.5, 5],
        [3.0, 5.5, 8],
        colors="black",
        linestyles="dashed",
    )
    # plt.title("Accuracy of LIME and R-LIME")
    plt.ylabel("Accuracy")
    plt.savefig("box_plot.eps")


def read_data(filename: str) -> tuple[list[float], list[float]]:
    """Reads data from a file and returns it as a list of lists."""
    data = []
    with open(filename, newline="", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            data.append([float(x) for x in row])
    return data[0], data[1]


if __name__ == "__main__":
    main()
