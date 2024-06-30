"""Compare LIME and R-LIME on the recidivism dataset."""

import csv
import multiprocessing
from functools import partial
from pathlib import Path

import numpy as np
from rlime import rlime_lime
from rlime.rlime_types import Dataset, IntArray, Rule
from rlime.sampler import Sampler
from rlime.utils import get_trg_sample, load_dataset
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.preprocessing import StandardScaler

EXAMPLES_DIR_NAME = "src/rlime-examples/examples"


def main() -> None:
    """Main function."""
    # Load the dataset.
    dataset = load_dataset("recidivism", "src/datasets/", balance=True)

    # Learn the black box model.
    black_box = RandomForestClassifier(n_estimators=100, n_jobs=1)
    black_box.fit(dataset.train, dataset.labels_train)

    # Get the target instances.
    sample_num = 100

    for tau in [0.90]:
        print(f"tau = {tau:.2f}")

        result_list = multiprocessing.Manager().list()

        func = partial(
            compare_lime_and_newlime,
            dataset=dataset,
            black_box=black_box,
            tau=tau,
            result_list=result_list,
        )

        with multiprocessing.Pool() as pool:
            pool.map(func, range(sample_num))

        lime_acc, rlime_acc = zip(*result_list)
        print("LIME")
        print(f"  Mean: {np.mean(lime_acc):.4f}")
        print(f"  Std : {np.std(lime_acc):.4f}")
        print("R-LIME")
        print(f"  Mean: {np.mean(rlime_acc):.4f}")
        print(f"  Std : {np.std(rlime_acc):.4f}")

        # Save the results to a CSV file.
        with open(f"{int(tau*100)}.csv", "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(lime_acc)
            writer.writerow(rlime_acc)


def calc_accuracy(
    samples: IntArray,
    labels: IntArray,
    lime_weights: list[float],
    rlime_weights: list[float],
    lime_scaler: StandardScaler,
) -> tuple[float, float]:
    """Calculate the precision of LIME and R-LIME.

    Parameters
    ----------
    samples : IntArray
        The samples to evaluate.
    labels : IntArray
        The labels of the samples.
    lime_weights : list[float]
        The weights of LIME.
    lime_scaler : preprocessing.StandardScaler
        The scaler of LIME.
    arm : Arm
        The rule of R-LIME.

    Returns:
    -------
    tuple[float, float]
        The precision of LIME and R-LIME.
    """
    rlime_scaler = StandardScaler()
    rlime_scaler.fit(samples)

    # Get predictions of LIME and R-LIME.
    lime_pred = np.dot(lime_scaler.transform(samples), lime_weights) > 0
    rlime_pred = np.dot(rlime_scaler.transform(samples), rlime_weights) > 0

    # Calculate the precision of LIME and R-LIME.
    lime_acc = np.sum(lime_pred == labels) / len(labels)
    rlime_acc = np.sum(rlime_pred == labels) / len(labels)

    return lime_acc, rlime_acc


def compare_lime_and_newlime(
    idx: int,
    dataset: Dataset,
    black_box: RandomForestClassifier,
    tau: float,
    result_list: list[tuple[float, float]],
) -> None:
    """Compare LIME and R-LIME.

    Parameters
    ----------
    idx : int
        The index of the target instance.
    dataset : Dataset
        The dataset.
    black_box : RandomForestClassifier
        The black box model.
    tau : float
        The threshold of R-LIME.
    result_list : list[tuple[float, float]]
        The list to store the results.
    """
    print(f"Process {idx:03d} Started.")
    trg, _, _ = get_trg_sample(idx, dataset)

    # Initialize sampler.
    sampler = Sampler(
        trg, dataset.train, black_box.predict, dataset.categorical_names
    )

    lime_weights, scaler = rlime_lime.explain(trg, sampler, 5000)

    def get_rlime_weights() -> tuple[list[float], Rule] | None:
        try:
            csv_name = (
                EXAMPLES_DIR_NAME
                + f"/newlime-{idx:04d}-{int(tau*100):02d}.csv"
            )
            with Path(csv_name).open() as f:
                reader = csv.reader(f)
                weights = [float(x) for x in next(reader)]
                rule = tuple([int(x) for x in next(reader)])
                print(next(reader))
            return weights, Rule(rule)
        except FileNotFoundError:
            print(f"Process {idx:03d} Found No explanations.")
            return None

    if (result := get_rlime_weights()) is not None:
        rlime_weights, rule = result
        samples, labels = sampler.sample(10000, rule)
        print(rlime_weights, rule)

        lime_acc, rlime_acc = calc_accuracy(
            samples, labels, lime_weights, rlime_weights, scaler
        )

        # Print and save the results.
        print(f"Process {idx:03d} Finished.")
        print(f"  LIME  : {lime_acc:.4f}")
        print(f"  R-LIME: {rlime_acc:.4f}")
        result_list.append((lime_acc, rlime_acc))


if __name__ == "__main__":
    main()
