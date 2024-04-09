"""Compare LIME and R-LIME on the recidivism dataset."""

import csv
import multiprocessing
from functools import partial

import numpy as np
import pandas as pd  # type: ignore
from river import preprocessing
from sklearn.ensemble import RandomForestClassifier  # type: ignore

from rlime import rlime_lime
from rlime.arm import Arm
from rlime.rlime import HyperParam, explain_instance
from rlime.rlime_types import Dataset, IntArray
from rlime.sampler import Sampler
from rlime.utils import get_trg_sample, load_dataset


def main() -> None:
    """Main function."""

    # Load the dataset.
    dataset = load_dataset(
        "recidivism", "rlime/examples/datasets/", balance=True
    )

    # Learn the black box model.
    black_box = RandomForestClassifier(n_estimators=100, n_jobs=1)
    black_box.fit(dataset.train, dataset.labels_train)

    # Get the target instances.
    sample_num = 100

    for tau in [0.70, 0.80, 0.90]:

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
    lime_scaler: preprocessing.StandardScaler,
    arm: Arm,
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

    Returns
    -------
    tuple[float, float]
        The precision of LIME and R-LIME.
    """

    # Get predictions of LIME and R-LIME.
    lime_pred = np.dot(lime_scaler.transform_many(samples), lime_weights) > 0
    rlime_pred = arm.surrogate_model.predict_many(pd.DataFrame(samples))

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

    def get_rlime_arm() -> Arm | None:
        hyper_param = HyperParam(
            tau=tau,
            delta=0.05,
            epsilon=0.1,
            epsilon_stop=0.05,
            beam_size=10,
            batch_size=500,
            coverage_samples_num=10000,
            max_rule_length=None,
        )
        result = explain_instance(trg, dataset, black_box.predict, hyper_param)
        if result is None:
            print(f"Process {idx:03d} Found No explanations.")
            return None

        _, arm = result
        return arm

    if (arm := get_rlime_arm()) is not None:

        # Sample from the rule.
        samples, labels = sampler.sample(10000, arm.rule)
        lime_acc, rlime_acc = calc_accuracy(
            samples, labels, lime_weights, scaler, arm
        )

        # Print and save the results.
        print(f"Process {idx:03d} Finished.")
        print(f"  LIME  : {lime_acc:.4f}")
        print(f"  R-LIME: {rlime_acc:.4f}")
        result_list.append((lime_acc, rlime_acc))


if __name__ == "__main__":
    main()
