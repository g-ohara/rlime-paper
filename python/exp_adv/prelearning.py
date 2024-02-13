"""This module generates the experiment data for the paper. The data is used to
measure the gap between the precision returned by NewLIME and the true
precision under the rule. The gap is calculated by the following formula:

        gap = returned_precision - true_precision

The gap is calculated for each target sample. The mean and the standard
deviation of the gaps are calculated for all the target samples. The mean and
the standard deviation are used to evaluate the performance of NewLIME. The
smaller the mean and the standard deviation are, the better the performance of
NewLIME is.
"""

import multiprocessing
import sys
from typing import Any

import numpy as np
import pandas as pd
from river import compose
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

sys.path.append("../NewLIME")


# pylint: disable=import-error,wrong-import-position
import newlime_tabular  # pyright: ignore[reportMissingImports] # noqa: E402
import newlime_utils  # pyright: ignore[reportMissingImports] # noqa: E402


def main() -> None:
    """Main function."""

    # Set the hyperparameters.
    hyper_param = newlime_tabular.HyperParam(
        tau=0.80,
        delta=0.05,
        epsilon=0.05,
        epsilon_stop=0.05,
        beam_size=10,
        batch_size=1000,
        init_sample_num=0,
        coverage_samples_num=1000,
        max_rule_length=None,
    )

    # Set the number of processes.
    process_num = 16

    # Set the number of target samples for each process.
    samples_num = 200

    # Load the dataset.
    dataset = newlime_utils.load_dataset(
        "recidivism", "../NewLIME/anchor-experiments/datasets/", balance=True
    )

    # Learn the black box model.
    black_box = RandomForestClassifier(n_estimators=100, n_jobs=5)
    black_box.fit(dataset.train, dataset.labels_train)

    processes = []
    result_list = multiprocessing.Manager().list()

    # Measure the gap between the precision returned by NewLIME and the true
    # precision. Caluculating the gap is time-consuming, so we use multiple
    # processes to accelerate the calculation.
    for i in range(process_num):
        np.random.seed(i)
        process = multiprocessing.Process(
            target=measure_gap,
            args=(samples_num, dataset, black_box, hyper_param, result_list),
        )
        process.start()
        processes.append(process)

    # Wait for all the processes to finish.
    for process in processes:
        process.join()

    # Calculate the mean and the standard deviation of the gaps.
    names = ["Returned Prec", "True Prec", "Gap"]
    for name, vals in zip(names, zip(*sum(result_list, []))):
        print(name)
        print(" Mean: ", np.mean(vals))
        print(" Std : ", np.std(vals))


def get_samples(
    test_set: np.ndarray,
    samples_num: int | None = None,
) -> np.ndarray:
    """Get samples randomly.

    Parameters
    ----------
    test_set
        The test set.
    samples_num
        The number of samples to be returned. If None, all the samples in the
        test set will be returned.

    Returns
    -------
    np.ndarray
        The samples covered by the rule.
    """

    if samples_num is None:
        samples_num = test_set.shape[0]

    # Get the samples randomly from the test set.
    return test_set[np.random.randint(test_set.shape[0], size=samples_num), :]


def get_covered_samples(
    features: list[int],
    trg: np.ndarray,
    test_set: np.ndarray,
    samples_num: int | None = None,
) -> np.ndarray:
    """Get samples uncovered by the rule randomly. The features of the samples
    will be replaced by the ones covered by the rule.

    Parameters
    ----------
    features
        The features covered by the rule.
    trg
        The target sample.
    test_set
        The test set.
    samples_num
        The number of samples to be returned. If None, all the samples in the
        test set will be returned.

    Returns
    -------
    np.ndarray
        The samples covered by the rule.
    """

    # Get the samples randomly from the test set.
    samples = get_samples(test_set, samples_num)

    # Replace the features with the ones covered by the rule.
    samples[:, features] = trg[features]
    return samples


def surrogate_precision(
    features: list[int],
    trg: np.ndarray,
    test_set: np.ndarray,
    black_box: RandomForestClassifier,
    surrogate_model: compose.Pipeline,
) -> float:
    """Calculate the precision of the surrogate model returned by NewLIME
    under the rule.
    """

    samples = get_covered_samples(features, trg, test_set)
    labels = black_box.predict(samples)
    pred_labels = surrogate_model.predict_many(pd.DataFrame(samples))
    return float(np.sum(labels == pred_labels) / len(labels))


def measure_gap(
    samples_num: int,
    dataset: newlime_utils.Dataset,
    black_box: RandomForestClassifier,
    hyper_param: newlime_tabular.HyperParam,
    result_list: list[Any],
) -> None:
    """Measure the gap between the precision returned by NewLIME and the true
    precision under the rule.
    """

    trgs = get_samples(dataset.test, samples_num)
    gaps_prec = []
    for trg in tqdm(trgs):
        result = newlime_tabular.explain_instance(
            trg, dataset, black_box.predict, hyper_param
        )

        if result is None:
            break
        anchor_exp, surrogate_model = result
        if surrogate_model is None:
            break

        # Calculate the gap between the precision returned by NewLIME and
        # the true precision.
        returned_prec = anchor_exp.precision()
        true_prec = surrogate_precision(
            anchor_exp.features(),
            trg,
            dataset.data,
            black_box,
            surrogate_model,
        )
        gaps_prec.append((returned_prec, true_prec, returned_prec - true_prec))
        print(returned_prec, true_prec)

    result_list.append(gaps_prec)


if __name__ == "__main__":
    main()
