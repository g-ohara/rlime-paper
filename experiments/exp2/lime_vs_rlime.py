"""Compare LIME and R-LIME on the recidivism dataset."""

import csv
import multiprocessing
from functools import partial

import numpy as np
import pandas as pd  # type: ignore
from lime import lime_tabular  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore

from RLIME import mylime, newlime_base, newlime_tabular, newlime_utils
from RLIME.newlime_base import Arm
from RLIME.newlime_types import IntArray
from RLIME.sampler import Sampler


def main() -> None:
    """Main function."""

    # Load the dataset.
    dataset = newlime_utils.load_dataset(
        "recidivism", "RLIME/datasets/", balance=True
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
    sample: IntArray, labels: IntArray, lime_weights: list[float], arm: Arm
) -> tuple[float, float]:
    """Calculate the precision of LIME and R-LIME.

    Parameters
    ----------
    sample: newlime_utils.FloatArray
        The samples.
    labels: newlime_utils.IntArray
        The labels of the samples.
    lime_weights: newlime_utils.FloatArray
        The weights of LIME.
    surrogate_model: newlime_tabular.SurrogateModel
        The surrogate model of R-LIME.

    Returns
    -------
    tuple[float, float]
        The precision of LIME and R-LIME.
    """

    # Get predictions of LIME and R-LIME.
    lime_pred = np.dot(sample, lime_weights) > 0
    rlime_pred = arm.surrogate_model.predict_many(pd.DataFrame(sample))

    # Calculate the precision of LIME and R-LIME.
    lime_acc = np.sum(lime_pred == labels) / len(labels)
    rlime_acc = np.sum(rlime_pred == labels) / len(labels)

    return lime_acc, rlime_acc


def original_lime(
    dataset: newlime_tabular.Dataset,
    trg: IntArray,
    model: RandomForestClassifier,
    pred_label: int,
) -> list[float]:
    """Calculate the weights of LIME."""

    lime_explainer = lime_tabular.LimeTabularExplainer(
        dataset.train,
        feature_names=dataset.feature_names,
        class_names=dataset.class_names,
        discretize_continuous=False,
    )
    lime_exp = lime_explainer.explain_instance(
        trg, model.predict_proba, num_features=15, top_labels=1
    )
    weights = [0.0] * len(dataset.feature_names)
    for t in lime_exp.local_exp[pred_label]:
        weights[t[0]] = t[1] * (pred_label * 2 - 1)
    return weights


def compare_lime_and_newlime(
    idx: int,
    dataset: newlime_tabular.Dataset,
    black_box: RandomForestClassifier,
    tau: float,
    result_list: list[tuple[float, float]],
) -> None:
    """Compare LIME and R-LIME.

    Parameters
    ----------
    idx: int
        The index of the target instance.
    trg: newlime_utils.IntArray
        The target instance.
    dataset: newlime_utils.Dataset
        The dataset.
    black_box: RandomForestClassifier
        The black box model.
    hyper_param: newlime_tabular.HyperParam
        The hyperparameters of R-LIME.
    """

    print(f"Process {idx:03d} Started.")
    trg, _, _ = newlime_utils.get_trg_sample(idx, dataset)

    # Initialize sampler.
    sampler = Sampler(
        trg, dataset.train, black_box.predict, dataset.categorical_names
    )

    # lime_weights = mylime.explain(trg, sampler, 5000)
    lime_weights = original_lime(
        dataset, trg, black_box, black_box.predict(trg.reshape(1, -1))[0]
    )

    hyper_param = newlime_base.HyperParam(
        tau=tau,
        delta=0.05,
        epsilon=0.1,
        epsilon_stop=0.05,
        beam_size=10,
        batch_size=500,
        coverage_samples_num=10000,
        max_rule_length=None,
    )
    result = newlime_tabular.explain_instance(
        trg, dataset, black_box.predict, hyper_param
    )
    if result is None:
        print(f"Process {idx:03d} Found No explanations.")
        return

    _, arm = result

    # Sample from the rule.
    sample, labels = sampler.sample(10000, arm.rule)
    lime_acc, rlime_acc = calc_accuracy(sample, labels, lime_weights, arm)

    # Print and save the results.
    print(f"Process {idx:03d} Finished.")
    print(f"  LIME  : {lime_acc:.4f}")
    print(f"  R-LIME: {rlime_acc:.4f}")
    result_list.append((lime_acc, rlime_acc))


if __name__ == "__main__":
    main()
