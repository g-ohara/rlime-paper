"""This module generates the experiment data for the paper."""

import random
import sys

import numpy as np
import pandas as pd
from river import compose
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from prelearning import get_covered_samples, surrogate_precision

sys.path.append("NewLIME")
# pylint: disable=import-error,wrong-import-position
import newlime_tabular  # pyright: ignore[reportMissingImports] # noqa: E402
import newlime_utils  # pyright: ignore[reportMissingImports] # noqa: E402


def main() -> None:
    """Main function."""

    # Load the dataset.
    balanced = newlime_utils.load_dataset(
        "recidivism", "NewLIME/anchor-experiments/datasets/", balance=True
    )
    imbalanced = newlime_utils.get_imbalanced_dataset(balanced, 0.10)

    # Set the hyper-parameters for NewLIME.
    hyper_param = newlime_tabular.HyperParam(
        delta=0.05,
        epsilon=0.05,
        epsilon_stop=0.05,
        beam_size=10,
        batch_size=1000,
        desired_confidence=0.70,
        coverage_samples_num=10000,
        max_rule_length=None,
    )

    for dataset in [imbalanced, balanced]:
        # Learn the black box model.
        black_box = RandomForestClassifier(n_estimators=100, n_jobs=5)
        black_box.fit(dataset.train, dataset.labels_train)

        # Print the number of labels in the dataset.
        print_labels_count(dataset, black_box)

        # Print the precision, recall, and coverage of the surrogate model
        # returned by NewLIME under the rule.
        idx_list = random.sample(range(len(dataset.test)), 10)
        trgs = dataset.test[idx_list]
        explainer = newlime_tabular.NewLimeTabularExplainer(
            dataset.class_names,
            dataset.feature_names,
            dataset.train,
            dataset.categorical_names,
        )
        sum_vals = [0.0, 0.0, 0.0]
        names = ["Precision", "Recall", "Coverage"]
        for trg in tqdm(trgs):
            vals = get_prec_and_recall(
                explainer, hyper_param, black_box, dataset, trg
            )
            sum_vals = [sum_val + val for sum_val, val in zip(sum_vals, vals)]

        for name, sum_val in zip(names, sum_vals):
            print(f"{name}: {sum_val / float(len(idx_list))}")


def get_prec_and_recall(
    explainer: newlime_tabular.NewLimeTabularExplainer,
    hyper_param: newlime_tabular.HyperParam,
    black_box: RandomForestClassifier,
    dataset: newlime_utils.Dataset,
    trg: newlime_tabular.Samples,
) -> tuple[float, float, float]:
    """Get the precision and recall of the surrogate model returned by NewLIME
    under the rule.
    """

    result = explainer.my_explain_instance(trg, black_box.predict, hyper_param)

    while True:
        if result is None:
            continue
        anchor_exp, surrogate_model = result
        if surrogate_model is None:
            continue

        prec = surrogate_precision(
            anchor_exp.features(),
            trg,
            dataset.test,
            black_box,
            surrogate_model,
        )
        recall = surrogate_recall(
            anchor_exp.features(), trg, dataset, black_box, surrogate_model
        )
        return prec, recall, anchor_exp.coverage()


def print_labels_count(
    dataset: newlime_utils.Dataset, black_box: RandomForestClassifier
) -> None:
    """Print the number of labels in the dataset and the number of labels
    predicted by the black box model.
    """

    print("Label 0  : ", np.sum(dataset.labels == 0))
    print("Label 1  : ", np.sum(dataset.labels == 1))

    for i in [0, 1]:
        cnt = (
            np.sum(black_box.predict(dataset.train) == i)
            + np.sum(black_box.predict(dataset.test) == i)
            + np.sum(black_box.predict(dataset.validation) == i)
        )
        print(f"Predict {i}  : ", cnt)


def surrogate_recall(
    features: list[int],
    trg: newlime_tabular.Samples,
    dataset: newlime_utils.Dataset,
    black_box: RandomForestClassifier,
    surrogate_model: compose.Pipeline,
) -> float:
    """Calculate the recall of the surrogate model returned by NewLIME
    under the rule.
    """

    exs_arr = get_covered_samples(features, trg, dataset.test)
    labels = black_box.predict(exs_arr)
    preds = surrogate_model.predict_many(pd.DataFrame(exs_arr))
    true_pos = np.sum([la == p == 1 for la, p in zip(labels, preds)])
    return float(true_pos / np.sum(labels == 1))


if __name__ == "__main__":
    main()
