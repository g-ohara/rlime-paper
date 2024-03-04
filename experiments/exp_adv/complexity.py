"""Calculate the complexity of queries for the black-box classifiers
in R-LIME."""

import sys
from typing import Any

from sklearn.ensemble import RandomForestClassifier

sys.path.append("RLIME")

from RLIME import newlime_base, newlime_tabular, newlime_utils
from RLIME.newlime_types import IntArray


class BlackBoxClassifier(RandomForestClassifier):  # type: ignore
    """A black-box classifier for R-LIME."""

    def __init__(self, *args: str, **kwargs: int) -> None:
        """Initialize the black-box classifier."""

        super().__init__(*args, **kwargs)
        self.predict_count = 0

    def predict(self, X: IntArray) -> Any:
        """Predict the labels of the input instances."""

        self.predict_count += X.shape[0]
        print(f"Predict count: {self.predict_count}")
        return super().predict(X)


def main() -> None:
    """Main function."""

    # Load the dataset.
    dataset = newlime_utils.load_dataset(
        "recidivism", "datasets/", balance=True
    )

    # Learn the black box model.
    black_box = BlackBoxClassifier(n_estimators=100, n_jobs=1)
    black_box.fit(dataset.train, dataset.labels_train)

    idx = 0
    trg, _, _ = newlime_utils.get_trg_sample(idx, dataset)
    hyper_param = newlime_base.HyperParam()
    result = newlime_tabular.explain_instance(
        trg, dataset, black_box, hyper_param
    )

    if result is None:
        print("No explanation found.")
        return
    names, arm = result
    print(f"Names: {names}")
    print(f"ARM: {arm}")
