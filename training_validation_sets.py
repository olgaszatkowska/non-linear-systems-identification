import numpy as np
from sklearn.model_selection import train_test_split
from samples import DataSample
from numpy.typing import NDArray


def get_training_validation_and_testing_sets(
    data_samples: list[DataSample], window_size: int
) -> NDArray:
    X, y = [], []

    for i in range(0, len(data_samples), window_size):
        window = data_samples[i : i + window_size]
        X_window = np.array([sample.X for sample in window])
        y_window = np.array(window[-1].y)
        X.append(X_window)
        y.append(y_window)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        np.asarray(X, dtype="object"),
        np.asarray(y, dtype="object"),
        test_size=0.2,
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )

    return [
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ]