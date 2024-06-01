from samples import (
    get_data_samples,
)
from training_validation_sets import get_training_validation_and_testing_sets

import pickle


def pickle_objects():
    data_samples = get_data_samples()

    with open("data_samples.pkl", "wb") as data_samples_pickle:
        pickle.dump(data_samples, data_samples_pickle)

    sets = get_training_validation_and_testing_sets(data_samples, 500)

    with open("sets.pkl", "wb") as sets_pickle:
        pickle.dump(sets, sets_pickle)

    sets_scaled = get_training_validation_and_testing_sets(
        data_samples, 500, scaled=True
    )

    with open("scaled_sets.pkl", "wb") as sets_scaled_pickle:
        pickle.dump(sets_scaled, sets_scaled_pickle)


if __name__ == "__main__":
    pickle_objects()
