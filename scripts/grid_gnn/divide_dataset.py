#!/usr/bin/env python3

from pathlib import Path
import pickle
import random

import numpy as np


if __name__ == "__main__":

    data_dir = Path("/scratch/work/oinonen1/density_db")
    sample_paths = sorted(list(data_dir.glob("*.npz")))

    random.seed(0)
    random.shuffle(sample_paths)

    divide = [0.8, 0.07, 0.13]
    n_sample = len(sample_paths)
    cumulative = [round(n_sample * n) for n in np.cumsum(divide)]
    cumulative = [0] + cumulative

    samples_train, samples_val, samples_test = [[p.name for p in sample_paths[cumulative[i] : cumulative[i + 1]]] for i in range(3)]

    assert n_sample == (len(samples_train) + len(samples_val) + len(samples_test))

    print(samples_test)
    print(len(samples_train))
    print(len(samples_val))
    print(len(samples_test))

    sample_dict = {
        "train": samples_train,
        "val": samples_val,
        "test": samples_test,
    }

    with open("sample_dict.pickle", "wb") as f:
        pickle.dump(sample_dict, f)
