import argparse
import json
import pathlib
import sys

import numpy as np
import tensorflow as tf

from tensorflow_template.ingestion import NpyDataset


def compute_statistics(root_dir: str) -> None:
    batch_size = 20

    # TODO: add transform if needed
    npy_dataset = NpyDataset(root_dir=root_dir, split='train')

    dataset = tf.data.Dataset.from_generator(
        npy_dataset.__iter__,
        {'features': tf.float32, 'target': tf.float32, 'filename': tf.string},
    )
    dataset = dataset.batch(batch_size)

    n = 0
    mean = None
    m2 = None

    for samples in dataset.as_numpy_iterator():
        features = samples['features']
        if mean is None or m2 is None:
            mean = np.zeros(features.shape[1], dtype=np.float32)
            m2 = np.zeros(features.shape[1], dtype=np.float32)
        # Here mean and standard deviation are computed via Chan et. al. algorithm
        # (see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
        count = features.shape[0]
        avg = features.mean(axis=0)
        var = features.var(axis=0)
        delta = avg - mean
        mean = (mean * n + features.sum(axis=0)) / (n + count)
        m2 += var * (count - 1) + np.power(delta, 2) * n * count / (n + count)
        n += count

    std = np.sqrt(m2 / (n - 1))
    statistics = {'mean': mean.tolist(), 'std': std.tolist()}
    print(statistics)

    with open(pathlib.Path(root_dir) / 'statistics.json', 'w') as f:
        json.dump(statistics, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Collect the mean and standard deviation for input data',
        usage='python3.7 dataset_statistics.py root-dir',
    )
    parser.add_argument(
        'root_dir', metavar='root-dir', type=str, help='Path to data root directory.'
    )
    args = parser.parse_args(sys.argv[1:])
    compute_statistics(args.root_dir)
