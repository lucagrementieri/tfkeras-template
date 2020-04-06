from pathlib import Path

import numpy as np
import tensorflow as tf


# TODO: update class to manage your data
class NpzLoader:
    def __init__(self, npz_dir: str):
        self.npz_dir = Path(npz_dir).expanduser()

    def check_split(self, split: str) -> bool:
        return (self.npz_dir / split).is_dir()

    def get_split_dataset(self, split: str, batch_size: int) -> tf.data.Dataset:
        split_dir = self.npz_dir / split
        features = []
        targets = []
        for path in split_dir.glob('*.npz'):
            with np.load(str(path)) as sample:
                features.append(sample['features'])
                targets.append(sample['target'])
        features = np.stack(features)
        targets = np.stack(targets)
        dataset = tf.data.Dataset.from_tensor_slices((features, targets))

        if split == 'train':
            # TODO update parameter
            shuffle_size = int(1e6)
            dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True)

        # TODO update parameter
        prefetch_size = 128
        dataset = dataset.batch(batch_size=batch_size).prefetch(
            buffer_size=prefetch_size
        )
        return dataset
