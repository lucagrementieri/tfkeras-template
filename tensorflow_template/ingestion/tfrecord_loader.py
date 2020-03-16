import os
from pathlib import Path

import tensorflow as tf

from .npy_codec import NpyCodec


class TFRecordLoader:
    def __init__(self, tfrecords_dir: str, feature_size: int):
        self.tfrecords_dir = Path(tfrecords_dir).expanduser()
        self.codec = NpyCodec(feature_size)

    def check_split(self, split: str) -> bool:
        return (self.tfrecords_dir / split).is_dir()

    def get_split_dataset(self, split: str, batch_size: int) -> tf.data.Dataset:
        pattern = str(self.tfrecords_dir / split / '*.tfrecords')
        paths = tf.data.Dataset.list_files(pattern, shuffle=(split == 'train'))
        dataset = tf.data.TFRecordDataset(filenames=paths)
        dataset = dataset.map(
            map_func=self.codec.decode, num_parallel_calls=os.cpu_count()
        )

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
