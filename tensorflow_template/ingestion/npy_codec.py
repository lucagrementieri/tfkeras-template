import gzip
from typing import Union, Tuple, Dict

import numpy as np
import tensorflow as tf

from .codec import Codec


class NpyCodec(Codec):
    def __init__(self, features_size: int):
        self.features_size = features_size

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(features_size={self.features_size})'

    def encode(self, example: Dict[str, Union[np.array, float, str]]) -> bytes:
        # TODO update example structure
        feature = {
            'features': Codec._float_features(example['features'].tolist()),
            'target': Codec._float_feature(example['target']),
            'filename': Codec._bytes_feature(example['filename'].encode('utf-8')),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        record = example_proto.SerializeToString()
        record = gzip.compress(record)
        return record

    def decode(self, record: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        record = tf.io.decode_compressed(record, 'GZIP')
        # TODO update record structure
        record_description = {
            'features': tf.io.FixedLenFeature(self.features_size, tf.float32),
            'target': tf.io.FixedLenFeature(1, tf.float32),
            'filename': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(record, record_description)

        # TODO REAL: check if all these operations are needed
        # TODO REAL: check if decoding works
        features = tf.io.decode_raw(example['features'], tf.float32)
        features = features.set_shape(self.features_size)
        features = tf.cast(features, tf.float32)

        target = tf.io.decode_raw(example['target'], tf.float32)
        target = target.set_shape(1)
        target = tf.cast(target, tf.float32)

        return features, target
