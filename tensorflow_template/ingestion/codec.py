from abc import ABC, abstractmethod
from typing import Any, Sequence

import tensorflow as tf


class Codec(ABC):
    @staticmethod
    def _bytes_feature(value: Any) -> tf.train.Feature:
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value: float) -> tf.train.Feature:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value: int) -> tf.train.Feature:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_features(values: Sequence[Any]) -> tf.train.Feature:
        values = [
            value.numpy() if isinstance(value, type(tf.constant(0))) else value
            for value in values
        ]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    @staticmethod
    def _float_features(values: Sequence[float]) -> tf.train.Feature:
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    @staticmethod
    def _int64_features(values: Sequence[int]) -> tf.train.Feature:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    @abstractmethod
    def encode(self, example: Any) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def decode(self, record: tf.Tensor) -> Any:
        raise NotImplementedError
