from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras import layers


class Linear(layers.Layer):
    def __init__(self, units: int, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape: tf.TensorShape) -> None:
        # TODO REAL: check type of input_shape
        print('Type of input_shape', type(input_shape))
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units), initializer='he_normal', trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer='zeros', trainable=True
        )

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({'units': self.units})
        return config


class LinearRegression(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # TODO: update model layers
        self.fc = Linear(1)

    # TODO: update call function
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        prediction = tf.squeeze(self.fc(inputs))
        return prediction