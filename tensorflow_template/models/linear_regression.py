import tensorflow as tf
from tensorflow.keras import layers


# TODO: update model

# TODO: remove the imperative version if not needed
class LinearRegression(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc = layers.Dense(1, kernel_initializer='he_normal')

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        prediction = tf.squeeze(self.fc(inputs), axis=-1)
        return prediction


# TODO: remove the functional version if not needed
def linear_regression(units: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(units,))
    dense = layers.Dense(1, kernel_initializer='he_normal')
    x = dense(inputs)
    outputs = tf.squeeze(x, axis=-1)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='linear_regression')
    return model
