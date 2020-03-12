import tensorflow as tf


def linear_regression(units: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(units,))
    dense = tf.keras.layers.Dense(1, kernel_initializer='he_normal')
    x = dense(inputs)
    outputs = tf.squeeze(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='linear_regression')
    return model
