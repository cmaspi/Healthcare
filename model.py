from keras import layers, models
import tensorflow as tf


def get_model(input_size):
    inp = layers.Input(shape=(input_size, 1))
    x = layers.Conv1D(4, 5, activation='leaky_relu', strides=2)(inp)
    x = layers.Conv1D(8, 5, activation='leaky_relu', strides=2)(x)
    x = layers.Conv1D(16, 5, activation='leaky_relu', strides=2)(x)
    x = layers.Conv1D(32, 5, activation='leaky_relu', strides=2)(x)
    x = layers.Conv1D(64, 5, activation='leaky_relu', strides=2)(x)
    x = layers.Conv1D(128, 5, activation='leaky_relu', strides=2)(x)
    x1 = layers.GlobalAveragePooling1D()(x)
    x2 = layers.GlobalMaxPool1D()(x)
    x = tf.concat([x1, x2], axis=-1)
    x = layers.Dense(24, activation='leaky_relu')(x)
    x = layers.Dropout(rate=0.1)(x)
    x = layers.Dense(4, activation='leaky_relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inp, outputs=x, name='v1')

