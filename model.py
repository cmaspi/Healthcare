# from tensorflow.keras import layers, models
from keras import layers, models
import tensorflow as tf

# def get_model():
#     model = models.Sequential()
#     model_layers = [
#         layers.Input(shape=(15_000, 1)),
#         layers.Conv1D(4, 5, activation='relu', strides=2),
#         layers.Conv1D(8, 5, activation='relu', strides=2),
#         layers.BatchNormalization(),
#         layers.Conv1D(16, 5, activation='relu', strides=2),
#         layers.Conv1D(64, 5, activation='relu', strides=2),
#         layers.Conv1D(32, 5, activation='relu', strides=2),
#         layers.BatchNormalization(),
#         layers.Conv1D(16, 5, activation='relu', strides=2),
#         layers.Conv1D(8, 5, activation='relu', strides=2),
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.BatchNormalization(),
#         layers.Dense(8, activation='relu'),
#         layers.Dense(1, activation='sigmoid')
#     ]
#     for ml in model_layers:
#         model.add(ml)
#     return model

# def get_model(input_size):
#     model = models.Sequential()
#     model_layers = [
#         layers.Input(shape=(input_size, 1)),
#         layers.Conv1D(4, 7, activation='leaky_relu', strides=3),
#         layers.BatchNormalization(),
#         # layers.Conv1D(8, 7, activation='leaky_relu', strides=3),
#         layers.Conv1D(16, 7, activation='leaky_relu', strides=3),
#         # layers.Conv1D(32, 5, activation='leaky_relu', strides=3),
#         layers.Conv1D(32, 7, activation='leaky_relu', strides=3),
#         layers.Conv1D(16, 7, activation='leaky_relu', strides=3),
#         layers.Dropout(rate=0.1),
#         layers.Conv1D(4, 7, activation='leaky_relu', strides=3),
#         layers.Flatten(),
#         layers.Dense(8, activation='leaky_relu'),
#         layers.BatchNormalization(),
#         layers.Dropout(rate=0.1),
#         layers.Dense(1, activation='sigmoid')
#     ]
#     for ml in model_layers:
#         model.add(ml)
#     return model


# def get_model(input_size):
#     model = models.Sequential()
#     model_layers = [
#         layers.Input(shape=(input_size, 1)),
#         layers.BatchNormalization(),
#         layers.Conv1D(4, 3, activation='leaky_relu'),
#         layers.Conv1D(8, 3, activation='leaky_relu'),
#         layers.Conv1D(16, 3, activation='leaky_relu'),
#         layers.BatchNormalization(),
#         layers.Conv1D(32, 3, activation='leaky_relu'),
#         layers.Conv1D(64, 3, activation='leaky_relu'),
#         layers.Conv1D(128, 3, activation='leaky_relu'),
#         layers.GlobalAveragePooling1D(),
#         layers.Dense(24, activation='leaky_relu'),
#         layers.Dropout(rate=0.1),
#         layers.Dense(4, activation='leaky_relu'),
#         layers.Dense(1, activation='sigmoid')
#     ]
#     for ml in model_layers:
#         model.add(ml)
#     return model


def get_model(input_size):
    inp = layers.Input(shape=(input_size, 1))
    x = layers.Conv1D(4, 3, activation='leaky_relu')(inp)
    # x = layers.BatchNormalization()(x)
    x = layers.Conv1D(8, 3, activation='leaky_relu')(x)
    x = layers.Conv1D(16, 3, activation='leaky_relu')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, 3, activation='leaky_relu')(x)
    x = layers.Conv1D(64, 3, activation='leaky_relu')(x)
    x = layers.Conv1D(128, 3, activation='leaky_relu')(x)
    x1 = layers.GlobalAveragePooling1D()(x)
    x2 = layers.GlobalMaxPool1D()(x)
    x = tf.concat([x1, x2], axis=-1)
    x = layers.Dense(24, activation='leaky_relu')(x)
    x = layers.Dropout(rate=0.1)(x)
    x = layers.Dense(4, activation='leaky_relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inp, outputs=x)
