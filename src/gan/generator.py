# %% Packages

import tensorflow as tf
from tensorflow.keras import layers

# %% Function


def generator_model(config):

    input_shape = (config.generator.size_of_latent_vector,)

    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Dense(7 * 7 * 256, use_bias=False, input_shape=input_shape)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(config.generator.dropout_rate)(x)
    x = layers.Reshape((7, 7, 256))(x)

    x = layers.Conv2DTranspose(
        128, kernel_size=(5, 5), strides=(1, 1), padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(config.generator.dropout_rate)(x)

    x = layers.Conv2DTranspose(
        64, kernel_size=(5, 5), strides=(2, 2), padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(config.generator.dropout_rate)(x)

    outputs = layers.Conv2DTranspose(
        1, kernel_size=(5, 5), strides=(2, 2), padding="same", use_bias=False
    )(x)
    model = tf.keras.Model(inputs, outputs)

    return model
