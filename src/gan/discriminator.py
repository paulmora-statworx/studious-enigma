# %% Packages

import tensorflow as tf
from tensorflow.keras import layers

# %% Function

def discriminator_model(config):

    input_shape = (config.discriminator.image_size, config.discriminator.image_size, 1)

    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=input_shape)(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(config.discriminator.dropout_rate)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(config.discriminator.dropout_rate)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return x
