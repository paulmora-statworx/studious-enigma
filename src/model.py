# %% Packages

import numpy as np
import tensorflow as tf

# %% Classes


class MNISTModelLoader:
    def __init__(self, config):
        self.config = config
        self.model = self.load_model()

    def load_model(self):

        preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
        input_shape = (self.config.model.target_size, self.config.model.target_size, 3)

        mobilenet_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                        include_top=False,
                                                           weights="imagenet")

        mobilenet_model.trainable = False

        output_layer = tf.keras.layers.Dense(self.config.model.number_of_classes)
        dropout_layer = tf.keras.layers.Dropout(0.2)

        inputs = tf.keras.Input(shape=input_shape)
        x = preprocessing_function(inputs)
        x = mobilenet_model(x, training=False)
        x = dropout_layer(x)
        outputs = output_layer(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.learning_rate),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=["accuracy"])

        print(model.summary())
        return model

