
# %% Packages

import tensorflow as tf

# %%

class GanTrainer:

    def __init__(self, generator, discriminator, config):
        self.generator = generator
        self.discriminator = discriminator
        self.config = config

        self.create_loss_function()
        self.create_optimizers()

        self.train_gan()

    def create_optimizers(self):
        self.generator_optimizer = tf.keras.optimizers.Adam(self.config.trainer.learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.config.trainer.learning_rate)

    def create_loss_function(self):
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_iteration(self, images):
        noise = tf.random.normal([self.config.trainer.batch_size, self.config.trainer.size_of_latent_vector])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)


