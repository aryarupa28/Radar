import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np

class TimeGAN:
    def __init__(self, seq_len, feature_dim, hidden_dim=64):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Loss function
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def build_generator(self):
        model = models.Sequential([
            Input(shape=(self.seq_len, self.feature_dim)),
            layers.GRU(self.hidden_dim, return_sequences=True),
            layers.GRU(self.hidden_dim, return_sequences=True),
            layers.TimeDistributed(layers.Dense(self.feature_dim, activation='tanh'))
        ], name='Generator')
        return model

    def build_discriminator(self):
        model = models.Sequential([
            Input(shape=(self.seq_len, self.feature_dim)),
            layers.GRU(self.hidden_dim, return_sequences=True),
            layers.GRU(self.hidden_dim, return_sequences=False),
            layers.Dense(1, activation='sigmoid')
        ], name='Discriminator')
        return model

    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        noise = tf.random.normal([batch_size, self.seq_len, self.feature_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.generator(noise, training=True)

            real_output = self.discriminator(real_data, training=True)
            fake_output = self.discriminator(generated_data, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def generate_data(self, num_samples):
        noise = tf.random.normal([num_samples, self.seq_len, self.feature_dim])
        return self.generator(noise, training=False)
