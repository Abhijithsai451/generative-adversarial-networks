import logging
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D
from keras.losses import BinaryCrossentropy
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras_preprocessing.image import array_to_img
from keras.callbacks import Callback

from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)

ds = tfds.load('fashion_mnist', split='train')
# Getting the data out of the pipeline
# Using the numpy iterator fetches a batch of data everytime a next function is called.
data_iterator = ds.as_numpy_iterator()

# Visualization
fig, ax = plt.subplots(ncols=4, figsize=(10, 10))
for idx in range(4):
    batch = data_iterator.next()
    ax[idx].imshow(np.squeeze(batch['image']))
    ax[idx].title.set_text(batch['label'])


# plt.show()

# Scaling the data
def scale_images(data):
    image = data['image']
    return image / 255


# Data Preprocessing
ds = tfds.load('fashion_mnist', split='train')
logging.info("Performing the Preprocessing operations on the imported data")
ds = ds.map(scale_images)
ds = ds.cache()
ds = ds.shuffle(60000)
ds = ds.batch(128)
ds = ds.prefetch(64)
logging.info("Preprocessing the Data (Scaling, Caching, Shuffling, batching, prefetching) : Finished")

logging.info(ds.as_numpy_iterator().next().shape)

# Building the Neural Network
# This includes implementing
# 1. Generator
# 2. Descriminator

logging.info("Implementing the neural network")


def build_generator():
    model = Sequential()
    # Beginning of a generated image
    model.add(Dense(7 * 7 * 128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7, 7, 128)))

    # Upsampling the generated image to add more features
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    # Final Conv layer block for output
    model.add(Conv2D(1, 4, padding='same', activation='sigmoid'))

    return model


logging.info("Building a generator model")
generator = build_generator()
generator.summary()
"""img = test_model.predict(np.random.randn(4,128,1))
logging.info(img)"""


# Implementing a discriminator
def build_discriminator():
    model = Sequential()
    # First Conv Block
    model.add(Conv2D(32, 5, input_shape=(28, 28, 1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Second Conv Block
    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Third Conv Block
    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Fourth Conv Block
    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Flatten and Passing to the dense layer for final output
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    return model


logging.info("Building a discriminator model")
discriminator = build_discriminator()
discriminator.summary()
"""res = test_model.predict(img)
print(res)"""

# Training the Generator and Discriminator model and Back propagation
g_opt = Adam(learning_rate=0.0001)
d_opt = Adam(learning_rate=0.00001)
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()


class GAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, *kwargs)

        # Create attributes for gen and disc
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        super().compile(*args, **kwargs)
        logging.debug("compiling the model @ Debug")
        # creating attributes for losses and optimzers
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):
        # Get the data
        real_images = batch
        fake_images = self.generator(tf.random.normal((128, 128, 1)), training=False)

        # Train the discriminator
        with tf.GradientTape() as d_tape:
            # Pass the real and fake images to the discriminator model
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            # Create labels for real and fake images
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

            # Add some noise to the TRUE outputs
            noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)

            # Calculate loss = BINARY CROSS
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)

        # Apply backpropagation - nn Learn
        d_grad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))

        # Traiing the generator
        with tf.GradientTape() as g_tape:
            # Generate some new images
            gen_images = self.generator(tf.random.normal((128,128,1)), training= True)

            # Create the predicted labels from the generator
            predicted_labels = self.discriminator(gen_images,training=False)

            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        # Apply backprop
        g_grad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(g_grad, self.generator.trainable_variables))


        return {"d_loss":total_d_loss,"g_loss": total_g_loss}

# Callbacks
class ModelMonitor(Callback):
    def __init__(self,num_img =3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
    def on_epoch_end(self, epoch,logs =None):
        random_latent_vectors = tf.random.uniform((self.num_img,self.latent_dim,1))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(os.path.join('images',f'generated_img_{epoch}_{i}.png'))
monitor = ModelMonitor()

logging.info("creating the GAN model using generator and discriminator model")
fashion_gan = GAN(generator, discriminator)

# Compile the model
logging.info("compiling the model")
fashion_gan.compile(g_opt,d_opt,g_loss,d_loss)

# Training the model
hist = fashion_gan.fit(ds, epochs=20, callbacks= monitor)
