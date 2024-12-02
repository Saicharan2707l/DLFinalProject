import os
import glob
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Set constants
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 1
EPOCHS = 10
LAMBDA = 10  # Weight for cycle consistency loss

# Paths to the datasets
real_images_path = 'data/real'     # Directory containing real images
cartoon_images_path = 'data/cartoon'  # Directory containing cartoon images

# Function to load and preprocess images
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  # Ensure images have 3 channels
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = (img / 127.5) - 1  # Normalize images to [-1, 1]
    return img

# Function to create a TensorFlow dataset
def load_dataset(image_paths):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Get list of image file paths
real_image_paths = glob.glob(os.path.join(real_images_path, '*.jpg')) + \
                   glob.glob(os.path.join(real_images_path, '*.png'))

cartoon_image_paths = glob.glob(os.path.join(cartoon_images_path, '*.jpg')) + \
                      glob.glob(os.path.join(cartoon_images_path, '*.png'))

# Load datasets
real_dataset = load_dataset(real_image_paths)
cartoon_dataset = load_dataset(cartoon_image_paths)

# Custom InstanceNormalization layer
class InstanceNormalization(layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True)
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (inputs - mean) * inv
        return self.gamma * normalized + self.beta

# Function to define a ResNet block for the generator
def resnet_block(input_layer, filters):
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(input_layer)
    x = InstanceNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
    x = InstanceNormalization()(x)
    x = layers.add([x, input_layer])
    return x

# Function to build the generator model
def build_generator():
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    # Initial convolutional layer
    x = layers.Conv2D(64, kernel_size=7, strides=1, padding='same')(inputs)
    x = InstanceNormalization()(x)
    x = layers.Activation('relu')(x)
    # Downsampling layers
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = InstanceNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = InstanceNormalization()(x)
    x = layers.Activation('relu')(x)
    # ResNet blocks
    for _ in range(9):
        x = resnet_block(x, 256)
    # Upsampling layers
    x = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
    x = InstanceNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    x = InstanceNormalization()(x)
    x = layers.Activation('relu')(x)
    # Output layer
    x = layers.Conv2D(3, kernel_size=7, strides=1, padding='same')(x)
    outputs = layers.Activation('tanh')(x)
    return keras.Model(inputs, outputs)

# Function to build the discriminator model (PatchGAN)
def build_discriminator():
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(512, kernel_size=4, strides=1, padding='same')(x)
    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(x)
    return keras.Model(inputs, x)

# Instantiate the models
generator_g = build_generator()  # Translates real images to cartoons (G: X -> Y)
generator_f = build_generator()  # Translates cartoons to real images (F: Y -> X)

discriminator_x = build_discriminator()  # Discriminates real images and generated images (DX)
discriminator_y = build_discriminator()  # Discriminates cartoons and generated cartoons (DY)

# Define loss functions
loss_obj = keras.losses.MeanSquaredError()

def generator_loss(generated_output):
    return loss_obj(tf.ones_like(generated_output), generated_output)

def discriminator_loss(real_output, generated_output):
    real_loss = loss_obj(tf.ones_like(real_output), real_output)
    generated_loss = loss_obj(tf.zeros_like(generated_output), generated_output)
    total_loss = (real_loss + generated_loss) * 0.5
    return total_loss

def calc_cycle_loss(real_image, cycled_image):
    loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss

def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

# Define optimizers
generator_g_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_x_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)

# Remove @tf.function decorator for debugging
# @tf.function
def train_step(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        fake_y = generator_g(real_x, training=True)
        # Generator F translates Y -> X
        fake_x = generator_f(real_y, training=True)

        # Discriminator output
        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)
        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # Cycle consistency
        cycled_x = generator_f(fake_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # Identity mapping
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        # Calculate losses
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate gradients
    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)
    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    # Apply gradients
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

# Training function
def train(epochs):
    for epoch in range(epochs):
        start = time.time()
        print(f"Epoch {epoch + 1}/{epochs}")
        n = 0
        for real_x_batch, real_y_batch in tf.data.Dataset.zip((real_dataset, cartoon_dataset)):
            train_step(real_x_batch, real_y_batch)
            if n % 100 == 0:
                print('.', end='')
            n += 1
        print(f' Time taken: {time.time() - start:.2f} sec')

        # Generate sample images for visualization
        sample_real = next(iter(real_dataset))
        sample_fake_y = generator_g(sample_real, training=False)
        sample_cycled_x = generator_f(sample_fake_y, training=False)

        plt.figure(figsize=(12, 12))
        display_list = [sample_real[0], sample_fake_y[0], sample_cycled_x[0]]
        title = ['Real Image', 'Cartoonified Image', 'Reconstructed Image']
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # Rescale images from [-1, 1] to [0, 1]
            plt.imshow((display_list[i] * 0.5 + 0.5).numpy())
            plt.axis('off')
        plt.show()

# Start training
train(EPOCHS)
