import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Define directories
data_dir_real = 'data/real'
data_dir_cartoon = 'data/cartoon'

# Hyperparameters
batch_size = 16
img_height = 128
img_width = 128
epochs = 50
buffer_size = 200

# Data Preprocessing
train_data_gen_real = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.2
)

train_data_gen_cartoon = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.2
)

train_real = train_data_gen_real.flow_from_directory(
    data_dir_real,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=None
)

train_cartoon = train_data_gen_cartoon.flow_from_directory(
    data_dir_cartoon,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=None
)

# Build Generator Model (U-Net based)
def build_generator():
    inputs = layers.Input(shape=[img_height, img_width, 3])
    
    # Encoder
    x = layers.Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    # Bottleneck
    x = layers.Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    # Decoder
    x = layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2DTranspose(3, (4, 4), strides=2, padding='same')(x)
    outputs = layers.Activation('tanh')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Build Discriminator Model
def build_discriminator():
    inputs = layers.Input(shape=[img_height, img_width, 3])
    
    x = layers.Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    outputs = layers.Activation('sigmoid')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Create Generators and Discriminators
generator_g = build_generator()  # Real -> Cartoon
generator_f = build_generator()  # Cartoon -> Real
discriminator_x = build_discriminator()  # Real or Fake Discriminator
discriminator_y = build_discriminator()  # Cartoon or Fake Discriminator

# Loss Functions
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Define Optimizers
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# CycleGAN Training Step
@tf.function
def train_step(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        fake_y = generator_g(real_x, training=True)
        # Generator F translates Y -> X
        fake_x = generator_f(real_y, training=True)
        
        # Discriminator X and Y
        disc_real_x = discriminator_x(real_x, training=True)
        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)
        
        # Generator loss
        gen_g_loss = loss_obj(tf.ones_like(disc_fake_y), disc_fake_y)
        gen_f_loss = loss_obj(tf.ones_like(disc_fake_x), disc_fake_x)
        
        # Discriminator loss
        disc_x_loss = (loss_obj(tf.ones_like(disc_real_x), disc_real_x) +
                       loss_obj(tf.zeros_like(disc_fake_x), disc_fake_x)) * 0.5
        disc_y_loss = (loss_obj(tf.ones_like(disc_real_y), disc_real_y) +
                       loss_obj(tf.zeros_like(disc_fake_y), disc_fake_y)) * 0.5
    
    # Calculate the gradients
    generator_g_gradients = tape.gradient(gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(gen_f_loss, generator_f.trainable_variables)
    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)
    
    # Apply the gradients
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

# Training Loop
for epoch in range(epochs):
    for real_image, cartoon_image in zip(train_real, train_cartoon):
        train_step(real_image, cartoon_image)
    print(f"Epoch {epoch+1}/{epochs} completed")

# Save the model
generator_g.save('cartoon_generator.h5')

# Test the Model
def display_images(model, test_input):
    prediction = model(test_input, training=False)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    plt.imshow((test_input[0] + 1) / 2)
    plt.subplot(1, 2, 2)
    plt.title('Cartoonified Image')
    plt.imshow((prediction[0] + 1) / 2)
    plt.show()

# Load and Test
sample_image = next(iter(train_real))
display_images(generator_g, sample_image)
