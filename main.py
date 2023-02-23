import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist

# Load the Fashion-MNIST dataset
(x_train, _), (x_test, _) = fashion_mnist.load_data()

# Normalize the input data to range between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Define the encoder model
input_shape = (28, 28)
hidden_size = 64

encoder_input = keras.Input(shape=input_shape, name="input")
flatten_layer = layers.Flatten()(encoder_input)
hidden_layer = layers.Dense(hidden_size, activation="relu", activity_regularizer=keras.regularizers.l1(10e-5))(flatten_layer)
encoder_output = layers.Dense(hidden_size, activation="sigmoid")(hidden_layer)

encoder_model = keras.Model(encoder_input, encoder_output, name="encoder")

# Define the decoder model
decoder_input = keras.Input(shape=hidden_size, name="encoded")
hidden_layer_2 = layers.Dense(hidden_size, activation="relu")(decoder_input)
flatten_layer_2 = layers.Dense(28 * 28, activation="sigmoid")(hidden_layer_2)
decoder_output = layers.Reshape(input_shape)(flatten_layer_2)

decoder_model = keras.Model(decoder_input, decoder_output, name="decoder")

# Define the autoencoder model
autoencoder_input = keras.Input(shape=input_shape, name="input")
encoded = encoder_model(autoencoder_input)
decoded = decoder_model(encoded)
autoencoder = keras.Model(autoencoder_input, decoded, name="autoencoder")

# Compile the autoencoder model
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# Train the autoencoder model
autoencoder.fit(x_train, x_train, epochs=20, batch_size=128, shuffle=True, validation_data=(x_test, x_test))

# Use the trained encoder and decoder models to generate encoded representations and reconstructed images
encoded_imgs = encoder_model.predict(x_test)
decoded_imgs = decoder_model.predict(encoded_imgs)

# Plot some example images and their corresponding reconstructed images and encoded representations
n = 10  # Number of images to visualize
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original image
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Encoded representation
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(encoded_imgs[i].reshape((8, 8)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstructed image
    ax = plt.subplot(3, n, i + 2 * n + 1)
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
