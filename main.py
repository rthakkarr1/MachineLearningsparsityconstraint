import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras.regularizers import l1
from tensorflow.keras.datasets import fashion_mnist
# This is the number of hidden nodes
encoding_dim = 64  # 64 floats -> compression of factor 12.25, assuming the input is 784 floats
# Load the Fashion MNIST dataset
(x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()
# Normalize pixel values between 0 and 1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# Reshape the images to a flat vector of size 784 (28x28)
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))
# This is our input image
input_img = keras.Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded1 = layers.Dense(128, activation='relu')(input_img)
encoded2 = layers.Dense(64, activation='relu')(encoded1)
encoded3 = layers.Dense(32, activation='relu')(encoded2)
# This model maps an input to its encoded representation
encoder = keras.Model(input_img, encoded3)
# "decoded" is the lossy reconstruction of the input
decoded1 = layers.Dense(64, activation='relu')(encoded3)
decoded2 = layers.Dense(128, activation='relu')(decoded1)
decoded = layers.Dense(784, activation='sigmoid')(decoded2)
# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
# Encode and decode some images
encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)
# Use Matplotlib (don't ask)
import matplotlib.pyplot as plt
n = 10  # How many images we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original image
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Display reconstructed image
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Display encoded representation
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(encoded_imgs[i].reshape(4, 8))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
