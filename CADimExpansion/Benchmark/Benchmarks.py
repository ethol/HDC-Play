import numpy as np
from sklearn import datasets
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_binary_mnist():
    mnist = tf.keras.datasets.mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    data = np.concatenate((X_train, X_test), axis=0)
    labels = np.concatenate((y_train, y_test), axis=0)

    data_bool = np.round(data / 255).astype("uint8")
    data_bool = data_bool.reshape(data_bool.shape[0], -1)

    # holdout 33 % of the data for later.
    data_bool, data_holdout, labels, labels_holdout = train_test_split(
        data_bool, labels, test_size=0.33, random_state=42)

    return data_bool, labels


def load_binary_digits():
    digits = datasets.load_digits()

    targets = digits.target
    data = digits.data
    high = np.max(data)

    data_bool = np.round(data / high).astype("uint8")
    return data_bool, targets
