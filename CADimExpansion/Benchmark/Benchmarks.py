import numpy as np
import pandas as pd
from sklearn import datasets
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os


def load_ucr_benchmark(ID):
    script_directory = os.path.dirname(__file__)
    file_path = os.path.join(script_directory, 'UCRArchive', 'DataSummary.csv')
    dataset = pd.read_csv(file_path, index_col=0)
    # print(dataset)
    # name = dataset.loc[[number]]["Name"]
    benchmark_name = dataset.loc[ID, 'Name']
    print(benchmark_name)

    data_train = np.loadtxt(os.path.join(script_directory, 'UCRArchive', benchmark_name, f'{benchmark_name}_TRAIN.tsv'))
    data_test = np.loadtxt(os.path.join(script_directory, 'UCRArchive', benchmark_name, f'{benchmark_name}_TEST.tsv'))

    data_train = np.nan_to_num(data_train, nan=0.0)
    data_test = np.nan_to_num(data_test, nan=0.0)

    return data_train, data_test


def find_length_of_UCR_benchmark(ID):
    script_directory = os.path.dirname(__file__)
    file_path = os.path.join(script_directory, 'UCRArchive', 'DataSummary.csv')
    dataset = pd.read_csv(file_path, index_col=0)
    # print(dataset)
    # name = dataset.loc[[number]]["Name"]
    benchmark_length = dataset.loc[ID, 'Length']
    if benchmark_length.isdigit():
        return int(benchmark_length)
    data_train, data_test = load_ucr_benchmark(ID)
    return data_train.shape[1] - 1


def load_binary_mnist_original_split():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    data_bool_train = np.round(x_train / 255).astype("uint8")
    x_train = data_bool_train.reshape(data_bool_train.shape[0], -1)

    data_bool_test = np.round(x_test / 255).astype("uint8")
    x_test = data_bool_test.reshape(data_bool_test.shape[0], -1)
    return x_train, y_train, x_test, y_test


def load_binary_mnist_rounded():
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


def load_binary_mnist_similarity_preserved():
    dim = 32
    mnist = tf.keras.datasets.mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    data = np.concatenate((X_train, X_test), axis=0)
    labels = np.concatenate((y_train, y_test), axis=0)

    data = data.reshape(data.shape[0], -1)

    # holdout 33 % of the data for later.
    data, data_holdout, labels, labels_holdout = train_test_split(
        data, labels, test_size=0.33, random_state=42)

    ids = np.random.randint(0, 2, size=((28 * 28), dim), dtype="uint8")

    vec_0 = np.random.randint(0, 2, size=dim, dtype="uint8")
    vec_1 = np.random.randint(0, 2, size=dim, dtype="uint8")

    data_expanded = []
    j = 0
    for d in data:
        j += 1
        new_vec = []

        for i in range(len(d)):
            if d[i] == 0:
                vec = vec_0
            else:
                vec = np.hstack((vec_0[int((d[i] / 256) * dim):], vec_1[:int((d[i] / 256) * dim)]))
            new_vec.append(np.bitwise_xor(vec, ids[i]))
        new_vec = np.array(new_vec)
        data_expanded.append(new_vec.flatten())
        if j % 1000 == 0:
            print(j / len(data))

    data_expanded = np.array(data_expanded)
    return data_expanded, labels


def load_binary_digits():
    digits = datasets.load_digits()

    targets = digits.target
    data = digits.data
    high = np.max(data)

    data_bool = np.round(data / high).astype("uint8")
    return data_bool, targets
