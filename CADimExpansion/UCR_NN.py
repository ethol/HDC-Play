import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import Benchmark.Benchmarks as BM
import tensorflow as tf

exp_df = pd.read_csv('data/UCRData.csv', index_col=0)


def create_model(input_size, nr_of_classes):
    # Build a simple neural network
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_size,)),  # Flattened input shape
        # # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Dense(512, activation="relu"),
        # # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Dense(256, activation="relu"),
        # # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Dense(128, activation="relu"),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(nr_of_classes, activation='softmax')  # Output layer with 10 units for 10 digits
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


for benchmark_nr in range(1, 129):
    data_train, data_test = BM.load_ucr_benchmark(benchmark_nr)

    # Split the dataset into features and labels
    X_train = data_train[:, 1:]  # Time series features (excluding the first column)
    y_train = data_train[:, 0]  # Class labels (in the first column)

    # Split the dataset into features and labels
    X_test = data_test[:, 1:]  # Time series features (excluding the first column)
    y_test = data_test[:, 0]  # Class labels (in the first column)

    min_label = np.min(y_train)
    y_train = y_train + abs(min_label)
    y_test = y_test + abs(min_label)

    base_model = create_model(len(X_train[0]), np.max(y_train + 1))

    base_model.fit(X_train, y_train, epochs=100, verbose=0)

    test_loss, score_base = base_model.evaluate(X_test, y_test)
    print(f'Test accuracy base: {score_base}')

    error_rate = "{:.4f}".format(1 - score_base)
    exp_df.loc[benchmark_nr, "NN"] = error_rate
    exp_df.to_csv("data/UCRData.csv")
