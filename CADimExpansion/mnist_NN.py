import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import Benchmark.Benchmarks as BM
import CA.CA as BHVCA

print("Available computing substrates: ", tf.config.list_physical_devices())

exp_df = pd.read_csv('data/exp_ml.csv')


def create_model(input_size):
    # Build a simple neural network
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_size,)),  # Flattened input shape
        # # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Dense(512, activation="relu"),
        # # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Dense(256, activation="relu"),
        # # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Dense(128, activation="relu"),
        # # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 units for 10 digits
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Load the MNIST dataset
data_bool, labels = BM.load_binary_mnist()

split_training = 0.90
steps = 3
keep = 4
rule = 126
epochs = 20

for i in range(100):
    seed = np.random.randint(np.iinfo(np.int32).max)
    data_train, data_test, labels_train, labels_test = train_test_split(
        data_bool, labels, test_size=1 - split_training, random_state=seed)

    base_model = create_model(784)
    # Train the model
    base_model.fit(data_train, labels_train, epochs=epochs)

    # Evaluate the model on the test dataset
    test_loss, score_base = base_model.evaluate(data_test, labels_test)
    print(f'Test accuracy base: {score_base}')

    data_expanded = BHVCA.expand(data_bool, rule, steps, keep, vocal=False)

    data_train, data_test, labels_train, labels_test = train_test_split(
        data_expanded, labels, test_size=1 - split_training, random_state=seed)

    model = create_model((784*keep))
    model.fit(data_train, labels_train, epochs=epochs)

    test_loss, score_exp = model.evaluate(data_test, labels_test)
    print(f'Test accuracy: {score_exp}')
    print("improved:", score_exp - score_base)

    temp_exp = pd.DataFrame.from_records({'rule': rule, 'difference_base': score_exp - score_base}, index=[0])
    temp_exp["seed"] = seed
    temp_exp["baseline"] = score_base
    temp_exp["benchmark"] = "MNIST"
    temp_exp["classifier"] = f"NN linear epochs {epochs}"
    temp_exp["split_training"] = split_training
    temp_exp["steps"] = steps
    temp_exp["keep"] = keep

    exp_df = pd.concat([exp_df, temp_exp], ignore_index=True)
    exp_df.to_csv("data/exp_ml.csv", index=False)

# for layer in model.layers:
#     if isinstance(layer, tf.keras.layers.Dense):
#         weights, biases = layer.get_weights()
#         # Define the number of input features and the number of sections
#         num_input_features = (784*keep)
#         num_sections = keep
#
#         # Compute and visualize the average weights for each section
#         section_size = num_input_features // num_sections
#         average_weights = np.zeros(num_sections)
#
#         for i in range(num_sections):
#             start = i * section_size
#             end = (i + 1) * section_size
#             average_weights[i] = np.mean(weights[start:end])
#
#         print("Average weights for each section:")
#         print(average_weights)
#
#

