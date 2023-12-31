from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from scipy import spatial
import time
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import CA.CA as BHVCA
import Benchmark.Benchmarks as BM
from sklearn.svm import SVC
import pandas as pd
import tensorflow as tf


def train(data_t, labels_t):
    grouped_data = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

    for i in range(len(labels_t)):
        grouped_data[labels_train_base[i]].append(data_t[i])
    bundles = np.zeros((10, len(data_t[0])), dtype="uint8")

    for i in range(10):
        bundles[i] = np.round(np.sum(grouped_data[i], axis=0) / len(grouped_data[i]))

    return bundles


def asses(bundles, data_t, labels_t):
    hamming_distances = np.sum(data_t == bundles[:, np.newaxis], axis=2)

    # Find the class with the highest Hamming distance for each sample
    predicted_labels = np.argmax(hamming_distances, axis=0)

    # Calculate the accuracy
    correct_predictions = np.sum(predicted_labels == labels_t)
    accuracy = correct_predictions / len(labels_t)

    return accuracy


start = time.time()

data_bool, labels = BM.load_binary_digits()

exp_df = pd.read_csv('data/exp_ml.csv')

ME = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33,
      34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 50, 51, 54, 56, 57, 58, 60, 62, 72, 73, 74, 76, 77, 78, 90,
      94, 104, 105, 106, 108, 110, 122, 126, 128, 130, 132, 134, 136, 138, 140, 142, 146, 150, 152, 154, 156,
      160, 162, 164, 168, 170, 172, 178, 184, 200, 204, 232, ]

split_training = 0.67
steps = 3
keep = 4

for i in range(100):
    scores = []
    seed = np.random.randint(np.iinfo(np.int32).max)

    data_train_base, data_test_base, labels_train_base, labels_test_base = train_test_split(
        data_bool, labels, test_size=1-split_training, random_state=seed)

    bundles = train(data_train_base, labels_train_base)
    score_base = asses(bundles, data_test_base, labels_test_base)

    print(score_base)
    for ru in ME:
        data_expanded = BHVCA.expand(data_bool, ru, steps, keep, vocal=False)

        data_train, data_test, labels_train, labels_test = train_test_split(
                data_expanded, labels, test_size=1-split_training, random_state=seed)

        bundles = train(data_train, labels_train)
        score_exp = asses(bundles, data_test, labels_test)

        print("expanded:", score_exp)
        print("improved:", score_exp - score_base)
        scores.append((ru, score_exp - score_base))

    temp_exp = pd.DataFrame(scores, columns=['rule', 'difference_base'])
    temp_exp["seed"] = seed
    temp_exp["baseline"] = score_base
    temp_exp["benchmark"] = "digits"
    temp_exp["classifier"] = "Bundle simple"
    temp_exp["split_training"] = split_training
    temp_exp["steps"] = steps
    temp_exp["keep"] = keep


    print(temp_exp.head())
    scores.sort(reverse=True, key=lambda x: x[1])
    print(scores)

    exp_df = pd.concat([exp_df, temp_exp], ignore_index=True)
    exp_df.to_csv("data/exp_ml.csv", index=False)

