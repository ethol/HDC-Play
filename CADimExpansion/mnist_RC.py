import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import CA.CA as BHVCA
import time
import pandas as pd
from libsvm.svmutil import *

start = time.time()

# Load the full MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

data = np.concatenate((X_train, X_test), axis=0)
labels = np.concatenate((y_train, y_test), axis=0)

data_bool = np.round(data / 255).astype("uint8")
data_bool = data_bool.reshape(data_bool.shape[0], -1)
print(data_bool.shape)

# holdout 33 % of the data for later.
data_bool, data_holdout, labels, labels_holdout = train_test_split(
    data_bool, labels, test_size=0.33, random_state=42)

ME = [
    # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    # 12, 13, 14, 15, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33,
    # 34, 35, 36,
    # 37, 38, 40, 41, 42, 43, 44, 45, 46, 50, 51, 54, 56, 57,
    58,
    60, 62, 72, 73, 74, 76, 77, 78, 90,
    94, 104, 105, 106, 108, 110, 122,
    126,
    128, 130, 132, 134, 136, 138, 140, 142, 146, 150, 152, 154, 156,
    160, 162, 164, 168, 170, 172, 178, 184, 200, 204, 232,
    ]

exp_df = pd.read_csv('data/exp_ml.csv')

scores = []
seed = np.random.randint(np.iinfo(np.int32).max)
data_train, data_test, labels_train, labels_test = train_test_split(
    data_bool, labels, test_size=0.90, random_state=seed)

# svm_base = SVC(kernel="linear")
print("split", time.time() - start)
# svm_base.fit(data_train, labels_train)
svm_base = svm_train(labels_train, data_train, '-q -t 0')
print("fitted", time.time() - start)
# score_base = svm_base.score(data_test, labels_test)
p_label, p_acc, p_val = svm_predict(labels_test, data_test, svm_base)
score_base = p_acc[0]/100
print("scored", time.time() - start)

print("base:", score_base)

for ru in ME:
    print(ru)
    data_expanded = []
    rule = BHVCA.make_rule(ru)
    i = 0
    for d in data_bool:
        exp = BHVCA.run_rule(d, rule, 3, 4)
        data_expanded.append(exp)
        i += 1
        if i % 1000 == 0:
            print(i / len(labels), time.time() - start)

    data_expanded = np.array(data_expanded)

    data_train, data_test, labels_train, labels_test = train_test_split(
        data_expanded, labels, test_size=0.90, random_state=seed)
    # svm_exp = SVC(kernel="linear")
    svm_exp = svm_train(labels_train, data_train, '-q -t 0')
    # svm_exp.fit(data_train, labels_train)
    p_label, p_acc, p_val = svm_predict(labels_test, data_test, svm_exp)
    score_exp = p_acc[0] / 100
    # score_exp = svm_exp.score(data_test, labels_test)
    print("expanded:", score_exp)
    print("improved:", score_exp - score_base)
    scores.append((ru, score_exp - score_base))

    temp_exp = pd.DataFrame.from_records({'rule': ru, 'difference_base': score_exp - score_base}, index=[0])
    temp_exp["seed"] = seed
    temp_exp["baseline"] = score_base
    temp_exp["benchmark"] = "MNIST"
    temp_exp["classifier"] = "SVM Linear"


    exp_df = pd.concat([exp_df, temp_exp], ignore_index=True)
    exp_df.to_csv("data/exp_ml.csv", index=False)

scores.sort(reverse=True, key=lambda x: x[1])
print(scores)
