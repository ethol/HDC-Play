import numpy as np
import pandas as pd
import Benchmark.Benchmarks as BM
from sklearn.svm import SVC
from libsvm.svmutil import *



exp_df = pd.read_csv('data/UCRData.csv', index_col=0)



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

    X_train = np.concatenate((X_train, X_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)

    svm_exp = svm_train(y_train, X_train, '-q -t 0')
    p_label, p_acc, p_val = svm_predict(y_train, X_train, svm_exp)
    score_exp = p_acc[0] / 100

    print(f'Test accuracy base: {score_exp}')

    exp_df.loc[benchmark_nr, "Linearize Both"] = score_exp
    exp_df.to_csv("data/UCRData.csv")
