import numpy as np
import pandas as pd
import Benchmark.Benchmarks as BM
import CA.CA as BHVCA
import HDC.HDC as HDC
from sklearn.svm import SVC
from libsvm.svmutil import *

import sys

if len(sys.argv) != 2:
    print("Usage: python experiment_script.py BHV_DIMENSION")
    sys.exit(1)
benchmark_nr = int(sys.argv[1])

exp_df = pd.read_csv('data/UCRData.csv', index_col=0)

steps = 3
keep = 4
rule = 94

data_train, data_test = BM.load_ucr_benchmark(benchmark_nr)

# Split the dataset into features and labels
X_train = data_train[:, 1:]  # Time series features (excluding the first column)
y_train = data_train[:, 0]  # Class labels (in the first column)

# Split the dataset into features and labels
X_test = data_test[:, 1:]  # Time series features (excluding the first column)
y_test = data_test[:, 0]  # Class labels (in the first column)
# X_train, X_test = HDC.similarity_preserving_expansion(X_train, X_test, 64)
X_train, X_test = HDC.simple_thresholding(X_train, X_test)

min_label = np.min(y_train)
y_train = y_train + abs(min_label)
y_test = y_test + abs(min_label)

svm_base = svm_train(y_train, X_train, '-q -t 0')
p_label, p_acc, p_val = svm_predict(y_test, X_test, svm_base)
score_base = p_acc[0] / 100

print(f'Test accuracy base: {score_base}')

error_rate = "{:.4f}".format(1 - score_base)
exp_df.loc[benchmark_nr, "SIM EXP T SVM LINEAR"] = error_rate
exp_df.to_csv("data/UCRData.csv")

overwrite_DIMENSION = len(X_train[0])

X_train = BHVCA.expand(X_train, rule, steps, keep, vocal=True)
X_test = BHVCA.expand(X_test, rule, steps, keep, vocal=True)

svm_exp = svm_train(y_train, X_train, '-q -t 0')
p_label, p_acc, p_val = svm_predict(y_test, X_test, svm_exp)
score_exp = p_acc[0] / 100

print(f'Test accuracy base: {score_exp}')

error_rate = "{:.4f}".format(1 - score_exp)
exp_df.loc[benchmark_nr, f"SIM EXP T CA {rule} SVM LINEAR"] = error_rate
exp_df.to_csv("data/UCRData.csv")

