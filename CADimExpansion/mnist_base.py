from sklearn.model_selection import train_test_split
import CA.CA as BHVCA
import Benchmark.Benchmarks as BM
import time
import pandas as pd
import numpy as np
from libsvm.svmutil import *

start = time.time()


data_bool, labels = BM.load_binary_mnist()

data_train, data_test, labels_train, labels_test = train_test_split(
    data_bool, labels, test_size=0.90, random_state=1630762881)

print("split", time.time() - start)
svm_base = svm_train(labels_train, data_train, '-q -t 0')
print("fitted", time.time() - start)
p_label, p_acc, p_val = svm_predict(labels_test, data_test, svm_base)
score_base = p_acc[0] / 100
print("scored", time.time() - start)

print("base 10% train Linear:", score_base)

# print("split", time.time() - start)
# svm_base = svm_train(labels_train, data_train, '-q -t 2')
# print("fitted", time.time() - start)
# p_label, p_acc, p_val = svm_predict(labels_test, data_test, svm_base)
# score_base = p_acc[0] / 100
# print("scored", time.time() - start)
#
# print("base 10% RBF:", score_base)
# #
#
x_train, y_train, x_test, y_test = BM.load_binary_mnist_original_split()
#
#
# svm_base = svm_train(y_train, x_train, '-q -t 0')
# print("fitted", time.time() - start)
# # score_base = svm_base.score(data_test, labels_test)
# p_label, p_acc, p_val = svm_predict(y_test, x_test, svm_base)
# score_base = p_acc[0] / 100
# print("scored", time.time() - start)
#
# print("base linear:", score_base)
#
#
# svm_base = svm_train(y_train, x_train, '-q -t 2')
# print("fitted", time.time() - start)
# # score_base = svm_base.score(data_test, labels_test)
# p_label, p_acc, p_val = svm_predict(y_test, x_test, svm_base)
# score_base = p_acc[0] / 100
# print("scored", time.time() - start)
#
# print("base RBF:", score_base)
