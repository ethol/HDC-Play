from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from scipy import spatial
import time
import numpy as np
from itertools import repeat
import seaborn as sns
import matplotlib.pylab as plt



digits = datasets.load_digits()

data = []
start = time.time()
heatmap = {
    0: {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []},
    1: {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []},
    2: {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []},
    3: {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []},
    4: {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []},
    5: {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []},
    6: {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []},
    7: {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []},
    8: {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []},
    9: {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []},
}
count_heatmap = np.zeros((10, 10), dtype="int")

for i in range(0, len(digits.target)):
    for j in range(i + 1, len(digits.target)):
        heatmap[digits.target[i]][digits.target[j]].append(
            # spatial.distance.euclidean(digits.data[i], digits.data[j]))
        1 - spatial.distance.cosine(digits.data[i], digits.data[j]))
        count_heatmap[digits.target[i]][digits.target[j]] += 1
    if i % 100 == 0:
        print(i / len(digits.target), time.time() - start)

print(time.time() - start)
avg_heatmap = np.zeros((10, 10), dtype="float")

for i in range(0, 10):
    for j in range(0, 10):
        avg_heatmap[i][j] = sum(heatmap[i][j])/ count_heatmap[i][j]


ax = sns.heatmap(avg_heatmap, linewidth=0.5)
plt.show()
