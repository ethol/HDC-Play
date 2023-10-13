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

for i in range(0, int(len(digits.target)/1)):
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
        avg_heatmap[i][j] = sum(heatmap[i][j]) / count_heatmap[i][j]

print("avg distance between digits", np.sum(avg_heatmap) / 100)
sorted_avg = avg_heatmap.flatten()
sorted_avg.sort()

for i in range(10):
    loc = np.where(avg_heatmap == sorted_avg[i])
    print(f'{loc[0]}:{loc[1]} = {sorted_avg[i]}')

print(sorted_avg)
print(avg_heatmap)

ax = sns.heatmap(avg_heatmap, linewidth=0.5)
plt.show()
