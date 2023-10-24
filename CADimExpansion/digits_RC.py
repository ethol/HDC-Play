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


def heat(data, targets):
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

    for i in range(0, len(targets)):
        for j in range(i + 1, len(targets)):
            heatmap[targets[i]][targets[j]].append(
                # spatial.distance.euclidean(digits.data[i], digits.data[j]))
                1 - spatial.distance.cosine(data[i], data[j]))
            count_heatmap[targets[i]][targets[j]] += 1
        if i % 100 == 0:
            print(i / len(targets), time.time() - start)

    print(time.time() - start)
    avg_heatmap = np.zeros((10, 10), dtype="float")

    for i in range(0, 10):
        for j in range(0, 10):
            avg_heatmap[i][j] = sum(heatmap[i][j]) / count_heatmap[i][j]

    print("avg distance between digits", np.sum(avg_heatmap) / 100)
    sorted_avg = avg_heatmap.flatten()
    sorted_avg.sort()
    sorted_avg = np.flip(sorted_avg)
    i = 0
    found = 0
    while found < 10:
        loc = np.where(avg_heatmap == sorted_avg[i])
        if loc[0] != loc[1]:
            print(f'{loc[0]}:{loc[1]} = {sorted_avg[i]}')
            found += 1
        i += 1

    print(sorted_avg)
    print(avg_heatmap)

    ax = sns.heatmap(avg_heatmap, linewidth=0.5)
    plt.show()


start = time.time()

data_bool, labels = BM.load_binary_digits()

ME = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33,
      34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 50, 51, 54, 56, 57, 58, 60, 62, 72, 73, 74, 76, 77, 78, 90,
      94, 104, 105, 106, 108, 110, 122, 126, 128, 130, 132, 134, 136, 138, 140, 142, 146, 150, 152, 154, 156,
      160, 162, 164, 168, 170, 172, 178, 184, 200, 204, 232, ]

exp_df = pd.read_csv('data/exp_ml.csv')

for i in range(100):
    scores = []
    seed = np.random.randint(np.iinfo(np.int32).max)
    svm_base = SVC(kernel="linear")
    data_train, data_test, labels_train, labels_test = train_test_split(
        data_bool, targets, test_size=0.33, random_state=seed)
    svm_base.fit(data_train, labels_train)
    score_base = svm_base.score(data_test, labels_test)
    print("base:", score_base)

    for ru in ME:
        print(ru)
        data_expanded = []
        rule = BHVCA.make_rule(ru)
        i = 0
        for d in data_bool:
            exp = BHVCA.run_rule(d.flatten(), rule, 3, 4)
            data_expanded.append(exp)
            i += 1
            # if i % 100 == 0:
            #     print(i / len(targets), time.time() - start)

        data_expanded = np.array(data_expanded)
        # heat(data_bool, targets)
        # heat(data_expanded, targets)

        svm_exp = SVC(kernel="linear")
        data_train, data_test, labels_train, labels_test = train_test_split(
            data_expanded, targets, test_size=0.33, random_state=seed)
        svm_exp.fit(data_train, labels_train)
        score_exp = svm_exp.score(data_test, labels_test)
        print("expanded:", score_exp)
        print("improved:", score_exp - score_base)
        scores.append((ru, score_exp - score_base))

    temp_exp = pd.DataFrame(scores, columns=['rule', 'difference_base'])
    temp_exp["seed"] = seed
    temp_exp["baseline"] = score_base
    temp_exp["benchmark"] = "digits"
    temp_exp["classifier"] = "SVM Linear"

    print(temp_exp.head())
    scores.sort(reverse=True, key=lambda x: x[1])
    print(scores)
    exp_df = pd.concat([exp_df, temp_exp], ignore_index=True)
    exp_df.to_csv("data/exp_ml.csv", index=False)
