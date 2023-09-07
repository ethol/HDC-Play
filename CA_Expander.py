import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
# from rapidfuzz.distance import DamerauLevenshtein

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.connection.random_boolean_net as rbn
import evodynamic.connection.custom as conn_custom
import evodynamic.cells.activation as act
import evodynamic.connection as connection

digits = datasets.load_digits()

clf = svm.SVC(kernel="linear")


def ca_run(init, width, ca_rule, steps):
    exp = experiment.Experiment()
    g_ca = exp.add_group_cells(name="g_ca", amount=width)
    neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
    g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin', init=init)
    g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn', width,
                                               neighbors=neighbors,
                                               center_idx=center_idx)

    exp.add_connection("g_ca_conn",
                       connection.WeightedConnection(g_ca_bin, g_ca_bin,
                                                     act.rule_binary_ca_1d_width3_func,
                                                     g_ca_bin_conn, fargs_list=ca_rule))

    exp.initialize_cells()

    im_ca = np.zeros((steps, width))
    im_ca[0] = exp.get_group_cells_state("g_ca", "g_ca_bin")[:, 0]

    for i in range(1, steps):
        exp.run_step()
        im_ca[i] = exp.get_group_cells_state("g_ca", "g_ca_bin")[:, 0]

    exp.close()
    return im_ca


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

numbers = [[], [], [], [], [], [], [], [], [], []]

for i in range(0, len(X_train)):
    numbers[y_train[i]].append(X_train[i])

# clf.fit(X_train, y_train)

def avg_func(group):
    return np.rint(np.array(group).mean(0)).astype(int)
    # return np.array(group).mean(0)

avg_digits = []
for group in numbers:
    avg_digits.append(avg_func(group))

def img_to_init(img):
    init_cond = np.zeros(1, dtype=int)
    for val in np.rint(img).astype(int):
        if val != 0:
            init_cond = np.hstack((init_cond, np.ones((val,), dtype=int)))
        init_cond = np.hstack((init_cond, np.zeros((16 - val,), dtype=int)))
    return init_cond

avg_expanded_digit = []
ca_rule = [(a,) for a in [1]]

for digit in avg_digits:
    init = img_to_init(digit)
    ca_gen = ca_run(init=init.reshape(len(init), 1), width=len(init), ca_rule=ca_rule, steps=2)
    avg_expanded_digit.append(ca_gen.flatten())



# predict = clf.predict(X_test)
predict = []
i = 0
for test in X_test:
    i += 1
    if i % 10:
        print(i)
    results = []
    init = img_to_init(test)
    ca_gen = ca_run(init=init.reshape(len(init), 1), width=len(init), ca_rule=ca_rule, steps=2)
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    for dig in avg_expanded_digit:
        results.append(1 - spatial.distance.cosine(ca_gen.flatten(), dig))
        # results.append(DamerauLevenshtein.distance(test, dig))
    predict.append(np.argmax(results))

print(
    f"{metrics.classification_report(y_test, predict)}\n"
)


disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predict,
                                                       # normalize="true", values_format=".2f"
                                                       )
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()