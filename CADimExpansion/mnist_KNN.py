import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import Benchmark.Benchmarks as BM
import CA.CA as BHVCA

exp_df = pd.read_csv('data/exp_ml.csv')

# Load the MNIST dataset
data_bool, labels = BM.load_binary_mnist_rounded()

split_training = 0.90
steps = 3
keep = 4
rule = 126
n = 5
for i in range(100):
    seed = np.random.randint(np.iinfo(np.int32).max)

    # Split the dataset into a training set and a test set
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(data_bool, labels, test_size=1 - split_training, random_state=seed)

    # Preprocess the data
    scaler_base = StandardScaler()
    X_train_base = scaler_base.fit_transform(X_train_base)
    X_test_base = scaler_base.transform(X_test_base)

    # Create a KNN classifier
    knn_base = KNeighborsClassifier(n_neighbors=n)

    # Fit the KNN classifier on the training data
    knn_base.fit(X_train_base, y_train_base)

    # Make predictions using the KNN classifier
    y_pred_base = knn_base.predict(X_test_base)

    # Calculate accuracy
    score_base = accuracy_score(y_test_base, y_pred_base)
    print("Accuracy: {:.2f}%".format(score_base * 100))

    data_expanded = BHVCA.expand(data_bool, rule, steps, keep, vocal=False)

    X_train, X_test, y_train, y_test = train_test_split(data_expanded, labels, test_size=1 - split_training,
                                                                            random_state=seed)

    # Preprocess the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n)

    # Fit the KNN classifier on the training data
    knn.fit(X_train, y_train)

    # Make predictions using the KNN classifier
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    score_exp = accuracy_score(y_test, y_pred)
    print("Accuracy expanded: {:.2f}%".format(score_exp * 100))

    temp_exp = pd.DataFrame.from_records({'rule': rule, 'difference_base': score_exp - score_base}, index=[0])
    temp_exp["seed"] = seed
    temp_exp["baseline"] = score_base
    temp_exp["benchmark"] = "MNIST"
    temp_exp["classifier"] = f"KNN n={n}"
    temp_exp["split_training"] = split_training
    temp_exp["steps"] = steps
    temp_exp["keep"] = keep

    exp_df = pd.concat([exp_df, temp_exp], ignore_index=True)
    exp_df.to_csv("data/exp_ml.csv", index=False)
