import pandas as pd
import numpy as np

exp = pd.read_csv('data/exp.csv')
pd.set_option('display.max_columns', None)
exp = exp.drop(["id", "dimensions_small", "dimensions_large", "steps", "steps_to_keep",
                "bundle_size", "n_trails", "experiment_desc",
                "timestamp"], axis=1)
exp["neg_z"] = exp.apply(lambda x: x["neg_mean"] / x["neg_std"], axis=1)
exp["pos_z"] = exp.apply(lambda x: x["pos_mean"] / x["pos_std"], axis=1)
exp["sep_z"] = exp.apply(lambda x: (x["pos_mean"] - x["neg_mean"]) / np.sqrt(x["neg_std"] ** 2 + x["pos_std"] ** 2),
                         axis=1)
exp = exp.sort_values(by=['sep_z'], ascending=False)

exp = exp.drop(["pos_z", "neg_z"], axis=1)

# filter by only minimum equivivalent rules
ca_min_eq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33,
             34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 50, 51, 54, 56, 57, 58, 60, 62, 72, 73, 74, 76, 77, 78, 90,
             94, 104, 105, 106, 108, 110, 122, 126, 128, 130, 132, 134, 136, 138, 140, 142, 146, 150, 152, 154, 156,
             160, 162, 164, 168, 170, 172, 178, 184, 200, 204, 232, ]

ca_class_1 = [0, 8, 32, 40, 128, 136, 160, 168]

ca_class_2 = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 19, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36, 37, 38,
              42, 43, 44, 46, 50, 51, 56, 57, 58, 62, 72, 73, 74, 76, 77, 78, 94, 104, 108, 130, 132, 134, 138, 140,
              142, 152, 154, 156, 162, 164, 170, 172, 178, 184, 200, 204, 232]

ca_class_3 = [18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150]

ca_class_4 = [41, 54, 106, 110]

exp = exp[exp["rule"].isin(ca_min_eq)]
# exp = exp[exp['sep_z'] > 3.1]

print(exp.head(50))
