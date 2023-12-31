import pandas as pd

exp = pd.DataFrame({}, columns=["id", "dimensions_small", "dimensions_large", "steps", "steps_to_keep", "rule",
                                "bundle_size", "n_trails", "experiment_desc",
                                "neg_mean", "neg_std", "pos_mean", "pos_std", "timestamp"])
exp = exp.astype(
    {'id': 'int',
     'dimensions_small': 'int',
     'dimensions_large': 'int',
     'steps': 'int',
     'steps_to_keep': 'int',
     'rule': 'int',
     'bundle_size': 'int',
     'n_trails': 'int',
     "experiment_desc": 'str',
     'neg_mean': 'float',
     'neg_std': 'float',
     'pos_mean': 'float',
     'pos_std': 'float',
     'timestamp': 'int',
     }
)

exp_vec = pd.DataFrame({}, columns=["id", "exp_id", "vec_before", "vec_after"])
exp_vec = exp_vec.astype(
    {'id': 'int',
     'exp_id': 'int',
     'vec_before': 'str',
     'vec_after': 'str'
     }
)

exp_bundle = pd.DataFrame({}, columns=["id", "exp_id", "bundle_source", "bundle_vec"])
exp_bundle = exp_bundle.astype(
    {'id': 'int',
     'exp_id': 'int',
     'bundle_source': 'str',
     'bundle_vec': 'str'
     }
)

exp_ml = pd.DataFrame({}, columns=["rule", "baseline", "difference_base", "seed", "benchmark", "classifier", 'steps',
                                   'keep', 'split_training'])
exp_ml = exp_ml.astype(
    {
        'rule': 'int',
        'baseline': 'float',
        'difference_base': 'float',
        'seed': 'int',
        'benchmark': 'str',
        'classifier': 'str',
        'steps': "int",
        'keep': "int",
        'split_training': "float"
    }
)

# exp.to_csv("data/exp.csv", index=False)
# exp_vec.to_csv("data/exp_vec.csv", index=False)
# exp_bundle.to_csv("data/exp_bundle.csv", index=False)

# exp_ml.to_csv("data/exp_ml.csv", index=False)


# exp_df = pd.read_csv('data/exp_ml.csv')
#
# exp_df["split_training"] = 0.10
# exp_df.loc[exp_df.classifier == "SVM RBF 10% test", "split_training"] = 0.9
# exp_df.loc[exp_df.benchmark == "digits", "split_training"] = 0.67
# exp_df.loc[exp_df.benchmark == "MNIST Original split", "split_training"] = 0.90
#
# exp_df.to_csv("data/exp_ml.csv", index=False)
