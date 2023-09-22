import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def vector_normalization(x):
    magnitude = np.linalg.norm(x)
    return x / magnitude

def run_expts(mem, M, K, n_trials):
    expts = []

    for _ in range(n_trials):
        j = np.random.choice(M, K, replace=False)
        x = vector_normalization(np.sum(mem[j], axis=0))
        y = np.dot(mem, x)
        expt = {'pos': y[j], 'neg': y[np.delete(np.arange(M), j)]}
        expts.append(expt)

    return expts

def plot_hist(expts, param):
    fig, ax = plt.subplots()
    neg = np.array([expt['neg'] for expt in expts]).flatten()
    pos = np.array([expt['pos'] for expt in expts]).flatten()

    sns.histplot(neg, ax=ax, stat="density", color=[0.863, 0.1835, 0.1835, 0.5], common_norm=False)
    sns.histplot(pos, ax=ax, stat="density", color=[0.21875, 0.5546875, 0.234375, 0.5], common_norm=False)
    plt.title(f"Dimensions={param['Dimensions']}, Number of vectors={param['Vectors']},\n"
              f" Vectors in bundle={param['bundled vectors']}, trails={param['number of trials']}",
              fontdict={'fontweight': "bold"})
    plt.show()

# Define the bundle_sep_expt function
def bundle_sep_expt(D=32, M=1000, K=4, n_trials=1000, plot=True):
    # Generate mem matrix of normalized vectors
    mem = np.apply_along_axis(vector_normalization, axis=1, arr=np.random.random((M, D)) * 2 - 1)

    # Run experiments
    expts = run_expts(mem, M, K, n_trials)

    # Calculate signal statistics
    neg_mean = np.mean([expt['neg'] for expt in expts])
    pos_mean = np.mean([expt['pos'] for expt in expts])
    neg_std = np.std([expt['neg'] for expt in expts])
    pos_std = np.std([expt['pos'] for expt in expts])

    param = {'Dimensions': D, 'Vectors': M, 'bundled vectors': K, 'number of trials': n_trials}

    if plot:
        plot_hist(expts, param)

    return {
        'param': param,
        'res': pd.DataFrame({
            'row.names': ['neg', 'pos', 'sep'],
            'mean': [neg_mean, pos_mean, pos_mean - neg_mean],
            'st-dev': [neg_std, pos_std, np.sqrt(neg_std**2 + pos_std**2)],
            'z-score': [neg_mean / neg_std, pos_mean / pos_std, (pos_mean - neg_mean) / np.sqrt(neg_std**2 + pos_std**2)]
        })
    }

# Example usage:
result = bundle_sep_expt()
print(result["param"])
print(result["res"])