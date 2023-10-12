import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.manifold import TSNE
from scipy import linalg


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


def plot_hist(expts, param, thresholds=None):
    fig, ax = plt.subplots()
    neg = np.array([expt['neg'] for expt in expts]).flatten()
    pos = np.array([expt['pos'] for expt in expts]).flatten()

    sns.histplot(neg, ax=ax, stat="density", color=[0.863, 0.1835, 0.1835, 0.5], common_norm=False)
    sns.histplot(pos, ax=ax, stat="density", color=[0.21875, 0.5546875, 0.234375, 0.5], common_norm=False)

    # Add vertical lines at specified float values if provided
    if thresholds is not None:
        ax.axvline(thresholds["opt"]["threshold"], color='gray', linestyle='--',
                   label=f'opt at {thresholds["opt"]["threshold"]}')
        ax.axvline(thresholds["mid"]["threshold"], color='gray', linestyle='--',
                   label=f'mid at {thresholds["mid"]["threshold"]}')

    plt.title(f"Dimensions={param['Dimensions']}, Number of vectors={param['Vectors']},\n"
              f" Vectors in bundle={param['bundled vectors']}, trails={param['number of trials']}",
              fontdict={'fontweight': "bold"})

    # Add a legend for vertical lines if they are present
    if thresholds is not None:
        ax.legend()

    plt.show()


def plot_sim(m_small, m_large):
    cp_small = np.dot(m_small, m_small.T)  # Equivalent to crossprod in R
    cp_large = np.dot(m_large, m_large.T)  # Equivalent to crossprod in R

    # Create a DataFrame for the data
    data = {'cp_small': cp_small.flatten(), 'cp_large': cp_large.flatten()}
    df = pd.DataFrame(data)

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='cp_small', y='cp_large', data=df, alpha=0.5)
    plt.xlabel('cos in low-d space')
    plt.ylabel('cos in high-d space')
    plt.title('Similarity mapping low-d -> high-d')
    plt.show()


def optimal_threshold(expts, mean_thresh):
    def decision_stats(expts, threshold, details=False):
        # Calculate decision statistics for each experiment
        decisions = np.array([
            [
                np.sum(expt['pos'] < threshold),  # FN
                np.sum(expt['neg'] >= threshold),  # FP
                len(expt['pos']),  # n_pos
                len(expt['neg']),  # n_neg
                len(expt['pos']) + len(expt['neg'])  # n
            ]
            for expt in expts
        ])

        # Calculate overall error rates
        p_wrong_decision = (np.sum(decisions[:, 1]) + np.sum(decisions[:, 0])) / np.sum(decisions[:, 4])
        p_wrong_expt = np.sum(np.sum(decisions[:, [1, 0]], axis=1) > 0) / decisions.shape[0]

        if details:
            return {
                'threshold': threshold,
                'p_wrong_decision': p_wrong_decision,
                'p_wrong_expt': p_wrong_expt,
                'n_expts': len(expts),
                'n_decisions': np.sum(decisions[:, 4]),
                'n_FP': np.sum(decisions[:, 1]),
                'n_FN': np.sum(decisions[:, 0]),
                'p_FP': np.sum(decisions[:, 1]) / np.sum(decisions[:, 3]),
                'p_FN': np.sum(decisions[:, 0]) / np.sum(decisions[:, 2])
            }
        else:
            return p_wrong_decision

    # wrapper for the sake of scipy, prolly a better way to do this
    def objective_function(threshold):
        return decision_stats(expts, threshold)

    # Initial guess for the threshold
    initial_threshold = 0.5

    # Minimize the objective function to find the threshold
    result = minimize(objective_function, initial_threshold, method='Nelder-Mead')

    # The result contains the optimal threshold
    optimal_threshold = result.x[0]

    # Calculate decision statistics for the optimal and mean thresholds
    res = {
        'opt': decision_stats(expts, threshold=optimal_threshold, details=True),
        'mid': decision_stats(expts, threshold=mean_thresh, details=True)
    }

    return res


def succ_from_threshold(thresh, data):
    successful_experiments = sum(((exp['pos'] > thresh).all() and (exp['neg'] < thresh).all()) for exp in data)
    return successful_experiments / len(data)


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

    threshold_mean = (neg_mean + pos_mean) / 2
    thres = optimal_threshold(expts, threshold_mean)

    if plot:
        plot_hist(expts, param, thres)

    return {
        'param': param,
        'res': pd.DataFrame({
            'row.names': ['neg', 'pos', 'sep'],
            'mean': [neg_mean, pos_mean, pos_mean - neg_mean],
            'st-dev': [neg_std, pos_std, np.sqrt(neg_std ** 2 + pos_std ** 2)],
            'z-score': [neg_mean / neg_std, pos_mean / pos_std,
                        (pos_mean - neg_mean) / np.sqrt(neg_std ** 2 + pos_std ** 2)]
        }),
        "threshold": thres
    }


def bundle_sep_lin_dimex_expt(D=128, Ds=32, M=1000, K=4, n_trials=1000, plot=True):
    # tsne = TSNE(n_components=D, method='exact')  # Set the desired higher dimension (e.g., 64)

    # X is a matrix to transform from ns to n: [K x ns] [ns x n] -> [Ds x n]
    exp = np.random.normal(0, 1 / np.sqrt(Ds), size=(D, Ds))
    Svd = linalg.svd(exp, full_matrices=False)

    # ms is the collection of vectors in the low-d embedding space
    ms = np.apply_along_axis(vector_normalization, axis=1, arr=np.random.random((M, Ds)) * 2 - 1)
    # me = tsne.fit_transform(ms)
    # mem is the collection of vectors in the high-d space
    mem = np.apply_along_axis(vector_normalization, axis=1, arr=np.dot(ms, Svd[0].T))

    expts = run_expts(mem, M, K, n_trials)

    param = {'Dimensions': D, 'Vectors': M, 'bundled vectors': K, 'number of trials': n_trials}

    # Calculate signal statistics
    neg_mean = np.mean([expt['neg'] for expt in expts])
    pos_mean = np.mean([expt['pos'] for expt in expts])
    neg_std = np.std([expt['neg'] for expt in expts])
    pos_std = np.std([expt['pos'] for expt in expts])

    threshold_mean = (neg_mean + pos_mean) / 2
    thres = optimal_threshold(expts, threshold_mean)

    if plot:
        plot_hist(expts, param, thres)
        plot_sim(ms, mem)

    return {
        'param': param,
        'res': pd.DataFrame({
            'row.names': ['neg', 'pos', 'sep'],
            'mean': [neg_mean, pos_mean, pos_mean - neg_mean],
            'st-dev': [neg_std, pos_std, np.sqrt(neg_std ** 2 + pos_std ** 2)],
            'z-score': [neg_mean / neg_std, pos_mean / pos_std,
                        (pos_mean - neg_mean) / np.sqrt(neg_std ** 2 + pos_std ** 2)]
        }),
        "threshold": thres
    }

def bundle_sep_nonlin_dimex1_expt(D=128, Ds=32, M=1000, K=4, n_trials=1000, order=2, plot=True):

    # ms is the collection of vectors in the low-d embedding space
    ms = np.apply_along_axis(vector_normalization, axis=1, arr=np.random.random((M, Ds)) * 2 - 1)

    collapse_idx = np.random.choice(np.arange(D), size=Ds ** order, replace=True)
    def dimexp(x):
        y = np.outer(x, x).flatten()
        for j in range(order - 2):
            y = np.outer(x, y).flatten()
        out = np.zeros(D)
        for i in range(Ds**2):
            out[collapse_idx[i]] += y[i]

        return out

    # Apply dimensionality expansion and normalization
    mem = np.apply_along_axis(dimexp, axis=1, arr=ms)
    mem = mem / np.linalg.norm(mem, axis=1, keepdims=True)  # Normalize vectors

    expts = run_expts(mem, M, K, n_trials)

    param = {'Dimensions': D, 'Vectors': M, 'bundled vectors': K, 'number of trials': n_trials}

    # Calculate signal statistics
    neg_mean = np.mean([expt['neg'] for expt in expts])
    pos_mean = np.mean([expt['pos'] for expt in expts])
    neg_std = np.std([expt['neg'] for expt in expts])
    pos_std = np.std([expt['pos'] for expt in expts])

    threshold_mean = (neg_mean + pos_mean) / 2
    thres = optimal_threshold(expts, threshold_mean)

    if plot:
        plot_hist(expts, param, thres)
        plot_sim(ms, mem)

    return {
        'param': param,
        'res': pd.DataFrame({
            'row.names': ['neg', 'pos', 'sep'],
            'mean': [neg_mean, pos_mean, pos_mean - neg_mean],
            'st-dev': [neg_std, pos_std, np.sqrt(neg_std ** 2 + pos_std ** 2)],
            'z-score': [neg_mean / neg_std, pos_mean / pos_std,
                        (pos_mean - neg_mean) / np.sqrt(neg_std ** 2 + pos_std ** 2)]
        }),
        "threshold": thres
    }


# Example usage:
# result = bundle_sep_lin_dimex_expt()
# print(result["param"])
# print(result["res"])


# Example usage:

np.random.seed(42)
result = bundle_sep_expt()
print(result["param"])
print(result["res"])
print("Threshold", pd.DataFrame(result["threshold"]))

result = bundle_sep_lin_dimex_expt()
print(result["param"])
print(result["res"])
print("Threshold", pd.DataFrame(result["threshold"]))

result = bundle_sep_nonlin_dimex1_expt()
print(result["param"])
print(result["res"])
print("Threshold", pd.DataFrame(result["threshold"]))

result = bundle_sep_nonlin_dimex1_expt(order=3)
print(result["param"])
print(result["res"])
print("Threshold", pd.DataFrame(result["threshold"]))

result = bundle_sep_nonlin_dimex1_expt(order=4)
print(result["param"])
print(result["res"])
print("Threshold", pd.DataFrame(result["threshold"]))
