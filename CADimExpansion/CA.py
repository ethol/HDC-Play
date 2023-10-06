from bhv.symbolic import SymbolicBHV, Var
from bhv.vanilla import VanillaBHV as BHV, DIMENSION
from bhv.visualization import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import minimize



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

def plot_hist(expts, param, thresholds=None):
    fig, ax = plt.subplots()
    neg = np.array([expt['neg'] for expt in expts]).flatten()
    pos = np.array([expt['pos'] for expt in expts]).flatten()

    sns.histplot(neg, ax=ax, stat="density", color=[0.863, 0.1835, 0.1835, 0.5], common_norm=False)
    sns.histplot(pos, ax=ax, stat="density", color=[0.21875, 0.5546875, 0.234375, 0.5], common_norm=False)

    plt.title(f"Dimensions={param['Dimensions']}, Number of vectors={param['Vectors']},\n"
              f" Vectors in bundle={param['bundled vectors']}, trails={param['number of trials']}",
              fontdict={'fontweight': "bold"})

    # Add legend and vertical lines at specified float values if provided
    if thresholds is not None:
        ax.axvline(thresholds["opt"]["threshold"], color='gray', linestyle='--',
                   label=f'opt at {thresholds["opt"]["threshold"]}')
        ax.axvline(thresholds["mid"]["threshold"], color='gray', linestyle='--',
                   label=f'mid at {thresholds["mid"]["threshold"]}')

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


def run_expts(mem, M, K, n_trials):
    expts = []

    for _ in range(n_trials):
        j = np.random.choice(M, K, replace=False)
        # bundle = np.zeros(len(mem[0]))
        bundle = np.sum(mem[j], axis=0)
        bundle = np.round(bundle/K).astype("int")

        bundle = bundle.astype("uint8")
        y = np.zeros(M, dtype="int")
        for i in range(M):
            y[i] = np.sum(mem[i] != bundle)
        y = 1 - y/(len(mem[0]))
        expt = {'pos': y[j], 'neg': y[np.delete(np.arange(M), j)]}
        expts.append(expt)

    return expts


def make_rule(r: int):
    mask = [b == '1' for b in reversed(bin(r)[2:].rjust(8, "0"))]
    formula = SymbolicBHV.synth([Var("left"), Var("center"), Var("right")], mask)
    formula = formula.simplify()
    print("formula:", formula.show())
    return lambda x: formula.execute(vars={"left": x.roll_bits(1), "center": x, "right": x.roll_bits(-1)})


def run_rule(init, rule, steps, steps_to_Keep):
    # init.astype(bool)
    last_v = BHV.from_bitstream(init.astype(bool))
    vs = [last_v]

    for i in range(steps):
        vs.append(rule(vs[-1]))
    arr = np.array(vs[-steps_to_Keep:])
    get_strings = np.vectorize(lambda x: BHV.bitstring(x))
    int_arr = [int(char) for char in ''.join(get_strings(arr))]

    return int_arr


def bundle_sep_CA_dimex_expt(Ds=32, M=1000, K=4, n_trials=1000, RULE=110, steps=10, steps_to_keep=4, plot=True):
    # ms is the collection of vectors in the low-d embedding space
    if DIMENSION != Ds:
        raise NotImplementedError("BHV DIMENSION needs to match parameter dimension, set the DIMENSION inside bhv")
    ms = np.random.randint(0, 2, size=(M, Ds))

    rule = make_rule(RULE)
    # rul CA and keep the last steps
    mem = np.apply_along_axis(lambda x: run_rule(x, rule, steps, steps_to_keep), axis=1, arr=ms)

    expts = run_expts(mem, M, K, n_trials)

    param = {'Dimensions': DIMENSION, 'Vectors': M, 'bundled vectors': K, 'number of trials': n_trials}

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


result = bundle_sep_CA_dimex_expt(Ds=128, M=1000, K=5, n_trials=1000, RULE=204, steps=0, steps_to_keep=1, plot=True)
print(result["param"])
print(result["res"])
print("Threshold", pd.DataFrame(result["threshold"]))

result = bundle_sep_CA_dimex_expt(Ds=128, M=1000, K=5, n_trials=1000, RULE=74, steps=3, steps_to_keep=4, plot=True)
print(result["param"])
print(result["res"])
print("Threshold", pd.DataFrame(result["threshold"]))

# with open(f"rule{RULE}.pbm", 'wb') as f:
#     Image(vs).pbm(f, binary=True)
