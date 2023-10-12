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
    initial_threshold = mean_thresh

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


def plot_hist(expts, param, rule, thresholds=None):
    fig, ax = plt.subplots()
    neg = np.array([expt['neg'] for expt in expts]).flatten()
    pos = np.array([expt['pos'] for expt in expts]).flatten()

    sns.histplot(neg, ax=ax, stat="density", color=[0.863, 0.1835, 0.1835, 0.5], common_norm=False)
    sns.histplot(pos, ax=ax, stat="density", color=[0.21875, 0.5546875, 0.234375, 0.5], common_norm=False)

    plt.title(f"Dimensions={param['Dimensions'] * param['steps_to_keep']}, Number of vectors={param['Vectors']},\n"
              f" Vectors in bundle={param['bundled vectors']}, trails={param['number of trials']}",
              fontdict={'fontweight': "bold"})

    # Add legend and vertical lines at specified float values if provided
    if thresholds is not None:
        ax.axvline(thresholds["opt"]["threshold"], color='gray', linestyle='--',
                   label=f'opt at {thresholds["opt"]["threshold"]}')
        ax.axvline(thresholds["mid"]["threshold"], color='gray', linestyle='--',
                   label=f'mid at {thresholds["mid"]["threshold"]}')

        ax.legend()
    plt.savefig(f'data/figures/hist_rule_{rule}.png', bbox_inches="tight", dpi=300)
    # plt.show()


def plot_sim(m_small, m_large, rule):
    cp_small = np.dot(m_small, m_small.T)  # Equivalent to crossprod in R
    cp_large = np.dot(m_large, m_large.T)  # Equivalent to crossprod in R

    l2_small = np.linalg.norm(cp_small, ord=2)
    l2_large = np.linalg.norm(cp_large, ord=2)
    cp_small = cp_small / l2_small
    cp_large = cp_large / l2_large
    # Create a DataFrame for the data
    data = {'cp_small': cp_small.flatten(), 'cp_large': cp_large.flatten()}
    df = pd.DataFrame(data)

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='cp_small', y='cp_large', data=df, alpha=0.5)
    plt.xlabel('dot in low-d space')
    plt.ylabel('dot in high-d space')
    plt.title('Similarity mapping low-d -> high-d')
    # plt.show()
    plt.savefig(f'data/figures/sim_rule_{rule}.png', bbox_inches="tight", dpi=300)


def plot_sim_heat(m_small, m_large):
    size = len(m_small)
    y = np.zeros((size, size), dtype="uint32")
    for i in range(size):
        for j in range(size):
            y[i][j] = np.sum(m_small[i] != m_large[j])
    y = 1 - y / (len(m_small[0]))
    plt.figure(figsize=(14, 12))
    sns.heatmap(y)
    plt.show()


def run_expts(mem, M, K, n_trials, store=False, exp_id=0):
    expts = []
    bundles = []
    bundele_id = []

    for _ in range(n_trials):
        bundle_id_set = np.random.choice(M, K, replace=False)
        bundle = np.sum(mem[bundle_id_set], axis=0)
        bundle = np.round(bundle / K).astype("int")

        bundle = bundle.astype("uint8")
        dis = np.zeros(M, dtype="int")
        for i in range(M):
            dis[i] = np.sum(mem[i] != bundle)
        dis = 1 - dis / (len(mem[0]))
        expt = {'pos': dis[bundle_id_set], 'neg': dis[np.delete(np.arange(M), bundle_id_set)]}
        expts.append(expt)
        if store:
            bundles.append(bundle)
            bundele_id.append(bundle_id_set)

    if store:
        exp_bundle = pd.read_csv('data/exp_bundle.csv')
        id_prev_max = exp_bundle["id"].max() + 1 if not exp_bundle.empty else 0
        for i in range(0, n_trials):
            exp_bundle_temp = {"id": i + id_prev_max,
                               "exp_id": exp_id,
                               "bundle_source": '/'.join(map(str, bundele_id[i])),
                               "bundle_vec": ''.join(bundles[i].astype("str"))}
            exp_bundle = pd.concat([exp_bundle, pd.DataFrame.from_records(exp_bundle_temp, index=[0])],
                                   ignore_index=True)
        exp_bundle.to_csv("data/exp_bundle.csv", index=False)
    return expts


def make_rule(r: int):
    mask = [b == '1' for b in reversed(bin(r)[2:].rjust(8, "0"))]
    formula = SymbolicBHV.synth([Var("left"), Var("center"), Var("right")], mask)
    formula = formula.simplify()
    print("formula:", formula.show())
    return lambda x: formula.execute(vars={"left": x.roll_bits(1), "center": x, "right": x.roll_bits(-1)})


def run_rule(init, rule, steps, steps_to_Keep):
    last_v = BHV.from_bitstring("".join(init.astype("str")))
    vs = [last_v]

    for i in range(steps):
        vs.append(rule(vs[-1]))
    arr = np.array(vs[-steps_to_Keep:])
    get_strings = np.vectorize(lambda x: BHV.bitstring(x))
    int_arr = [int(char) for char in ''.join(get_strings(arr))]
    return int_arr


def bundle_sep_CA_dimex_expt(Ds=32, M=1000, K=4, n_trials=1000, rule=110, steps=10, steps_to_keep=4, plot=True,
                             store=True):
    if DIMENSION != Ds:
        raise NotImplementedError("BHV DIMENSION needs to match parameter dimension, set the DIMENSION inside bhv")
    exp = pd.read_csv('data/exp.csv')
    new_exp_id = exp["id"].max() + 1 if not exp.empty else 0

    # ms is the collection of vectors in the low-d embedding space
    ms = np.random.randint(0, 2, size=(M, Ds))

    # Apply the CA and recover the expanded vector
    rule_bhv_func = make_rule(rule)
    mem = np.apply_along_axis(lambda x: run_rule(x, rule_bhv_func, steps, steps_to_keep), axis=1, arr=ms)

    # test bundles
    expts = run_expts(mem, M, K, n_trials, store=store, exp_id=new_exp_id)

    param = {'Dimensions': DIMENSION, 'Vectors': M, 'bundled vectors': K,
             'number of trials': n_trials, "steps_to_keep": steps_to_keep}

    # Calculate signal statistics
    neg_mean = np.mean([expt['neg'] for expt in expts])
    pos_mean = np.mean([expt['pos'] for expt in expts])
    neg_std = np.std([expt['neg'] for expt in expts])
    pos_std = np.std([expt['pos'] for expt in expts])

    threshold_mean = (neg_mean + pos_mean) / 2
    thres = optimal_threshold(expts, threshold_mean)

    if plot:
        plot_hist(expts, param, rule, thres)
        plot_sim(ms, mem, rule)
        # plot_sim_heat(ms, mem)
    if store:
        exp = pd.read_csv('data/exp.csv')
        exp_temp = {'id': new_exp_id,
                    'dimensions_small': Ds,
                    'dimensions_large': Ds * steps_to_keep,
                    'steps': steps,
                    'steps_to_keep': steps_to_keep,
                    'rule': rule,
                    'bundle_size': K,
                    'n_trails': n_trials,
                    "experiment_desc": f"concatenating previous CA expansion",
                    'neg_mean': neg_mean,
                    'neg_std': neg_std,
                    'pos_mean': pos_mean,
                    'pos_std': pos_std,
                    'timestamp': pd.Timestamp.now()
                    }
        exp = pd.concat([exp, pd.DataFrame.from_records(exp_temp, index=[0])], ignore_index=True)
        exp.to_csv("data/exp.csv", index=False)

        exp_vec = pd.read_csv('data/exp_vec.csv')
        exp_vec_max = exp_vec["id"].max() if not exp_vec.empty else 0
        ids = np.arange(start=0, stop=M) + (exp_vec_max + 1)
        temp_pd_vec = pd.DataFrame({
            "id": ids,
            "vec_before": [''.join(map(str, row)) for row in ms],
            "vec_after": [''.join(map(str, row)) for row in mem],
        })
        temp_pd_vec["exp_id"] = new_exp_id
        exp_vec = pd.concat([exp_vec, temp_pd_vec], ignore_index=True)
        exp_vec.to_csv("data/exp_vec.csv", index=False)

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


result = bundle_sep_CA_dimex_expt(Ds=128, M=1000, K=5, n_trials=1000, rule=1,
                                  steps=3, steps_to_keep=4, plot=False, store=False)
print(result["param"])
print(result["res"])
print("Threshold", pd.DataFrame(result["threshold"]))
# for rule in range(1, 255):
#     result = bundle_sep_CA_dimex_expt(Ds=128, M=1000, K=5, n_trials=1000, rule=rule,
#                                       steps=3, steps_to_keep=4, plot=True, store=True)
#     print(result["param"])
#     print(result["res"])
#     print("Threshold", pd.DataFrame(result["threshold"]))
