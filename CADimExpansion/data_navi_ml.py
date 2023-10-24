import pandas as pd
import scipy.stats as stats
import numpy as np

# exp = pd.read_csv('data/exp_RC_dig.csv')
# exp = pd.read_csv('data/exp_RC_mnist.csv')
exp = pd.read_csv('data/exp_ml.csv')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def t_test(x):
    return stats.ttest_1samp(x, popmean=0.0).pvalue


# Custom function to calculate Bonferroni-corrected t-test
def bonferroni_t_test(x):
    baseline = 0  # Your baseline value
    alpha = 0.05  # Significance level
    n = 88  # Number of comparisons

    t_stat, p_value = stats.ttest_1samp(x, popmean=baseline)
    corrected_alpha = alpha / n

    if p_value < corrected_alpha:
        return "Significant"
    else:
        return "Not Significant"


# exp = exp[exp["benchmark"] == "MNIST"]
exp = exp[exp["benchmark"] == "digits"]
# exp = exp[exp["classifier"] == "SVM Linear"]
exp = exp[exp["classifier"] == "Bundle simple"]


print(exp.describe())

aggr = exp.groupby('rule').agg({'difference_base': ['mean', 'count', 'std',
                                                    np.max, np.min,
                                                    t_test, bonferroni_t_test
                                                    ]})
aggr.columns = ['Average', 'Count', 'Stdev',
                'Max', 'Min',
                'T_Test', "bonferroni_t_test"
                ]
aggr = aggr.sort_values(by=['Average'], ascending=False)
print(aggr)