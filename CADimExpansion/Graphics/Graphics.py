import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({'font.size': 24})
# Wong, Bang. "Color blindness." nature methods 8.6 (2011): 441.
palette = [
    [230 / 255, 159 / 255, 0],  # 0 orange
    [86 / 255, 180 / 255, 233 / 255],  # 1 Sky Blue
    [0 / 255, 158 / 255, 115 / 255],  # 2 Bluish Green
    [240 / 255, 228 / 255, 66 / 255],  # 3 Yellow
    [0 / 255, 114 / 255, 178 / 255],  # 4 Blue
    [213 / 255, 94 / 255, 0],  # 5 Vermilion
    [204 / 255, 121 / 255, 167 / 255],  # 6 Reddish purple
]


def createBarOfPandasMD(exp, baseline):
    fig, ax = plt.subplots(figsize=(42, 7))

    xNum = np.arange(len(exp["rule"]))
    error_rate = 1 - exp["Avg_Acc"]
    ax.bar(xNum, error_rate, color=palette[4])

    ax.axhline(y=1 - baseline, color='black', linestyle='--', label='Threshold', linewidth=2)

    plt.xticks(xNum, labels=exp["rule"], rotation=45, ha='right', rotation_mode='anchor')

    plt.margins(x=0.01)

    # plt.show()
    fig.savefig(f'bar.png', bbox_inches="tight", dpi=600)


def createBarOfPandasUCR(exp, data_name, baseline_name):
    fig, ax = plt.subplots(figsize=(42, 7))
    labels_trunc = [label[:5] + '...' + label[-3:] if len(label) > 11 else label for label in exp["Name"]]
    xNum = np.arange(len(labels_trunc))
    error_rate = exp[data_name]
    bars_data = ax.bar(xNum, error_rate, color=palette[4])

    for i, (data, baseline) in enumerate(zip(error_rate, exp[baseline_name])):
        color = palette[2] if baseline > data else palette[5]
        ax.bar(xNum[i], baseline - data, bottom=data, color=color)

    plt.xticks(xNum, labels=labels_trunc, rotation=45, ha='right', rotation_mode='anchor', fontsize=12)

    plt.margins(x=0.01)
    plt.xlabel("Benchmark")
    plt.ylabel("Error Rate")

    plt.axvline(labels_trunc.index("FacesUCR") + 0.5, color='black', linestyle='--', linewidth=2)
    plt.axvline(labels_trunc.index("Dista...xTW") + 0.5, color='black', linestyle=':', linewidth=2)

    legend_elements = [
        Patch(color=palette[5], label='Reduction in performance'),
        Patch(color=palette[2], label='Improvement in performance'),
        Line2D([0], [0], linestyle='--', linewidth=2, color='black', label='Linearizeable'),
        Line2D([0], [0], linestyle=':', linewidth=2, color='black', label='Linearizeable < 0.90 acc')
    ]

    ax.legend(handles=legend_elements)

    fig.savefig(f'bar_{data_name}.png', bbox_inches="tight", dpi=600)
