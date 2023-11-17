import pandas as pd
import scipy.stats as stats
import numpy as np
import Graphics.Graphics as GX

exp = pd.read_csv('data/UCRData.csv', index_col=0)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

exp = exp.drop(exp.columns[[
    # 0,  # type
    # 1,  # Name
    # 2,  # Train
    # 3,  # Test
    # 4,  # Class
    # 5,  # Length
    # 6,  # ED (w=0)
    # 7,  # DTW (learned_w)
    # 8,  # DTW (w=100)
    # 9,  # Default rate
    10,  # Data donor/editor
    11,  # NN
    # 12, # SVM Linear
    # 13,  # SVM RBF
    14,  # SVM Poly
    15,  # SVM Sig
    # 16,
]], axis=1)

# exp = exp[exp["Linearize Train"] < 0.9]

#
# exp["NN delta"] = exp["ED (w=0)"] - exp["NN"]
# exp["SVM L delta"] = exp["ED (w=0)"] - exp["SVM Linear"]
# exp["SVM R delta"] = exp["ED (w=0)"] - exp["SVM RBF"]
# # exp["SVM P delta"] = exp["ED (w=0)"] - exp["SVM Poly"]
# # exp["SVM S delta"] = exp["ED (w=0)"] - exp["SVM Sig"]
# exp["SVM E L delta"] = exp["ED (w=0)"] - exp["SIM EXP SVM LINEAR"]
# exp["SVM E D L delta"] = exp["ED (w=0)"] - exp["SIM EXP DET SVM LINEAR"]
exp["SVM E D L delta SVM L"] = exp["SVM Linear"] - exp["SIM EXP DET SVM LINEAR"]
exp["SVM E D C94 L delta SVM L"] = exp["SIM EXP DET SVM LINEAR"] - exp["SIM EXP DET CA 94 SVM LINEAR"]
exp["SVM E D C126 L delta SVM L"] = exp["SIM EXP DET SVM LINEAR"] - exp["SIM EXP DET CA 126 SVM LINEAR"]
# exp["SVM E D C126 K2 L delta SVM L"] = exp["SIM EXP DET SVM LINEAR"] - exp["SIM EXP DET CA 126 K2 SVM LINEAR"]
exp["SVM E T C126 L delta SVM L"] = exp["SIM EXP T SVM LINEAR"] - exp["SIM EXP T CA 126 SVM LINEAR"]
exp["SVM E T C94 L delta SVM L"] = exp["SIM EXP T SVM LINEAR"] - exp["SIM EXP T CA 94 SVM LINEAR"]

# exp = exp.sort_values(by=['SVM E T C126 L delta SVM L'], ascending=False)
print(exp)

# aggr = exp.groupby(['Type']).agg({'NN delta': ['mean', 'std', 'max', 'min', "count"]})
# print(aggr)

# aggr = exp.groupby(['Type']).agg({'SVM L delta': ['mean', 'std', 'max', 'min', "count"]})
# print(aggr)
#
# aggr = exp.groupby(['Type']).agg({'SVM R delta': ['mean', 'std', 'max', 'min', "count"]})
# print(aggr)

print(exp.describe())

GX.createBarOfPandasUCR(exp, "SIM EXP DET SVM LINEAR", "SVM Linear")
