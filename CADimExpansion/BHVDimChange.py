import sys
import Benchmark.Benchmarks as BM
import numpy as np

if len(sys.argv) != 2:
    print("Usage: python experiment_script.py BHV_DIMENSION")
    sys.exit(1)

benchmark_nr = int(sys.argv[1])

UCR_length = BM.find_length_of_UCR_benchmark(benchmark_nr) * 1

BHV_DIMENSION = int(np.ceil(UCR_length / 8) * 8)

# Read the content of BHV.py
with open("C:\\Users\\tomglove\\.conda\\envs\\HDC\\Lib\\site-packages\\bhv\\shared.py", "r") as bhv_file:
    lines = bhv_file.readlines()

# Modify line 6 (assuming that line numbers start from 1)
if len(lines) >= 6:
    lines[5] = f"DIMENSION = {BHV_DIMENSION}\n"

# Write the modified content back to BHV.py
with open("C:\\Users\\tomglove\\.conda\\envs\\HDC\\Lib\\site-packages\\bhv\\shared.py", "w") as bhv_file:
    bhv_file.writelines(lines)