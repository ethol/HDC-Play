import time

from bhv.symbolic import SymbolicBHV, Var
from bhv.vanilla import VanillaBHV as BHV, DIMENSION
import bhv.shared as sh
import numpy as np


def make_rule(r: int):
    mask = [b == '1' for b in reversed(bin(r)[2:].rjust(8, "0"))]
    formula = SymbolicBHV.synth([Var("left"), Var("center"), Var("right")], mask)
    formula = formula.simplify()
    print("formula:", formula.show())
    return lambda x: formula.execute(vars={"left": x.roll_bits(1), "center": x, "right": x.roll_bits(-1)})


def run_rule(init, rule, steps, steps_to_Keep):


    if DIMENSION != len(init):
        # print("change dimension")
        raise NotImplementedError(f"BHV DIMENSION needs to match parameter dimension, set the DIMENSION inside bhv, BHV dim: {DIMENSION} vs CA dim: {len(init)}")

    last_v = BHV.from_bitstring("".join(init.astype("str")))
    vs = [last_v]

    for i in range(steps):
        vs.append(rule(vs[-1]))
    arr = np.array(vs[-steps_to_Keep:])
    get_strings = np.vectorize(lambda x: BHV.bitstring(x))
    int_arr = np.array(list(''.join(get_strings(arr))), dtype=np.uint8)
    return int_arr


def expand(data, ru, steps, keep, vocal=False):
    start = time.time()

    data_expanded = []
    rule = make_rule(ru)
    i = 0
    for d in data:
        exp = run_rule(d, rule, steps, keep)
        data_expanded.append(exp)
        if vocal:
            i += 1
            if i % 1000 == 0:
                print(i / len(data), time.time() - start)

    return np.array(data_expanded)
