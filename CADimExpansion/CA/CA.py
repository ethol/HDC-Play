from bhv.symbolic import SymbolicBHV, Var
from bhv.vanilla import VanillaBHV as BHV
import numpy as np

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
