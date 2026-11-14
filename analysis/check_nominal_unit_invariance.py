"""Nominal-unit (numeraire) invariance check.

The household credit block must be homogeneous of degree one in the price level:
rescaling the initial price P0 (the numeraire) should leave every real path
unchanged and scale every nominal path by the same factor. This guards the
unit-correct form of the net-borrowing equation, where the real consumption-gap
term is multiplied by the current price before entering nominal debt.

Run:  python3 analysis/check_nominal_unit_invariance.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation.double_task_sim_unified import Params, simulate_unified  # noqa: E402

REAL_KEYS = ["C", "b", "sL", "LD", "EP", "Urem", "w_real", "TM"]
NOMINAL_KEYS = ["D", "P"]
TOL = 1e-9
P0_GRID = [0.5, 2.0, 3.0]
SCENARIOS = [
    dict(pe_policy="full_absorption", gA=0.01, tau=0.15),
    dict(pe_policy="full_absorption", gA=0.15, tau=0.30),
    dict(pe_policy="fixed", gA=0.01, tau=0.30),
]


def main() -> int:
    failures = 0
    for sc in SCENARIOS:
        base = simulate_unified(Params(P0=1.0, **sc))
        for p0 in P0_GRID:
            run = simulate_unified(Params(P0=p0, **sc))
            for key in REAL_KEYS:
                err = float(np.max(np.abs(np.asarray(run[key]) - np.asarray(base[key]))))
                if err > TOL:
                    failures += 1
                    print(f"REAL FAIL {sc} {key} P0={p0} err={err:.3e}")
            for key in NOMINAL_KEYS:
                err = float(np.max(
                    np.abs(np.asarray(run[key]) / p0 - np.asarray(base[key]))))
                if err > TOL:
                    failures += 1
                    print(f"NOMINAL FAIL {sc} {key} P0={p0} err={err:.3e}")
        print(f"scenario {sc} outcome={base['outcome']} "
              f"P0 grid {[1.0] + P0_GRID}: real paths identical")
    if failures:
        print(f"UNIT INVARIANCE: FAIL ({failures} violations)")
        return 1
    print("UNIT INVARIANCE: PASS (numeraire neutral, real paths invariant)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
