# -*- coding: utf-8 -*-
"""How Plan A (mpc_cap capital recycling) moves the gate gA*.

Full-absorption calibration (same as the paper). Structure: low gA is smooth,
high gA crashes. gA* is the LARGEST tolerable gA, so a higher gA* is better.
The binding gate is either the economic (demand) gate or the path gate
(early market employment must stay >= early_labor_floor = 0.50).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "simulation"))
import numpy as np
import double_task_sim_unified as U

CMIN = 0.30


def gate_detail(tau, mpc_cap, iters=24):
    lo, hi = 0.005, 0.25
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p = U.Params(eta=0.40, mu_TM=0.50, phi_max=0.02,
                     pe_policy="full_absorption", tau=tau, gA=mid,
                     mpc_cap=mpc_cap)
        r = U.simulate_unified(p)
        if np.min(r["C"][1:]) >= CMIN and r["outcome"] == "smooth":
            lo = mid
        else:
            hi = mid
    g = 0.5 * (lo + hi)
    # identify which gate binds just above g*
    pr = U.simulate_unified(U.Params(eta=0.40, mu_TM=0.50, phi_max=0.02,
                                     pe_policy="full_absorption", tau=tau,
                                     gA=g + 0.002, mpc_cap=mpc_cap))
    return g, pr["crash_type"]


TAUS = (0.10, 0.15, 0.20, 0.30)
MCAPS = (0.0, 0.025, 0.05, 0.10, 0.15)

print("Largest tolerable gA* (higher = withstands faster automation)")
print("tau \\ mpc_cap " + "".join("%9.3f" % m for m in MCAPS))
detail = {}
for tau in TAUS:
    cells = []
    for m in MCAPS:
        g, ctype = gate_detail(tau, m)
        detail[(tau, m)] = (g, ctype)
        cells.append(g)
    print("%-12s" % tau + "".join("%9.4f" % v for v in cells))

print("\nBinding gate just above g* (econ = demand; path = transition speed)")
for tau in TAUS:
    print("tau %.2f: " % tau + " | ".join(
        "mpc %4.3f -> %-9s" % (m, detail[(tau, m)][1]) for m in MCAPS))

print("\nIncrease in tolerable gA* vs baseline mpc_cap=0")
print("tau \\ mpc_cap " + "".join("%9.3f" % m for m in MCAPS[1:]))
for tau in TAUS:
    b = detail[(tau, 0.0)][0]
    print("%-12s" % tau + "".join(
        "%+9.4f" % (detail[(tau, m)][0] - b) for m in MCAPS[1:]))
