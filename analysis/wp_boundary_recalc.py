# -*- coding: utf-8 -*-
"""Recalculate the full-absorption fiscal frontier under the wage upgrade.

Sweeps gA at tau=0.15 for the legacy rule combination (fixed wage, market slack)
and the upgraded combination (dynamic wage, residual slack), then bisects the
consumption-floor boundary in each. The crash test here is the consumption floor
(min C over t>=10 below Cmin=0.30), matching the paper's fiscal-gate definition.
"""
import os, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIMDIR = os.path.join(ROOT, "simulation")

_spec = importlib.util.spec_from_file_location(
    "U", os.path.join(SIMDIR, "double_task_sim_unified.py"))
U = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(U)


def min_consumption(gA, mode, slack):
    par = U.Params(eta=0.40, mu_TM=0.50, tau=0.15, pE=1.0, phi_max=0.65,
                   gA=float(gA), wP_mode=mode, wage_slack=slack)
    r = U.simulate_unified(par)
    return float(np.min(r["C"][10:]))


C_MIN = 0.30


def holds_floor(gA, mode, slack):
    return min_consumption(gA, mode, slack) >= C_MIN - 1e-12


def bisect(mode, slack, lo=0.01, hi=0.25, tol=1e-4):
    assert holds_floor(lo, mode, slack) and not holds_floor(hi, mode, slack)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if holds_floor(mid, mode, slack):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


RULES = {"legacy (fixed/market)": ("fixed", "market"),
         "upgraded (dynamic/residual)": ("dynamic", "residual")}

grid = [0.01, 0.05, 0.07, 0.10, 0.14, 0.20]
for label, (mode, slack) in RULES.items():
    print("=" * 80)
    print(label)
    for gA in grid:
        mC = min_consumption(gA, mode, slack)
        print("  gA=%.2f  minC=%.4f  %s" %
              (gA, mC, "holds" if mC >= 0.30 - 1e-12 else "floor breached"))
    try:
        crit = bisect(mode, slack)
        print("  bisection critical gA = %.4f" % crit)
    except AssertionError:
        print("  no floor boundary inside [0.01, 0.25] (floor holds on whole range)")
