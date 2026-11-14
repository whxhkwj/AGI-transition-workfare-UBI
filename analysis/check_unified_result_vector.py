# -*- coding: utf-8 -*-
"""Verify the unified result vector (reviewer item 3 / A2).

The engine exposes ``r['summary']`` with three separately reported
dimensions: endpoint, consumption feasibility and social acceptability.
This check confirms that the structured summary is internally consistent
with the top-level fields across a range of policies, and that the three
acceptability dimensions are computed independently.
"""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from simulation.double_task_sim_unified import Params, simulate_unified

CONFIGS = [
    Params(pe_policy="full_absorption", gA=g, tau=t)
    for t in (0.05, 0.15, 0.20, 0.40)
    for g in (0.01, 0.07, 0.10, 0.20, 0.30)
]

checks = []
for par in CONFIGS:
    r = simulate_unified(par)
    s = r["summary"]
    ew = par.early_window
    C_below = float(np.min(r["C"][ew:])) < r["Cmin"]
    econ_crash = bool(C_below or r["C_early_below"])
    # Endpoint block matches the terminal arrays.
    endpoint_ok = (
        abs(s["endpoint"]["sL_end"] - r["sL"][-1]) < 1e-12
        and abs(s["endpoint"]["EP_end"] - r["EP"][-1]) < 1e-12
        and abs(s["endpoint"]["Urem_end"] - r["Urem"][-1]) < 1e-12
        and s["endpoint"]["outcome"] == r["outcome"]
    )
    # Consumption block matches the computed floor flags.
    consumption_ok = (
        abs(s["consumption"]["minC"] - float(np.min(r["C"]))) < 1e-12
        and s["consumption"]["C_below"] == C_below
        and s["consumption"]["C_early_below"] == r["C_early_below"]
        and s["consumption"]["econ_crash"] == econ_crash
    )
    # Social block matches the stress and path flags.
    social_ok = (
        abs(s["social"]["stress_peak"] - float(np.max(r["stress"]))) < 1e-12
        and s["social"]["social_crash"] == r["social_crash"]
        and s["social"]["path_crash"] == r["path_crash"]
        and s["social"]["crash_type"] == r["crash_type"]
    )
    checks.append((endpoint_ok, consumption_ok, social_ok))

n = len(CONFIGS)
all_endpoint = all(c[0] for c in checks)
all_consumption = all(c[1] for c in checks)
all_social = all(c[2] for c in checks)

print("configs checked:", n)
print("endpoint block consistent:    ", all_endpoint)
print("consumption block consistent: ", all_consumption)
print("social block consistent:      ", all_social)

# The three dimensions must be able to fail independently, demonstrating
# that terminal reach, the floor and the social threshold are not one flag.
econ_only = any(
    simulate_unified(p)["summary"]["consumption"]["econ_crash"]
    and not simulate_unified(p)["summary"]["social"]["social_crash"]
    for p in CONFIGS
)
path_only = any(
    simulate_unified(p)["summary"]["social"]["path_crash"]
    and not simulate_unified(p)["summary"]["consumption"]["econ_crash"]
    for p in CONFIGS
)
print("independent econ failure observed:", econ_only)
print("independent path failure observed:", path_only)

ok = all_endpoint and all_consumption and all_social and econ_only and path_only
print("UNIFIED RESULT VECTOR", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
