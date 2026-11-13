# -*- coding: utf-8 -*-
"""Leakage accounting on the corner path.

Locks the explicit, non-returning leakages that address the external review
point on macroeconomic accounts. The closure is exact for the task, population
and fiscal flows only; after-tax capital income and the unconsumed transfer
share are recorded as leakages to capital owners and are not recycled into the
modeled consumption block. Values are produced directly from the engine on the
corner calibration (tau=0.15, pE=0.45, phi_max=0.02) under the endogenous
reinstatement path of the main model; no value is hard-coded beyond the
manuscript-reported rounded figures.
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIMDIR = os.path.join(os.path.dirname(_HERE), "simulation")

_spec = importlib.util.spec_from_file_location(
    "v12", os.path.join(_SIMDIR, "double_task_sim_v12_endogenous_omega.py"))
v12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v12)

par = v12.Params(eta=0.40, tau=0.15, pE=0.45, phi_max=0.02)
r = v12.simulate_v12(par)

cap = r["cap_inc"]
tax = r["tax"]
GDP = r["GDP"]
TM = r["TM"]
C = r["C"]

after_tax_cap = (cap - tax).sum()
cap_share = after_tax_cap / GDP.sum()
unconsumed_tm = ((1.0 - par.mu_TM) * TM).sum()

print("=" * 70)
print("Leakage accounting on the corner path (endogenous)")
print("=" * 70)
print(f"after-tax capital income cum = {after_tax_cap:.4f} "
      f"({100.0 * cap_share:.1f}% of GDP)")
print(f"unconsumed transfer (1-mu)TM = {unconsumed_tm:.4f}")
print(f"cumulative GDP = {GDP.sum():.4f}, consumption = {C.sum():.4f}")

checks = [
    ("after-tax capital matches 265.85", abs(after_tax_cap - 265.85) < 0.02),
    ("capital share matches 81.0%", abs(cap_share - 0.810) < 0.002),
    ("unconsumed transfer matches 0.306", abs(unconsumed_tm - 0.306) < 0.005),
    ("after-tax capital is non-negative", bool((cap - tax >= -1e-12).all())),
    ("unconsumed transfer is non-negative",
     bool(((1.0 - par.mu_TM) * TM >= -1e-12).all())),
    ("leakages do not exceed GDP",
     after_tax_cap + unconsumed_tm + tax.sum() <= GDP.sum() + 1e-9),
]

print("\nchecks against the manuscript")
bad = 0
for name, ok in checks:
    print("  %-44s %s" % (name, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
print("\n%d/%d checks passed" % (len(checks) - bad, len(checks)))
raise SystemExit(1 if bad else 0)
