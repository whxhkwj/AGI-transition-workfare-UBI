# -*- coding: utf-8 -*-
"""Wage timing difference between the profit tax base and household income.

The engine deducts the lagged wage w[t-1] LD[t] when forming the profit tax
base, while households receive the current wage w[t] LD[t]. On the declining
wage path w[t] <= w[t-1], the difference (w[t-1]-w[t]) LD[t] is a wage cost
accrued for tax but not actually paid. It stays on the firm side and makes the
tax base conservative. This check locks its cumulative magnitude and sign.
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_spec = importlib.util.spec_from_file_location(
    "v12", os.path.join(_ROOT, "simulation",
                        "double_task_sim_v12_endogenous_omega.py"))
v12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v12)

p = v12.Params(tau=0.15, pE=0.45, phi_max=0.02)
r = v12.simulate_v12(p)

w = r["w_real"]
LD = r["LD"]
GDP = r["GDP"]

total = 0.0
nonneg = True
for t in range(1, len(LD)):
    d = (w[t - 1] - w[t]) * LD[t]
    total += d
    if d < -1e-12:
        nonneg = False

share = 100.0 * total / GDP.sum()
print("=" * 70)
print("Wage timing difference (lagged tax base vs current household wage)")
print("=" * 70)
print("cumulative difference = %.4f" % total)
print("share of cumulative GDP = %.4f%%" % share)

checks = [
    ("nonnegative (w_t <= w_{t-1})", nonneg),
    ("cumulative difference near 0.36", abs(total - 0.3612) < 0.02),
    ("small share of GDP, near 0.11%", abs(share - 0.1101) < 0.02),
]
print("\nchecks against the manuscript")
bad = 0
for name, ok in checks:
    print("  %-44s %s" % (name, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
print("\n%d/%d checks passed" % (len(checks) - bad, len(checks)))
raise SystemExit(1 if bad else 0)
