# -*- coding: utf-8 -*-
"""Numerically verify assumption (R2) of the reduced critical proposition:
the minimum consumption buffer B(g_A) = min_t(C_t) - C_min is strictly
decreasing in frontier growth and crosses zero at the reported critical value.

The proposition proves existence and uniqueness *conditional* on R1-R3; this
check supplies the evidence that the monotonicity assumption R2 holds on the
calibrated full-absorption core. It does not turn R2 into an analytic theorem.
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIMDIR = os.path.join(os.path.dirname(_HERE), "simulation")

_spec = importlib.util.spec_from_file_location(
    "unified", os.path.join(_SIMDIR, "double_task_sim_unified.py"))
unified = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(unified)

C_FLOOR = 0.30


def buffer(gA):
    par = unified.Params(gA=gA, tau=0.15, pe_policy="full_absorption")
    r = unified.simulate_unified(par)
    return min(r["C"][10:]) - C_FLOOR


grid = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.09, 0.0948,
        0.10, 0.12, 0.15, 0.20]
vals = [buffer(g) for g in grid]

print("g_A      B(g_A)")
for g, b in zip(grid, vals):
    print("%-8.4f %+.5f" % (g, b))

checks = []
diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
checks.append(("B strictly decreasing on the grid", all(d < 0 for d in diffs)))
checks.append(("B crosses zero between 0.09 and 0.10",
               vals[grid.index(0.09)] > 0 and vals[grid.index(0.10)] < 0))
checks.append(("B at the reported critical 0.0948 is within 1e-3 of zero",
               abs(vals[grid.index(0.0948)]) < 1e-3))

print("\nchecks")
bad = 0
for name, ok in checks:
    print("  %-48s %s" % (name, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
print("\n%d/%d checks passed" % (len(checks) - bad, len(checks)))
raise SystemExit(1 if bad else 0)
