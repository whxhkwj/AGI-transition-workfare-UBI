# -*- coding: utf-8 -*-
"""Early-window feasibility and minimum-transition-time constraint.

Locks the strengthened smoothness classification that addresses the external
review point on large-shock counterexamples. Two independent defects are
closed in the unified engine:
  * consumption feasibility is checked inside the early window (t=1..ew-1),
    not only after it, so an early consumption collapse is no longer ignored;
  * a minimum-transition-time constraint rejects paths whose market labour
    share is cleared below early_labor_floor inside the early window.

The large ability-growth counterexamples (g_A = 2, 10, 100) must now be
classified as crashes, while every reported policy path keeps its original
outcome and numerical trajectory. All values are produced directly from the
unified engine through the v12 endogenous entry; no value is hard-coded.
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIMDIR = os.path.join(os.path.dirname(_HERE), "simulation")

_spec = importlib.util.spec_from_file_location(
    "v12", os.path.join(_SIMDIR, "double_task_sim_v12_endogenous_omega.py"))
v12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v12)


def run(label, **kw):
    par = v12.Params(eta=0.40, mu_TM=0.50, **kw)
    r = v12.simulate_v12(par)
    ew = par.early_window
    print("%-22s outcome=%-11s crash=%-10s minC_early=%.4f min_sL_early=%.4f" % (
        label, r["outcome"], r["crash_type"],
        min(r["C"][1:ew]), min(r["sL"][1:ew])))
    return par, r


print("=" * 70)
print("Early-window feasibility and minimum-transition-time constraint")
print("=" * 70)

corner = dict(tau=0.15, pE=0.45, phi_max=0.02)
counter = [run("gA=%d" % g, gA=g, **corner) for g in (2, 10, 100)]
print("-" * 70)
legal = [
    ("CORNER",) + run("CORNER", **corner),
    ("STANDARD",) + run("STANDARD", tau=0.16, pE=0.45, phi_max=0.65),
    ("STRONG",) + run("STRONG", tau=0.15, pE=0.80, phi_max=0.65, retrain=0.8),
]

checks = []
# Counterexamples must be caught through both the early consumption and path rules.
for g, (par, r) in zip((2, 10, 100), counter):
    checks.append(("gA=%d crashes" % g, r["outcome"] == "crash"))
    checks.append(("gA=%d triggers path rule" % g, bool(r["path_crash"])))
    checks.append(("gA=%d triggers early consumption rule" % g,
                   bool(r["C_early_below"])))

# Reported policy paths must keep their original outcomes.
checks += [
    ("CORNER stays smooth", legal[0][2]["outcome"] == "smooth"),
    ("STANDARD stays smooth", legal[1][2]["outcome"] == "smooth"),
    ("STRONG stays not_reached", legal[2][2]["outcome"] == "not_reached"),
]
# Every legal path keeps market labour above the early floor and consumption
# above the subsistence floor inside the early window.
for label, par, r in legal:
    checks.append(("%s respects early labor floor" % label,
                   min(r["sL"][1:par.early_window]) >= par.early_labor_floor))
    checks.append(("%s respects early consumption floor" % label,
                   min(r["C"][1:par.early_window]) >= r["Cmin"]))

print("\nchecks against the manuscript")
bad = 0
for name, ok in checks:
    print("  %-44s %s" % (name, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
print("\n%d/%d checks passed" % (len(checks) - bad, len(checks)))
raise SystemExit(1 if bad else 0)
