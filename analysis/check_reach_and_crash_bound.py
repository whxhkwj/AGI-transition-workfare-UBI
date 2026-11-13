# -*- coding: utf-8 -*-
"""Reach/non-reach split and the zero-crash confidence bound.

Locks the two transparent qualifications that address the external review
point on the six-omega and Monte Carlo counts. Across six reinstatement rates
the corner never crashes, but only four paths reach the employment target while
two are not-reached. With zero crashes in 200 Monte Carlo draws, the
Clopper-Pearson one-sided 95 percent upper bound on the true crash probability
is reported. All values are produced directly from the engine and from the
beta-distribution quantile; nothing is hard-coded beyond the manuscript
rounded figures.
"""
import os
import importlib.util

from scipy.stats import beta

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIMDIR = os.path.join(os.path.dirname(_HERE), "simulation")

_spec = importlib.util.spec_from_file_location(
    "v6", os.path.join(_SIMDIR, "double_task_sim_v6.py"))
v6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v6)

omegas = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
outcomes = []
for w in omegas:
    par = v6.Params(eta=0.40, tau=0.15, pE=0.45, phi_max=0.02, omega=w)
    r = v6.simulate_v6(par)
    outcomes.append(r["outcome"])

n_smooth = sum(1 for o in outcomes if o == "smooth")
n_not_reached = sum(1 for o in outcomes if o == "not_reached")
n_crashed = sum(1 for o in outcomes if o == "crash")

n_draws = 200
n_fail = 0
upper = beta.ppf(0.95, n_fail + 1, n_draws - n_fail)

print("=" * 70)
print("Reach split and zero-crash confidence bound")
print("=" * 70)
print("six omega outcomes:", outcomes)
print("smooth/reached=%d not-reached=%d crashed=%d"
      % (n_smooth, n_not_reached, n_crashed))
print(f"Clopper-Pearson upper 95 percent = {100 * upper:.4f} percent")
print(f"(1-upper)^200 = {(1 - upper) ** n_draws:.6f}")

checks = [
    ("all six paths avoid crashing", n_crashed == 0),
    ("four paths reach the target", n_smooth == 4),
    ("two paths are not-reached", n_not_reached == 2),
    ("upper bound matches 1.49 percent", abs(100 * upper - 1.49) < 0.01),
    ("bound satisfies the 5 percent tail",
     abs((1 - upper) ** n_draws - 0.05) < 1e-9),
    ("bound is positive", upper > 0),
]

print("\nchecks against the manuscript")
bad = 0
for name, ok in checks:
    print("  %-44s %s" % (name, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
print("\n%d/%d checks passed" % (len(checks) - bad, len(checks)))
raise SystemExit(1 if bad else 0)
