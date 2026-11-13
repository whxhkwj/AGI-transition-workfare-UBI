# -*- coding: utf-8 -*-
"""Endpoint signs (E0)/(E1) verified numerically at the baseline.

Locks the engine verification of the endpoint assumptions that support the
interior workfare optimum. Rather than treating (E0) MB(0)>MC(0) and
(E1) MB(1)<MC(1) as assumptions only, the welfare loss is re-run under a
small perturbation of the workfare share at each endpoint and numerically
differenced. At the baseline, adding the first workfare unit lowers the loss
(E0 holds) and adding workfare at the top raises it (E1 holds), so the
interior optimum is supported by direct simulation rather than asserted.
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIMDIR = os.path.join(os.path.dirname(_HERE), "simulation")

_spec = importlib.util.spec_from_file_location(
    "v6", os.path.join(_SIMDIR, "double_task_sim_v6.py"))
v6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v6)


def loss_at(pe):
    p = v6.Params(eta=0.40, tau=0.15, pE=pe, phi_max=0.02)
    r = v6.simulate_v6(p)
    return v6.welfare(r, p)["total"]


eps = 0.01
d_near_zero = (loss_at(eps) - loss_at(0.0)) / eps
d_near_one = (loss_at(1.0) - loss_at(1.0 - eps)) / eps

print("=" * 70)
print("Endpoint signs (E0)/(E1) at the baseline")
print("=" * 70)
print("d(loss)/d(pE) near 0 = %.4f (E0: negative, MB>MC)" % d_near_zero)
print("d(loss)/d(pE) near 1 = %.4f (E1: positive, MB<MC)" % d_near_one)

e0 = d_near_zero < 0
e1 = d_near_one > 0
checks = [
    ("E0 holds: first workfare unit lowers loss", e0),
    ("E1 holds: workfare at the top raises loss", e1),
    ("both endpoint signs are verified", e0 and e1),
]

print("\nchecks against the manuscript")
bad = 0
for name, ok in checks:
    print("  %-44s %s" % (name, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
print("\n%d/%d checks passed" % (len(checks) - bad, len(checks)))
raise SystemExit(1 if bad else 0)
