# -*- coding: utf-8 -*-
"""Symmetric-limit welfare comparison on the same fiscal setting.

Locks the welfare-ranking reversal that addresses the external review point on
symmetric marginal propensities. On the identical fiscal setting
(tau=0.16, phi_max=0.65), with the transfer MPC set to one and the social
channel disabled (mu_TM=1, alpha_s=0), both regimes are smooth, but the pure
transfer (p_E=0) attains a lower engine welfare loss and a lower fiscal spend
than the public-employment path (p_E=0.45). The baseline asymmetric ranking is
also checked so the conditional nature of the reversal is explicit. All values
are produced directly from the engine; no value is hard-coded.
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIMDIR = os.path.join(os.path.dirname(_HERE), "simulation")

_spec = importlib.util.spec_from_file_location(
    "v6", os.path.join(_SIMDIR, "double_task_sim_v6.py"))
v6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v6)

import numpy as np


def evaluate(p_E, mu_TM, alpha_s):
    par = v6.Params(eta=0.40, tau=0.16, phi_max=0.65, pE=p_E,
                    mu_TM=mu_TM, alpha_s=alpha_s)
    r = v6.simulate_v6(par)
    w = v6.welfare(r, par)
    spend = r["TM"].sum() + par.wP * r["EP"].sum()
    return dict(outcome=r["outcome"], total=w["total"], spend=spend,
                minC=w["minC"], gap=w["gap"], cost=w["cost"])


print("=" * 70)
print("Symmetric-limit welfare comparison (same fiscal setting)")
print("=" * 70)
pe = evaluate(0.45, 1.0, 0.0)
tr = evaluate(0.0, 1.0, 0.0)
for label, d in (("public employment", pe), ("pure transfer", tr)):
    print("%-18s outcome=%-8s loss=%.4f spend=%.4f minC=%.4f" % (
        label, d["outcome"], d["total"], d["spend"], d["minC"]))

# Baseline asymmetric ranking on the same fiscal grid.
base_pe = evaluate(0.45, 0.5, 0.6)
base_tr = evaluate(0.0, 0.5, 0.6)
print("-" * 70)
for label, d in (("baseline PE", base_pe), ("baseline TR", base_tr)):
    print("%-18s outcome=%-8s loss=%.4f" % (label, d["outcome"], d["total"]))

checks = [
    ("symmetric both smooth",
     pe["outcome"] == "smooth" and tr["outcome"] == "smooth"),
    ("symmetric transfer loss lower", tr["total"] < pe["total"]),
    ("symmetric transfer spend lower", tr["spend"] < pe["spend"]),
    ("symmetric PE loss matches 55.34", abs(pe["total"] - 55.34) < 0.01),
    ("symmetric TR loss matches 51.23", abs(tr["total"] - 51.23) < 0.01),
    ("symmetric PE spend matches 36.12", abs(pe["spend"] - 36.12) < 0.02),
    ("symmetric TR spend matches 32.01", abs(tr["spend"] - 32.01) < 0.02),
    ("baseline PE loss lower than TR", base_pe["total"] < base_tr["total"]),
    ("baseline TR does not stay smooth", base_tr["outcome"] == "crash"),
]

print("\nchecks against the manuscript")
bad = 0
for name, ok in checks:
    print("  %-44s %s" % (name, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
print("\n%d/%d checks passed" % (len(checks) - bad, len(checks)))
raise SystemExit(1 if bad else 0)
