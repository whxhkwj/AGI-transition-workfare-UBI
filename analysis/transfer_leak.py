# -*- coding: utf-8 -*-
"""Unconsumed-transfer leakage and the effect of raising the transfer
consumption propensity mu_TM.

The transfer enters demand with coefficient mu_TM, so the unconsumed share is
(1-mu_TM) TM. This script measures the size of that leakage, sweeps mu_TM and
compares its marginal effect with capital-income recycling, and checks why a
pure-transfer path still fails once the transfer leakage is fully closed.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "simulation"))
import numpy as np
import double_task_sim_unified as U


def run(**kw):
    kw.setdefault("mpc_cap", 0.0)
    p = U.Params(**kw)
    r = U.simulate_unified(p)
    tm_cum = float(np.sum(r["TM"][1:]))
    return r, dict(outcome=r["outcome"], minC=float(np.min(r["C"][1:])),
                   C_end=float(r["C"][-1]), tm=tm_cum,
                   uncons=(1.0 - p.mu_TM) * tm_cum,
                   crash_type=r["crash_type"])


scenarios = [
    ("corner pE .45 phi .02", dict(tau=0.15, pE=0.45, phi_max=0.02)),
    ("pE .45 phi .20", dict(tau=0.15, pE=0.45, phi_max=0.20)),
    ("pure transfer pE 0 phi .65", dict(tau=0.15, pE=0.0, phi_max=0.65)),
    ("full absorption", dict(tau=0.15, pE=0.45, phi_max=0.02,
                             pe_policy="full_absorption")),
]

print("Scale of transfer leakage at baseline mu_TM=0.50")
print("%-28s %-11s %8s %8s %8s" % ("scenario", "outcome", "minC", "TM_cum", "uncons"))
for name, kw in scenarios:
    m = run(mu_TM=0.50, **kw)[1]
    print("%-28s %-11s %8.4f %8.3f %8.3f"
          % (name, m["outcome"], m["minC"], m["tm"], m["uncons"]))

print("\nmu_TM sweep")
for name, kw in scenarios:
    print(" " + name)
    print("  %-8s %-11s %8s %8s %8s" % ("mu_TM", "outcome", "minC", "C_end", "uncons"))
    for mu in (0.50, 0.75, 0.90, 1.00):
        m = run(mu_TM=mu, **kw)[1]
        print("  %-8.2f %-11s %8.4f %8.4f %8.3f"
              % (mu, m["outcome"], m["minC"], m["C_end"], m["uncons"]))

print("\nPure transfer at mu_TM=1 (zero transfer leakage):")
m = run(mu_TM=1.0, tau=0.15, pE=0.0, phi_max=0.65)[1]
print("  outcome=%s crash_type=%s minC=%.4f"
      % (m["outcome"], m["crash_type"], m["minC"]))

print("\nMarginal minC gain from closing each leakage (tau=0.15)")
base = run(mu_TM=0.50, tau=0.15, pE=0.45, phi_max=0.02)[1]
tm_full = run(mu_TM=1.0, tau=0.15, pE=0.45, phi_max=0.02)[1]
cap = run(mu_TM=0.50, tau=0.15, pE=0.45, phi_max=0.02, mpc_cap=0.10)
capm = cap[1]
print("  corner mu_TM .5->1 : %+.4f" % (tm_full["minC"] - base["minC"]))
print("  corner mpc_cap 0->.10: %+.4f" % (capm["minC"] - base["minC"]))
