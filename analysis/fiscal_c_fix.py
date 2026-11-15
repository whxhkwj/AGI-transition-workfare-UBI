# -*- coding: utf-8 -*-
"""Plan C corrected comparison: higher capital tax together with fiscal
reserve carry-over and/or relaxed spending caps.

Baseline closure (Plan C alone) locks workfare jobs at pE and transfers at
phi_max, with no carry-over, so extra tax revenue stays idle. This script
separates the two binding constraints and tests whether relaxing the spending
side turns the tax increase from ineffective to effective.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "simulation"))
import numpy as np
import double_task_sim_unified as U


def metrics(tau, pE, phi_max, reserve, pe_policy="fixed"):
    p = U.Params(tau=tau, pE=pE, phi_max=phi_max, fiscal_reserve=reserve,
                 pe_policy=pe_policy, mpc_cap=0.0)
    r = U.simulate_unified(p)
    tax_cum = float(np.sum(r["tax"][:-1]))
    spent_cum = float(np.sum(r["wP"][1:] * r["EP"][1:] + r["TM"][1:]))
    idle_pct = 100.0 * (tax_cum - spent_cum) / max(tax_cum, 1e-9)
    return dict(outcome=r["outcome"], minC=float(np.min(r["C"][1:])),
                C_end=float(r["C"][-1]), tax=tax_cum, spent=spent_cum,
                idle=idle_pct, EP=float(r["EP"][-1]), Urem=float(r["Urem"][-1]))


def show(label, **kw):
    m = metrics(**kw)
    print("%-30s %-11s %7.4f %7.4f %8.3f %8.3f %6.1f%% %7.3f %7.3f"
          % (label, m["outcome"], m["minC"], m["C_end"], m["tax"], m["spent"],
             m["idle"], m["EP"], m["Urem"]))


print("Constraint diagnosis at tau=0.40 (cap_share<1 = budget binding,")
print("cap_share=1 = spending caps binding):")
r0 = U.simulate_unified(U.Params(tau=0.40, pE=0.45, phi_max=0.02))
bb = [i for i in range(1, r0["cap_share"].size) if r0["cap_share"][i] < 0.999]
print("  budget-binding periods: %s" % bb)
print()

print("Plan C variants on the corner path (mpc_cap=0)")
print("%-30s %-11s %7s %7s %8s %8s %6s %7s %7s"
      % ("scenario", "outcome", "minC", "C_end", "tax", "spent", "idle%",
         "EP_end", "Urem"))

print("\n-- C alone: caps locked, no reserve --")
show("tau .15 pE .45", tau=0.15, pE=0.45, phi_max=0.02, reserve=False)
show("tau .40 pE .45", tau=0.40, pE=0.45, phi_max=0.02, reserve=False)

print("\n-- higher tax + fiscal reserve, caps unchanged --")
show("tau .40 + reserve", tau=0.40, pE=0.45, phi_max=0.02, reserve=True)

print("\n-- higher tax + relaxed spending caps, no reserve --")
show("tau .40 pE .80 phi .20", tau=0.40, pE=0.80, phi_max=0.20, reserve=False)
show("tau .40 full absorption", tau=0.40, pE=0.45, phi_max=0.02, reserve=False,
     pe_policy="full_absorption")

print("\n-- higher tax + reserve + relaxed caps --")
show("tau .40 pE .80 + reserve", tau=0.40, pE=0.80, phi_max=0.20, reserve=True)
