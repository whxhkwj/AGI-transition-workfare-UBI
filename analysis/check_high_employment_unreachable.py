# -*- coding: utf-8 -*-
"""Falsification check: is a high market-employment attractor reachable under
full substitution?

The reframed model treats a high market-employment endpoint as a counterfactual
under incomplete substitution rather than as a reachable basin. This script does
not assert that AGI must displace every task (that is the normative content of
Theorem 1 under c_d > 0). It tests the operational claim directly:

  * across the full-substitution parameter domain, no policy configuration,
    including one that raises every lever at once, holds market employment at a
    high share at the terminal date;
  * the configurations that stay smooth do so with a low terminal market share,
    supported by public employment rather than by market jobs;
  * a high market-employment share is recovered only by an exogenous high
    reinstatement rate, which is precisely the incomplete-substitution
    counterfactual, and the terminal share rises monotonically with that rate.

A high market-employment endpoint is defined as a terminal market share at or
above 0.50. The full-substitution ceiling stays below this while the exogenous
high-omega counterfactual crosses it, with a margin on both sides.
"""
import os, sys, importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SIMDIR = os.path.join(_ROOT, "simulation")
sys.path.insert(0, _SIMDIR)

import double_task_sim_unified as uni

T = 80
HIGH_SHARE = 0.50          # high market-employment threshold
CEILING = 0.45             # full-substitution policy ceiling must stay below this


def endo(**kw):
    par = uni.Params(T=T, **kw)
    r = uni.simulate_unified(par)
    return dict(outcome=r["outcome"], sL_end=float(r["sL_end"]),
                minC=float(min(r["C"][10:])), EP_end=float(r["EP"][-1]),
                Urem_end=float(r["Urem"][-1]),
                acct=float(abs(r["LD"][-1] + r["EP"][-1] + r["Urem"][-1]
                               - uni.Params(T=T, **kw).LS)))


print("=" * 70)
print("High market-employment attractor: reachability falsification check")
print("=" * 70)

# Full-substitution domain (endogenous, baseline beta_A = 1).
full = {
    "no policy": endo(tau=0.05, pE=0.0, phi_max=0.15),
    "corner pE=0.45": endo(tau=0.15, pE=0.45, phi_max=0.02),
    "strong pE=0.80": endo(tau=0.15, pE=0.80, phi_max=0.65, retrain=0.8),
    "maxed pE=0.95": endo(tau=0.15, pE=0.95, phi_max=0.65, retrain=1.0),
    "full absorption": endo(pe_policy="full_absorption", tau=0.15, phi_max=0.02),
}
print("\nfull-substitution domain")
print("%-18s %-11s %8s %8s %8s" %
      ("config", "outcome", "sL_end", "EP_end", "Urem_end"))
for name, d in full.items():
    print("%-18s %-11s %8.4f %8.4f %8.4f" %
          (name, d["outcome"], d["sL_end"], d["EP_end"], d["Urem_end"]))

# Incomplete-substitution counterfactual: exogenous high reinstatement rate.
print("\nexogenous reinstatement rate (incomplete-substitution counterfactual)")
om_share = {}
print("%-8s %-11s %8s" % ("omega", "outcome", "sL_end"))
for om in (0.3, 0.5, 0.7, 0.9):
    par = uni.Params(omega_mode="exogenous", omega=om, tau=0.15, pE=0.45, T=T)
    r = uni.simulate_unified(par)
    om_share[om] = float(r["sL_end"])
    print("%-8.2f %-11s %8.4f" % (om, r["outcome"], r["sL_end"]))

checks = []
# 1. No policy under full substitution loses market employment and crashes.
checks.append(("no-policy market employment collapses",
               full["no policy"]["sL_end"] < 0.05
               and full["no policy"]["outcome"] == "crash"))
# 2. Even with every policy lever raised, terminal market share stays below the
#    high-employment threshold and the tighter ceiling.
maxed = full["maxed pE=0.95"]["sL_end"]
checks.append(("policy maxed cannot reach high employment",
               maxed < CEILING and maxed < HIGH_SHARE))
# 3. No full-substitution configuration reaches the high-employment threshold.
checks.append(("no full-substitution config is high-employment",
               all(d["sL_end"] < HIGH_SHARE for d in full.values())))
# 4. The smooth full-substitution paths rely on public employment, not market
#    jobs: low market share and public employment present.
checks.append(("smooth paths rely on public employment",
               full["corner pE=0.45"]["sL_end"] < 0.20
               and full["corner pE=0.45"]["EP_end"] > 0.30
               and full["full absorption"]["sL_end"] < 0.20
               and full["full absorption"]["Urem_end"] < 0.05))
# 5. High market employment is recovered only by an exogenous high reinstatement
#    rate (incomplete substitution), which crosses the threshold.
checks.append(("high employment needs high omega (incomplete substitution)",
               om_share[0.9] > HIGH_SHARE and om_share[0.3] < HIGH_SHARE))
# 6. Terminal market share rises monotonically with the exogenous rate.
vals = [om_share[k] for k in sorted(om_share)]
checks.append(("market share monotone in omega",
               all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))))
# 7. Population identity closes on the full-absorption path.
checks.append(("population identity closes",
               full["full absorption"]["acct"] < 1e-9))

print("\n=== checks ===")
all_ok = True
for label, ok in checks:
    all_ok = all_ok and ok
    print("[%s] %s" % ("PASS" if ok else "FAIL", label))
print("\nOVERALL:", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
