# -*- coding: utf-8 -*-
"""Reproducibility check for the full-absorption public-employment policy.

Locks the global outcome bifurcation that replaces the old "high market
employment vs crash" framing with "crash vs smooth transition to full public
employment absorption". All values are produced from the unified engine; no
value is hard-coded.

Checks:
  * fixed-policy regression corner is unchanged;
  * full-absorption baseline (g_A=0.01) is smooth, residual unemployment ends
    at zero, and the population identity closes at machine precision;
  * the critical automation speed at tau=0.15 is found by bisection and is
    bracketed by the smooth (0.07) and crash (0.10) grid points;
  * the critical boundary rises with tau and saturates after tau ~ 0.20;
  * keeping a transfer buffer for the unfunded residual does not move the
    boundary, so the result is not an artifact of removing transfers.
"""
import os
import importlib.util
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIMDIR = os.path.join(os.path.dirname(_HERE), "simulation")

_spec = importlib.util.spec_from_file_location(
    "unified", os.path.join(_SIMDIR, "double_task_sim_unified.py"))
unified = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(unified)

FLOOR = 0.30


def run(keep_transfer=False, **kw):
    base = dict(eta=0.40, mu_TM=0.50, phi_max=0.02,
                pe_policy="full_absorption", full_keep_transfer=keep_transfer)
    base.update(kw)
    par = unified.Params(**base)
    return par, unified.simulate_unified(par)


def critical_gA(tau, keep_transfer=False):
    lo, hi = 0.005, 0.25
    for _ in range(14):
        mid = (lo + hi) / 2
        _, r = run(keep_transfer, tau=tau, gA=mid)
        if np.min(r["C"][1:]) >= FLOOR and r["outcome"] == "smooth":
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


print("=" * 70)
print("Full-absorption public-employment policy: reproducibility check")
print("=" * 70)

checks = []

# fixed-policy regression corner
pf = unified.Params(eta=0.40, mu_TM=0.50, tau=0.15, pE=0.45, phi_max=0.02)
rf = unified.simulate_unified(pf)
print("fixed corner: %s minC=%.4f" % (rf["outcome"], np.min(rf["C"][10:])))
checks.append(("fixed corner stays smooth", rf["outcome"] == "smooth"))
checks.append(("fixed corner minC near 0.356",
               abs(np.min(rf["C"][10:]) - 0.3562) < 5e-3))

# full-absorption baseline
par, r = run(tau=0.15, gA=0.01)
minc = np.min(r["C"][1:])
resid = np.max(np.abs(r["LD"] + r["EP"] + r["Urem"] - par.LS))
print("full baseline: %s minC=%.4f EP_end=%.4f Urem_end=%.4f acct=%.1e"
      % (r["outcome"], minc, r["EP"][-1], r["Urem"][-1], resid))
checks.append(("full baseline is smooth", r["outcome"] == "smooth"))
checks.append(("full baseline minC near 0.407", abs(minc - 0.4067) < 5e-3))
checks.append(("full baseline ends with no residual unemployment",
               r["Urem"][-1] < 0.05))
checks.append(("population identity closes", resid < 1e-12))

# grid points bracket the boundary
_, r07 = run(tau=0.15, gA=0.07)
_, r10 = run(tau=0.15, gA=0.10)
crit = critical_gA(0.15)
print("gA=0.07 %s, gA=0.10 %s, critical gA=%.4f"
      % (r07["outcome"], r10["outcome"], crit))
checks.append(("gA=0.07 smooth", r07["outcome"] == "smooth"))
checks.append(("gA=0.10 crashes", r10["outcome"] == "crash"))
checks.append(("critical gA near 0.095", abs(crit - 0.0948) < 8e-3))

# boundary rises with tau then saturates
taus = [0.10, 0.15, 0.20, 0.30, 0.40]
bounds = [critical_gA(t) for t in taus]
print("critical gA by tau:", ["%.4f" % b for b in bounds])
checks.append(("boundary rises tau 0.10->0.20", bounds[2] > bounds[0]))
checks.append(("boundary saturates after tau 0.20",
               (bounds[3] - bounds[2]) < 0.02 and (bounds[4] - bounds[3]) < 0.02))

# transfer buffer does not move the boundary
crit_buf = critical_gA(0.15, keep_transfer=True)
print("critical gA with transfer buffer=%.4f (pure=%.4f)" % (crit_buf, crit))
checks.append(("transfer buffer does not move boundary",
               abs(crit_buf - crit) < 8e-3))

print("\nchecks against the manuscript")
bad = 0
for name, ok in checks:
    print("  %-46s %s" % (name, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
print("\n%d/%d checks passed" % (len(checks) - bad, len(checks)))
raise SystemExit(1 if bad else 0)
