# -*- coding: utf-8 -*-
"""Two-gate, same-budget transfer and household-closure checks.

Reproduces the explicit modeling added in this revision:
  1. fiscal gate vs adjustment-speed gate: at low tax the default boundary is a
     real fiscal boundary (unchanged when the speed gate is removed), while at
     high tax the default boundary is the speed gate (the fiscal boundary moves
     to the search bound);
  2. same-budget positive-spending transfer: reserving revenue for transfers
     runs a genuine positive transfer flow, lowers minimum consumption at every
     step and contracts the critical frontier;
  3. household zero-truncation closure: the unfinanced gap is zero on a smooth
     path, positive on a crash, and the resource identity closes to machine
     precision.
"""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "simulation"))
import double_task_sim_unified as U


def run(tau, ga, share=0.0, no_floor=False):
    p = U.Params(eta=0.40, mu_TM=0.50, phi_max=0.02,
                 pe_policy="full_absorption", tau=tau, gA=ga,
                 transfer_budget_share=share)
    if no_floor:
        p.early_labor_floor = 0.0
    return U.simulate_unified(p), p


def boundary(tau, no_floor=False):
    lo, hi = 0.005, 0.25
    for _ in range(14):
        mid = (lo + hi) / 2
        p = U.Params(eta=0.40, mu_TM=0.50, phi_max=0.02,
                     pe_policy="full_absorption", tau=tau, gA=mid)
        if no_floor:
            p.early_labor_floor = 0.0
        r = U.simulate_unified(p)
        if np.min(r["C"][1:]) >= 0.30 and r["outcome"] == "smooth":
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


ok = True

# 1. gate separation
b15 = boundary(0.15); b15f = boundary(0.15, no_floor=True)
b30 = boundary(0.30); b30f = boundary(0.30, no_floor=True)
print("tau .15: default %.4f, speed-off %.4f (fiscal gate, should match)" % (b15, b15f))
print("tau .30: default %.4f, speed-off %.4f (speed gate; speed-off at bound %.2f)"
      % (b30, b30f, 0.25))
if abs(b15 - b15f) > 0.01 or b30f < 0.24:
    ok = False; print("GATE SEPARATION FAIL")

# 2. same-budget positive transfer (phi_max=0.02, same as the diagnostic)
print("\nshare   minC     sumTM")
prev_c = None
for share in [0.0, 0.25, 0.50, 0.75, 1.0]:
    r, _ = run(0.15, 0.10, share=share)
    mc = np.min(r["C"][1:]); stm = float(np.sum(r["TM"][1:]))
    print("%.2f   %.4f  %.4f" % (share, mc, stm))
    if share > 0 and stm <= 0:
        ok = False; print("TRANSFER NOT POSITIVE")
    if prev_c is not None and mc > prev_c + 1e-9:
        ok = False; print("MINC NOT DECLINING")
    prev_c = mc

# 3. household closure: a no-policy crash that actually truncates at zero
pc = U.Params(tau=0.0, pE=0.0, phi_max=0.65, eta=0.40, mu_TM=0.50)
rc = U.simulate_unified(pc)
ps = U.Params(eta=0.40, mu_TM=0.50, phi_max=0.02,
              pe_policy="full_absorption", tau=0.15, gA=0.01)
rs = U.simulate_unified(ps)
gap_s = float(np.sum(rs["unfin_gap"])); gap_c = float(np.sum(rc["unfin_gap"]))
print("\nsmooth total unfinanced gap = %.5f (expect 0)" % gap_s)
print("crash  total unfinanced gap = %.5f (expect positive)" % gap_c)
print("crash  max gap over t=60..79 = %.5f" % np.max(rc["unfin_gap"][60:80]))
if gap_s != 0.0 or gap_c <= 0:
    ok = False; print("CLOSURE FAIL")
# rebuild disposable resources and verify the identity Yd + net_b/P + gap = C
rd = np.maximum(0, rc["rd"])
Yd = (rc["w_real"] * rc["LD"] + pc.wP * rc["EP"]
      + pc.mu_TM * rc["TM"] - pc.mu * rd)
lhs = Yd + rc["net_b"] / rc["P"] + rc["unfin_gap"]
resid = float(np.max(np.abs(lhs - rc["C"])))
print("closure identity max residual = %.2e" % resid)
if resid > 1e-12:
    ok = False; print("IDENTITY RESIDUAL FAIL")

print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
