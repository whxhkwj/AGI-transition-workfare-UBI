# -*- coding: utf-8 -*-
"""Deflation-nesting check for the consumption-smooth classification (M2).

The smooth definition lists six finite-time criteria but does not run an
independent deflation veto: persistent deflation is nested inside the
consumption-floor and bounded-debt criteria. This script makes that nesting
operational and reproduces the supporting numbers:

  1. Persistent deflation is reported, not hidden: on the baseline corner the
     longest run of negative inflation is 79 prints, and across the full 8x8x6
     wide grid every one of the 384 paths has a longest run of 79.
  2. Real household debt is bounded by the hard credit ceiling: max b_t is at
     b_credit, and once it binds the nominal stock D_t contracts with P_t, so
     the Fisher debt-deflation channel cannot diverge.
  3. Real consumption is held above the floor (min C_t over t>=10 = 0.3562).
  4. The population partition still closes to machine precision.

Together (2)-(3) show that with the floor held and debt bounded, the sustained
fall in the price level produces no separate real welfare loss, which is the
stated nesting rule rather than an untested assumption.

Exit code is non-zero if any reproduced value falls outside the stated value.
"""
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "simulation"))

import double_task_sim_unified as engine  # noqa: E402


def longest_neg(pi):
    longest = cur = 0
    for x in pi < -0.001:
        cur = cur + 1 if x else 0
        longest = max(longest, cur)
    return longest


def corner():
    par = engine.Params(eta=0.40, mu_TM=0.50, T=80,
                        omega_mode="endogenous", omega=0.05,
                        tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0)
    return par, engine.simulate_unified(par)


def main():
    ok = True
    par, r = corner()
    pi, b, D, P, C = (np.array(r[k]) for k in ("pi", "b", "D", "P", "C"))
    LD, EP, Urem = r["LD"], r["EP"], r["Urem"]
    Cmin = par.q * par.w * par.LS

    run = longest_neg(pi)
    minC = float(np.min(C[10:]))
    partition_gap = max(abs(LD[t] + EP[t] + Urem[t] - par.LS) for t in range(par.T))

    print("baseline corner deflation metrics")
    print("  outcome                         :", r["outcome"])
    print("  longest negative-inflation run  :", run)
    print("  max real debt max b_t           : %.4f (ceiling b_credit=%.2f)" % (b.max(), par.b_credit))
    print("  nominal debt D[20] -> D[79]     : %.4f -> %.4f (P %.3f -> %.3f)" % (D[20], D[79], P[20], P[79]))
    print("  min C_t over t>=10              : %.4f (floor %.2f)" % (minC, Cmin))
    print("  partition max |LD+EP+Urem-LS|   : %.3e" % partition_gap)

    # (1) deflation is fully reported: 79 negative prints on the corner
    if r["outcome"] != "smooth" or run != 79:
        ok = False; print("MISMATCH: outcome/run", r["outcome"], run)
    # (2) real debt bounded by the hard ceiling and does not diverge
    if b.max() > par.b_credit + 1e-9 or b[-1] > par.b_credit + 1e-9:
        ok = False; print("MISMATCH: real debt exceeds the hard ceiling")
    if not (D[79] < D[20]):  # nominal stock contracts once b binds
        ok = False; print("MISMATCH: nominal debt did not contract with the price level")
    # (3) real consumption held above the floor
    if abs(minC - 0.3562) > 0.0005 or minC < Cmin:
        ok = False; print("MISMATCH: min consumption", minC)
    # (4) closed population partition
    if partition_gap > 1e-12:
        ok = False; print("MISMATCH: partition gap", partition_gap)

    # Wide grid: every one of the 384 paths has a longest negative run of 79.
    taus = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    phis = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.65]
    pes = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    runs = []
    for pe in pes:
        for tau in taus:
            for pm in phis:
                gp = engine.Params(eta=0.40, mu_TM=0.50, T=80,
                                   omega_mode="endogenous", omega=0.05,
                                   tau=tau, pE=pe, phi_max=pm, retrain=0.0)
                gr = engine.simulate_unified(gp)
                runs.append(longest_neg(np.array(gr["pi"])))
    runs = np.array(runs)
    print("\nwide grid (384 paths): longest negative-run min/median/max = %d/%d/%d"
          % (runs.min(), int(np.median(runs)), runs.max()))
    if runs.size != 384 or runs.min() != 79 or runs.max() != 79:
        ok = False; print("MISMATCH: deflation is not uniform across the wide grid")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
