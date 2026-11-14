# -*- coding: utf-8 -*-
"""Actual-spending match, unused budget and the carry-forward regime (C1).

Three reviewer-driven questions are answered from the unified engine:

1. Does actual spending reconcile with the plan? Actual = planned - unmet,
   and the unused budget (revenue above the plan) is quantified rather than
   hidden.
2. A carry-forward regime that lets unused budget accumulate as a fiscal
   reserve is run against the baseline no-reserve regime.
3. The bisection lower bound (0.005) is a search censoring limit, not a
   point estimate of zero, so a frontier that reaches it is reported as
   censored rather than as a sharp zero/nonzero dichotomy.
"""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from simulation.double_task_sim_unified import Params, simulate_unified

WP = 0.85
FLOOR = 0.30
SEARCH_LO = 0.005

checks = []


def run(fr, **kw):
    return simulate_unified(Params(fiscal_reserve=fr, **kw))


# --- 1. Actual-spending reconciliation across the policy grid ----------------
recon_ok = True
unused_total = 0.0
for g in (0.01, 0.07, 0.10, 0.20):
    r = run(False, pe_policy="full_absorption", tau=0.15, gA=g, phi_max=0.02)
    plan = np.array(r["Eplan"])
    actual = WP * np.array(r["EP"]) + np.array(r["TM"])
    unmet = np.array(r["shortfall"])
    revenue = np.array(r["tax"])
    unused = np.maximum(0.0, revenue - plan)
    unused_total += unused.sum()
    # Accounting identity: actual = planned - unmet within the funded region.
    identity = np.allclose(actual + unmet, plan, atol=1e-8)
    recon_ok = recon_ok and identity
print("actual = planned - unmet reconciles:", recon_ok)
print("cumulative unused budget (no reserve): %.3f" % unused_total)
checks.append(("spending reconciliation identity", recon_ok))

# --- 2. Carry-forward regime does not move the result ------------------------
carry_ok = True
detail = []
for g in (0.01, 0.07, 0.10, 0.15, 0.20):
    a = run(False, pe_policy="full_absorption", tau=0.15, gA=g, phi_max=0.02)
    b = run(True, pe_policy="full_absorption", tau=0.15, gA=g, phi_max=0.02)
    same = (
        np.allclose(a["shortfall"], b["shortfall"], atol=1e-8)
        and abs(np.min(a["C"]) - np.min(b["C"])) < 1e-8
        and a["outcome"] == b["outcome"]
    )
    carry_ok = carry_ok and same
    detail.append((g, np.min(a["C"]), np.min(b["C"]), a["outcome"], b["outcome"]))
print("carry-forward leaves shortfall, minC and outcome unchanged:", carry_ok)
for g, ma, mb, oa, ob in detail:
    print("  gA=%.2f minC %.4f/%.4f outcome %s/%s" % (g, ma, mb, oa, ob))
checks.append(("carry-forward regime neutral", carry_ok))

# --- 3. Search lower bound is censoring, not a zero dichotomy -----------------
# A pure-transfer frontier that reaches the lower bound is right-censored:
# the engine only states the critical value lies at or below the bound, not
# that it equals zero.
def critical_gA(reserve, transfer_share):
    lo, hi = SEARCH_LO, 0.25
    for _ in range(14):
        mid = (lo + hi) / 2
        r = run(reserve, pe_policy="full_absorption", tau=0.15, gA=mid,
                phi_max=0.02, transfer_budget_share=transfer_share)
        if np.min(r["C"][1:]) >= FLOOR and r["outcome"] == "smooth":
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

pure_transfer = critical_gA(False, 1.0)
# Bisection cannot distinguish a value at the lower bound from one just below
# it; treat anything within one effective step of the bound as right-censored.
tol = (0.25 - SEARCH_LO) / 2.0 ** 14
censored = pure_transfer <= SEARCH_LO + tol
print("pure-transfer frontier %.5f reaches search lower bound: %s"
      % (pure_transfer, censored))
print("interpreted as censored (<= %.3f), not a point estimate of zero"
      % SEARCH_LO)
checks.append(("lower bound reported as censoring", censored))

ok = all(c[1] for c in checks)
for name, passed in checks:
    print("[%s] %s" % ("PASS" if passed else "FAIL", name))
print("SPENDING MATCH AND RESERVE REGIME", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
