# -*- coding: utf-8 -*-
"""Experiment 1: safety margin of the within-budget EP-over-TM ranking.

The analytic claim (Proposition prop4) is that one fiscal yuan buys 1 unit of
consumption under public employment (EP) but only mu_TM under transfers (TM), so
the within-budget ordering strictly favors EP for every mu_TM < 1 and the
analytic flip sits exactly at mu_TM = 1. This experiment quantifies how far the
calibrated economy sits from that flip inside the *unified* engine:

  * hold the fiscal amount fixed (pe_policy = full_absorption) and compare the
    two allocation extremes, transfer_budget_share = 0 (pure EP) vs = 1 (pure
    TM), over a fine mu_TM grid;
  * record the consumption floor, outcome, welfare loss, peak social stress and
    terminal residual unemployment;
  * locate the numerical mu_TM at which the EP consumption-floor advantage
    vanishes in each scenario and compare it with the analytic boundary 1.

All numbers come directly from the unified engine; nothing is hard-coded.
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

# (tau, gA) scenarios: smooth baseline, calibrated edge, low-tax, high-tax.
SCENARIOS = [
    ("tau.15 gA.01 smooth", 0.15, 0.01),
    ("tau.15 gA.10 edge", 0.15, 0.10),
    ("tau.10 gA.05", 0.10, 0.05),
    ("tau.20 gA.10", 0.20, 0.10),
    ("tau.30 gA.14", 0.30, 0.14),
]
MU_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85,
           0.90, 0.93, 0.95, 0.97, 0.99]
MU_FINE = np.round(np.arange(0.80, 0.999, 0.01), 2)


def run(tau, ga, mu, share):
    par = unified.Params(eta=0.40, mu_TM=mu, phi_max=0.02,
                         pe_policy="full_absorption", tau=tau, gA=ga,
                         transfer_budget_share=share)
    r = unified.simulate_unified(par)
    w = unified.welfare(r, par)
    return dict(minC=float(np.min(r["C"][1:])), outcome=r["outcome"],
                total=w["total"], stress=float(np.max(r["stress"][1:])),
                urem=float(r["Urem"][-1]))


def critical_mu(tau, ga):
    """Smallest mu_TM at which pure EP no longer beats pure TM on minC."""
    flip = None
    for mu in MU_FINE:
        ep = run(tau, ga, float(mu), 0.0)
        tm = run(tau, ga, float(mu), 1.0)
        if ep["minC"] <= tm["minC"] + 1e-9:
            flip = float(mu)
            break
    return flip


print("=" * 88)
print("Experiment 1: within-budget EP vs TM, safety margin over mu_TM (unified engine)")
print("=" * 88)

checks = []
for label, tau, ga in SCENARIOS:
    print("\n### %s" % label)
    print("  mu_TM  EP_minC TM_minC  d_minC   EP_out    TM_out   EP_loss  TM_loss")
    violations = 0
    anchor_gap = None
    for mu in MU_GRID:
        ep = run(tau, ga, mu, 0.0)
        tm = run(tau, ga, mu, 1.0)
        d = ep["minC"] - tm["minC"]
        if abs(mu - 0.50) < 1e-9:
            anchor_gap = d
        if ep["minC"] < tm["minC"] - 1e-9:
            violations += 1
        print("  %5.2f  %.4f  %.4f  %+.4f  %-8s  %-8s  %8.2f  %8.2f" % (
            mu, ep["minC"], tm["minC"], d, ep["outcome"], tm["outcome"],
            ep["total"], tm["total"]))
    flip = critical_mu(tau, ga)
    print("  -> numerical flip mu_TM* = %s (analytic boundary = 1.00)"
          % ("none below 0.999" if flip is None else "%.2f" % flip))
    # Ranking must hold at every mu_TM < 1 on this grid.
    checks.append(("%s: EP minC >= TM at all mu<1" % label, violations == 0))
    # At the calibrated anchor the margin must be strictly positive.
    checks.append(("%s: positive margin at mu=0.50" % label,
                   anchor_gap is not None and anchor_gap > 1e-9))
    # Any numerical flip must sit close to the analytic boundary (large margin).
    checks.append(("%s: flip absent or > 0.90" % label,
                   flip is None or flip > 0.90))

print("\n" + "=" * 88)
print("checks against the manuscript")
bad = 0
for name, ok in checks:
    print("  %-52s %s" % (name, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
print("\n%d/%d checks passed" % (len(checks) - bad, len(checks)))
raise SystemExit(1 if bad else 0)
