# -*- coding: utf-8 -*-
"""Experiment 2: same-budget allocation across scenarios and outcome dimensions.

The current same-budget comparison (check_two_gates) uses one parameter setting
and one outcome dimension (the consumption floor minC). Proposition prop4 is
explicitly a marginal fixed-budget comparison rather than a full welfare
ranking. This experiment broadens the evidence on both axes:

  * hold the fiscal amount fixed (pe_policy = full_absorption) and sweep the
    transfer_budget_share from 0 (pure EP) to 1 (pure TM) on a fine grid;
  * repeat over five (tau, gA) scenarios;
  * record five outcome dimensions: consumption floor, welfare loss (with its
    gap/cost/deadweight split), peak social stress, terminal residual
    unemployment, and the total unfinanced gap.

The script reports, per dimension, the direction of the EP-to-TM ordering and
where it reverses, instead of assuming one direction for every dimension.
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

SCENARIOS = [
    ("tau.15 gA.01 smooth", 0.15, 0.01),
    ("tau.15 gA.10 edge", 0.15, 0.10),
    ("tau.10 gA.05", 0.10, 0.05),
    ("tau.20 gA.10", 0.20, 0.10),
    ("tau.30 gA.14", 0.30, 0.14),
]
SHARES = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def run(tau, ga, share):
    par = unified.Params(eta=0.40, mu_TM=0.50, phi_max=0.02,
                         pe_policy="full_absorption", tau=tau, gA=ga,
                         transfer_budget_share=float(share))
    r = unified.simulate_unified(par)
    w = unified.welfare(r, par)
    return dict(
        minC=float(np.min(r["C"][1:])), total=w["total"], gap=w["gap"],
        cost=w["cost"], dwl=w["dwl"],
        stress=float(np.max(r["stress"][1:])), social=bool(r["social_crash"]),
        urem=float(r["Urem"][-1]),
        unfin=float(np.sum(r["unfin_gap"][1:])), outcome=r["outcome"])


def nonincreasing(seq):
    return all(seq[i] >= seq[i + 1] - 1e-9 for i in range(len(seq) - 1))


def nondecreasing(seq):
    return all(seq[i] <= seq[i + 1] + 1e-9 for i in range(len(seq) - 1))


print("=" * 100)
print("Experiment 2: same-budget allocation, multi-scenario multi-dimension (unified engine)")
print("=" * 100)

# Dimension -> (key, "good" direction for the EP corner relative to TM).
checks = []
welfare_ep_wins = 0
for label, tau, ga in SCENARIOS:
    rows = [run(tau, ga, s) for s in SHARES]
    print("\n### %s" % label)
    print(" share  minC   welfare   gap     cost    dwl    stress  social  Urem_end  unfin")
    for s, d in zip(SHARES, rows):
        print("  %4.2f  %.4f  %7.2f  %6.2f  %6.2f  %5.2f  %.4f  %-6s  %.4f  %.4f" % (
            s, d["minC"], d["total"], d["gap"], d["cost"], d["dwl"],
            d["stress"], "yes" if d["social"] else "no", d["urem"], d["unfin"]))

    minc = [d["minC"] for d in rows]
    stress = [d["stress"] for d in rows]
    urem = [d["urem"] for d in rows]
    unfin = [d["unfin"] for d in rows]
    ep, tm = rows[0], rows[-1]
    print("  EP(corner) vs TM(corner): minC %+.4f | welfare %+.2f | stress %+.4f | "
          "Urem %+.4f | unfin %+.4f" % (
              ep["minC"] - tm["minC"], ep["total"] - tm["total"],
              ep["stress"] - tm["stress"], ep["urem"] - tm["urem"],
              ep["unfin"] - tm["unfin"]))
    if ep["total"] <= tm["total"] + 1e-9:
        welfare_ep_wins += 1

    # Consumption floor, social protection and job absorption must favor EP and
    # be monotone in the transfer share.
    checks.append(("%s: minC nonincreasing, EP>=TM" % label,
                   nonincreasing(minc) and ep["minC"] >= tm["minC"] - 1e-9))
    checks.append(("%s: stress nondecreasing, EP<=TM" % label,
                   nondecreasing(stress) and ep["stress"] <= tm["stress"] + 1e-9))
    checks.append(("%s: Urem nondecreasing, EP<=TM" % label,
                   nondecreasing(urem) and ep["urem"] <= tm["urem"] + 1e-9))
    checks.append(("%s: unfin gap nondecreasing" % label, nondecreasing(unfin)))

print("\n" + "=" * 100)
print("welfare-total ordering by scenario: EP lower loss in %d/%d; TM lower loss in %d/%d"
      % (welfare_ep_wins, len(SCENARIOS),
         len(SCENARIOS) - welfare_ep_wins, len(SCENARIOS)))
print("checks against the manuscript (dimensions Prop 4 directly addresses)")
bad = 0
for name, ok in checks:
    print("  %-52s %s" % (name, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
print("\n%d/%d checks passed" % (len(checks) - bad, len(checks)))
raise SystemExit(1 if bad else 0)
