# -*- coding: utf-8 -*-
"""
Population-accounting verification for the public-employment budget scaling.

This reproduces the bookkeeping behind the period-5 residual-unemployment figure
and asserts, for every period of the baseline corner configuration:

  1. The unemployed remainder is computed from the *financed* (actual) public-job
     headcount, not the planned headcount:
         Urem_t = max(0, u_t L^S - EP_t)
  2. The population partition closes to machine precision:
         L^D_t + EP_t + Urem_t = L^S.

It also reports the planned-vs-actual residual in the early periods, which is the
mechanism an earlier implementation got wrong by holding the residual at its
planned value after the budget had scaled the public-job headcount down.
"""
import os, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "simulation", "double_task_sim_unified.py")
spec = importlib.util.spec_from_file_location("unified", SIM)
unified = importlib.util.module_from_spec(spec); spec.loader.exec_module(unified)

# Baseline corner (exogenous reinstatement calibration used in the audit).
par = unified.Params(tau=0.15, pE=0.45, phi_max=0.02,
                     omega_mode="exogenous", omega=0.05)
r = unified.simulate_unified(par)
T = par.T
LD, EP, Urem = r["LD"], r["EP"], r["Urem"]
u = 1.0 - LD / par.LS

max_partition_gap = 0.0
max_residual_gap = 0.0
plan_residual = np.zeros(T)
for t in range(T):
    uLS = u[t] * par.LS
    EP_plan = min(par.pE * uLS, uLS)
    plan_residual[t] = max(0.0, uLS - EP_plan)

    expect_urem = max(0.0, uLS - EP[t])
    max_residual_gap = max(max_residual_gap, abs(Urem[t] - expect_urem))
    partition = LD[t] + EP[t] + Urem[t]
    max_partition_gap = max(max_partition_gap, abs(partition - par.LS))

    assert abs(Urem[t] - expect_urem) < 1e-12
    assert abs(partition - par.LS) < 1e-12

# Assertions: exact financed-headcount residual and a closed population partition.
assert max_residual_gap < 1e-12
assert max_partition_gap < 1e-12

print("population partition max |LD+EP+Urem-LS| = %.3e" % max_partition_gap)
print("residual max |Urem-max(0,uLS-EP)|         = %.3e" % max_residual_gap)
print()
print("period-5 bookkeeping (baseline corner):")
print("  planned residual (ignores budget scaling) : %.5f" % plan_residual[5])
print("  financed actual residual (reported)        : %.5f" % Urem[5])
print("  actual public-job headcount                : %.5f" % EP[5])
print()
print("ALL POPULATION-ACCOUNTING CHECKS PASSED")
