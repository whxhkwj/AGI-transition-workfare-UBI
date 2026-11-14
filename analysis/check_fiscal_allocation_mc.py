# -*- coding: utf-8 -*-
"""Experiment 3: Monte Carlo robustness of the same-budget allocation.

Experiments 1-2 are deterministic grid facts. This experiment asks whether the
feasibility separation survives uncertainty in the fiscal parameters. On each
draw the fiscal amount is held fixed (pe_policy = full_absorption) and the two
allocation extremes (pure EP, pure TM) are run under the same drawn parameters:

  * mu_TM    ~ Beta(5, 5), mean 0.50 (transfer consumption MPC);
  * alpha_s  ~ truncated Normal(0.60, 0.15) on [0, 1] (social coefficient);
  * phi_max  ~ Uniform(0.02, 0.30) (transfer ceiling).

The baseline scenario is tau = 0.15, gA = 0.01. We report the share of draws in
which EP is smooth while TM crashes, the share in which the EP consumption floor
is at least the TM floor, with exact Clopper-Pearson 95% intervals, and the
welfare ordering restricted to the subset where both are feasible.
All numbers come directly from the unified engine; nothing is hard-coded.
"""
import os
import importlib.util
import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIMDIR = os.path.join(os.path.dirname(_HERE), "simulation")

_spec = importlib.util.spec_from_file_location(
    "unified", os.path.join(_SIMDIR, "double_task_sim_unified.py"))
unified = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(unified)

N_DRAW = 400
SEED = 20261114


def cp_interval(k, n):
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def run(mu, alpha, pm, share):
    par = unified.Params(eta=0.40, mu_TM=mu, alpha_s=alpha, phi_max=pm,
                         pe_policy="full_absorption", tau=0.15, gA=0.01,
                         transfer_budget_share=share)
    r = unified.simulate_unified(par)
    w = unified.welfare(r, par)
    return dict(outcome=r["outcome"], minC=float(np.min(r["C"][1:])),
                total=w["total"])


rng = np.random.default_rng(SEED)
mu_draws = rng.beta(5, 5, N_DRAW)
a_lo, a_hi = (0 - 0.60) / 0.15, (1 - 0.60) / 0.15
alpha_draws = stats.truncnorm.rvs(a_lo, a_hi, loc=0.60, scale=0.15,
                                  size=N_DRAW, random_state=rng)
phi_draws = rng.uniform(0.02, 0.30, N_DRAW)

print("=" * 84)
print("Experiment 3: Monte Carlo same-budget allocation (%d draws, seed %d)"
      % (N_DRAW, SEED))
print("=" * 84)

n_sep = 0          # EP smooth while TM crashes
n_minc = 0         # EP floor >= TM floor
both_feasible = 0
ep_welfare_wins = 0
for i in range(N_DRAW):
    ep = run(mu_draws[i], alpha_draws[i], phi_draws[i], 0.0)
    tm = run(mu_draws[i], alpha_draws[i], phi_draws[i], 1.0)
    if ep["outcome"] == "smooth" and tm["outcome"] == "crash":
        n_sep += 1
    if ep["minC"] >= tm["minC"] - 1e-12:
        n_minc += 1
    if ep["outcome"] == "smooth" and tm["outcome"] == "smooth":
        both_feasible += 1
        if ep["total"] <= tm["total"] + 1e-9:
            ep_welfare_wins += 1

sep_lo, sep_hi = cp_interval(n_sep, N_DRAW)
mc_lo, mc_hi = cp_interval(n_minc, N_DRAW)
print("\nEP smooth while TM crashes: %d/%d = %.1f%%  CP95 [%.3f, %.3f]"
      % (n_sep, N_DRAW, 100 * n_sep / N_DRAW, sep_lo, sep_hi))
print("EP minC >= TM minC:         %d/%d = %.1f%%  CP95 [%.3f, %.3f]"
      % (n_minc, N_DRAW, 100 * n_minc / N_DRAW, mc_lo, mc_hi))
print("draws where both feasible:  %d/%d (welfare EP lower in %d)"
      % (both_feasible, N_DRAW, ep_welfare_wins))

checks = [
    ("EP floor at least TM floor in every draw", n_minc == N_DRAW),
    ("feasibility separation exceeds 90%", n_sep / N_DRAW > 0.90),
]
print("\nchecks against the manuscript")
bad = 0
for name, ok in checks:
    print("  %-52s %s" % (name, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
print("\n%d/%d checks passed" % (len(checks) - bad, len(checks)))
raise SystemExit(1 if bad else 0)
