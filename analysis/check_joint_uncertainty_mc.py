# -*- coding: utf-8 -*-
"""Extended joint-uncertainty Monte Carlo (C2).

Experiment 3 varies three fiscal parameters jointly. This check widens the
joint distribution to the fiscal, technological and social parameters and
asks whether the structural claims survive when all of them move together:

  * mu_TM   ~ Beta(5, 5), mean 0.50      (transfer MPC)
  * alpha_s ~ TruncNormal(0.60, 0.15)    (social coefficient)
  * phi_max ~ Uniform(0.02, 0.30)        (transfer ceiling)
  * eta     ~ Uniform(0.20, 0.60)        (tax growth inhibition)
  * wP      ~ Uniform(0.80, 0.90)        (public-employment wage ratio)
  * gA      ~ Uniform(0.01, 0.20)        (automation speed)

The full-absorption policy is held fixed. We report the endpoint-class
distribution, the minimum-consumption distribution, and the share of draws in
which a high market-employment endpoint remains unreachable, with exact
Clopper-Pearson intervals. All values come from the unified engine.
"""
import os, sys
import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from simulation.double_task_sim_unified import Params, simulate_unified

N_DRAW = 500
SEED = 20261114
HIGH_SHARE = 0.50


def cp_interval(k, n):
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return lo, hi


rng = np.random.default_rng(SEED)
mu_d = rng.beta(5, 5, N_DRAW)
a, b = (0 - 0.60) / 0.15, (1 - 0.60) / 0.15
alpha_d = stats.truncnorm.rvs(a, b, loc=0.60, scale=0.15, size=N_DRAW,
                              random_state=rng)
phi_d = rng.uniform(0.02, 0.30, N_DRAW)
eta_d = rng.uniform(0.20, 0.60, N_DRAW)
wp_d = rng.uniform(0.80, 0.90, N_DRAW)
ga_d = rng.uniform(0.01, 0.20, N_DRAW)

classes = {"smooth": 0, "crash": 0, "not_reached": 0}
minc_list = []
high_reachable = 0
for i in range(N_DRAW):
    par = Params(pe_policy="full_absorption", tau=0.15, mu_TM=mu_d[i],
                 alpha_s=alpha_d[i], phi_max=phi_d[i], eta=eta_d[i],
                 wP=wp_d[i], gA=ga_d[i])
    r = simulate_unified(par)
    key = r["outcome"]
    classes[key] = classes.get(key, 0) + 1
    minc_list.append(float(np.min(r["C"][1:])))
    if r["sL_end"] >= HIGH_SHARE:
        high_reachable += 1

minc_arr = np.array(minc_list)
print("joint-uncertainty MC: %d draws" % N_DRAW)
print("endpoint classes:", {k: "%d (%.1f%%)" % (v, 100 * v / N_DRAW)
                            for k, v in classes.items()})
print("minC quantiles 5/50/95: %.3f / %.3f / %.3f"
      % (np.quantile(minc_arr, 0.05), np.median(minc_arr),
         np.quantile(minc_arr, 0.95)))
print("share with high market-employment endpoint reachable: %.3f"
      % (high_reachable / N_DRAW))
lo, hi = cp_interval(high_reachable, N_DRAW)
print("Clopper-Pearson 95%% on that share: [%.4f, %.4f]" % (lo, hi))

checks = [
    ("high market-employment endpoint never reached", high_reachable == 0),
    ("both smooth and crash classes observed",
     classes["smooth"] > 0 and classes["crash"] > 0),
]
ok = all(c[1] for c in checks)
for name, passed in checks:
    print("[%s] %s" % ("PASS" if passed else "FAIL", name))
print("JOINT UNCERTAINTY MC", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
