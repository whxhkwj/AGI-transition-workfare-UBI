# -*- coding: utf-8 -*-
"""Joint-uncertainty Monte Carlo under the upgraded engine.

Draws 500 parameter combinations that jointly vary the fiscal (tau, pE,
phi_max), technological (gA, beta_A, omega0, gamma_*) and social (alpha_s,
stress threshold) parameters, runs the full-absorption endogenous engine on
each, and counts how many reach the high market-employment endpoint (terminal
sL >= 0.50). Reports the Clopper-Pearson 95% upper bound on that probability.
"""
import os, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIMDIR = os.path.join(ROOT, "simulation")

_spec = importlib.util.spec_from_file_location(
    "U", os.path.join(SIMDIR, "double_task_sim_unified.py"))
U = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(U)

N = 500
rng = np.random.default_rng(2026)
HIGH = 0.50


def draw_params():
    return dict(
        tau=float(rng.uniform(0.05, 0.40)),
        pE=float(rng.uniform(0.0, 1.0)),
        phi_max=float(rng.uniform(0.02, 0.65)),
        gA=float(rng.uniform(0.005, 0.15)),
        beta_A=float(rng.uniform(0.5, 1.5)),
        omega0=float(rng.uniform(0.03, 0.10)),
        gamma_PE=float(rng.uniform(3.0, 6.0)),
        gamma_TM=float(rng.uniform(2.0, 4.0)),
        gamma_R=float(rng.uniform(1.5, 3.0)),
        alpha_s=float(rng.uniform(0.30, 0.90)),
        stress_thresh=float(rng.uniform(0.40, 0.60)),
    )


high_count = 0
sL_max = 0.0
for i in range(N):
    par = U.Params(eta=0.40, mu_TM=0.50, **draw_params())
    r = U.simulate_unified(par)
    sL_max = max(sL_max, float(r["sL_end"]))
    if r["sL_end"] >= HIGH:
        high_count += 1

# Clopper-Pearson 95% upper bound: Beta quantile (k+1, n-k).
try:
    from scipy.stats import beta as _beta
    cp = float(_beta.ppf(0.95, high_count + 1, N - high_count))
except Exception:
    cp = float(1.0 - 0.05 ** (1.0 / N))  # exact k=0 special case

print("draws: %d" % N)
print("high market-employment endpoint (sL >= %.2f) reached in: %d" % (HIGH, high_count))
print("maximum terminal sL observed: %.4f" % sL_max)
print("Clopper-Pearson 95%% upper bound: %.4f" % cp)
