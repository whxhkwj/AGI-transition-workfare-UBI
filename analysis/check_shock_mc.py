# -*- coding: utf-8 -*-
"""Stochastic-shock Monte Carlo on the calibrated corner (unified engine).

The frontier growth follows an AR(1) process; the realized capability path is
passed through the engine's cap_path hook, so the shock drives both displacement
and the endogenous omega erosion. No in-memory source patching is required.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import os, sys
sys.path.insert(0, _HERE)
sys.path.insert(0, _SIMDIR)
import numpy as np
import double_task_sim_v12_endogenous_omega as v12


def mc(tau, pe, pm, sigma, rho=0.5, n=None, seed=7,
       eta_sd=0.0, mu_sd=0.0, retrain=0.0):
    if n is None:
        n = int(_os.environ.get("MC_DRAWS", "200"))
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p = v12.Params(eta=0.40 + rng.normal(0, eta_sd),
                       mu_TM=0.50 + rng.normal(0, mu_sd),
                       tau=tau, pE=pe, phi_max=pm, retrain=retrain, omega=0.05)
        T = p.T
        eps = rng.normal(0.0, sigma, T)
        dev = np.zeros(T)
        for t in range(1, T):
            dev[t] = rho * dev[t - 1] + eps[t]
        g_path = 0.010 + dev
        g_path[0] = 0.0
        p.cap_path = p.A0 * np.cumprod(1.0 + g_path)
        r = v12.simulate_v12(p)
        w = v12.welfare(r, p)
        out.append((r["outcome"], w["minC"], w["total"]))
    occ = {}
    for o, _, _ in out:
        occ[o] = occ.get(o, 0) + 1
    minc = np.array([m for _, m, _ in out])
    tot = np.array([t for _, _, t in out])
    return dict(n=n, occ=occ, p05=float(np.percentile(minc, 5)),
                p50=float(np.percentile(minc, 50)),
                mean_total=float(tot.mean()), max_total=float(tot.max()))


print("sigma = sd of the innovation in the frontier growth (AR(1), rho=0.5)")
print("%-8s %-7s %-6s %-8s %-8s %-7s %-9s %-9s" %
      ("config", "sigma", "n", "smooth", "crash", "minC05", "minC50", "meanLoss"))
for cfg, (tau, pe, pm) in [("headline", (0.15, 0.45, 0.020)),
                           ("refined", (0.13, 0.44, 0.001))]:
    for sigma in [0.0, 0.005, 0.010, 0.020, 0.030]:
        d = mc(tau, pe, pm, sigma)
        s = d["occ"].get("smooth", 0)
        c = d["occ"].get("crash", 0)
        print("%-8s %-7.3f %-6d %-8d %-8d %-7.4f %-9.4f %-9.2f" %
              (cfg, sigma, d["n"], s, c, d["p05"], d["p50"], d["mean_total"]))

print("\n--- parameter uncertainty on top of sigma=0.010 (eta sd .05, mu_TM sd .05) ---")
for cfg, (tau, pe, pm) in [("headline", (0.15, 0.45, 0.020)),
                           ("refined", (0.13, 0.44, 0.001))]:
    d = mc(tau, pe, pm, 0.010, eta_sd=0.05, mu_sd=0.05, seed=11)
    print("%-8s smooth=%d crash=%d minC05=%.4f meanLoss=%.2f" %
          (cfg, d["occ"].get("smooth", 0), d["occ"].get("crash", 0),
           d["p05"], d["mean_total"]))
