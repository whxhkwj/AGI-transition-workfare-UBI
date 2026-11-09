# -*- coding: utf-8 -*-
"""Smoke test + BEB verification for v12 endogenous omega.
Check: (1) no-policy omega stays at omega0; (2) strong policy pushes omega past
the theoretical BEB threshold omega_c = kappa_I/q + s_floor = 0.5667;
(3) sweep policy intensity to locate the crossing."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import sys, os
sys.path.insert(0, _SIMDIR)
import numpy as np
import double_task_sim_v12_endogenous_omega as v12

OMEGA_C = 0.020 / 0.050 + (1 - 1 / 1.20)     # kappa_I/q + s_floor
print(f"theoretical BEB threshold omega_c = {OMEGA_C:.6f}\n")

configs = {
    'no policy': dict(tau=0.05, pE=0.0, phi_max=0.15, retrain=0.0),
    'corner':    dict(tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0),
    'strong':    dict(tau=0.15, pE=0.80, phi_max=0.65, retrain=0.8),
}

print("config        outcome      omega_end   minC      omega crosses omega_c?")
for name, cfg in configs.items():
    par = v12.Params(eta=0.40, mu_TM=0.50, **cfg)
    r = v12.simulate_v12(par)
    crosses = np.max(r['omega']) >= OMEGA_C
    minC = float(np.min(r['C'][10:]))
    print(f"{name:12s} {r['outcome']:10s} {r['omega_end']:9.4f} {minC:8.4f}   {crosses}")

print("\n--- omega trajectory (every 8 periods) ---")
for name, cfg in configs.items():
    par = v12.Params(eta=0.40, mu_TM=0.50, **cfg)
    r = v12.simulate_v12(par)
    traj = " ".join(f"{r['omega'][j]:.3f}" for j in range(0, 80, 8))
    print(f"{name:12s}: {traj}")

print("\n=== sweep pE (public employment) at fixed tau=0.15, retrain=0 ===")
print(" pE     omega_end   max(omega)  outcome      minC")
for pE in [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]:
    par = v12.Params(eta=0.40, mu_TM=0.50, tau=0.15, pE=pE,
                     phi_max=0.30, retrain=0.0)
    r = v12.simulate_v12(par)
    minC = float(np.min(r['C'][10:]))
    print(f"{pE:5.2f} {r['omega_end']:9.4f} {np.max(r['omega']):10.4f} "
          f"{r['outcome']:10s} {minC:8.4f}")

print("\n=== sweep retrain (skill retention) at fixed pE=0.45 ===")
print(" retrain  omega_end   max(omega)  outcome      minC")
for rt in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    par = v12.Params(eta=0.40, mu_TM=0.50, tau=0.15, pE=0.45,
                     phi_max=0.30, retrain=rt)
    r = v12.simulate_v12(par)
    minC = float(np.min(r['C'][10:]))
    print(f"{rt:6.2f} {r['omega_end']:9.4f} {np.max(r['omega']):10.4f} "
          f"{r['outcome']:10s} {minC:8.4f}")
