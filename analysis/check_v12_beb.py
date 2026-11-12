# -*- coding: utf-8 -*-
"""BEB verification for v12 endogenous omega using the dynamic threshold.
A real BEB requires effective capture omega*skill to reach EFF_CRIT; the static
value omega_c = kappa_I/q + s_floor = 0.5667 is the skill=1 special case."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import sys
sys.path.insert(0, _HERE)
sys.path.insert(0, _SIMDIR)
import numpy as np
import double_task_sim_v12_endogenous_omega as v12
import dynamic_threshold as dt

print(f"static threshold (skill=1) omega_c = {dt.EFF_CRIT:.6f}\n")

configs = {
    'no policy': dict(tau=0.05, pE=0.0, phi_max=0.15, retrain=0.0),
    'corner':    dict(tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0),
    'strong':    dict(tau=0.15, pE=0.80, phi_max=0.65, retrain=0.8),
}

print("config        outcome      minC      real BEB crossing (eff=omega*skill)")
for name, cfg in configs.items():
    par = v12.Params(eta=0.40, mu_TM=0.50, T=200, **cfg)
    r = v12.simulate_v12(par)
    real_t = dt.first_real_crossing(r['omega'], r['skill'])
    minC = float(np.min(r['C'][10:]))
    print(f"{name:12s} {r['outcome']:10s} {minC:8.4f}   t={real_t}")

print("\n--- omega trajectory (every 20 periods) ---")
for name, cfg in configs.items():
    par = v12.Params(eta=0.40, mu_TM=0.50, T=200, **cfg)
    r = v12.simulate_v12(par)
    traj = " ".join(f"{r['omega'][j]:.3f}" for j in range(0, 200, 20))
    print(f"{name:12s}: {traj}")

print("\n=== sweep pE (public employment), retrain=0, phi_max=0.30 ===")
print(" pE     outcome      minC    real BEB")
for pE in [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]:
    par = v12.Params(eta=0.40, mu_TM=0.50, T=200, tau=0.15, pE=pE,
                     phi_max=0.30, retrain=0.0)
    r = v12.simulate_v12(par)
    print(f"{pE:5.2f} {r['outcome']:10s} {np.min(r['C'][10:]):.4f}   "
          f"t={dt.first_real_crossing(r['omega'], r['skill'])}")

print("\n=== sweep retrain (skill retention) at fixed pE=0.45 ===")
print(" retrain  outcome      minC    real BEB")
for rt in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    par = v12.Params(eta=0.40, mu_TM=0.50, T=200, tau=0.15, pE=0.45,
                     phi_max=0.30, retrain=rt)
    r = v12.simulate_v12(par)
    print(f"{rt:6.2f} {r['outcome']:10s} {np.min(r['C'][10:]):.4f}   "
          f"t={dt.first_real_crossing(r['omega'], r['skill'])}")
