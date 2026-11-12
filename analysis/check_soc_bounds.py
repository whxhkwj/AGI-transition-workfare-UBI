# -*- coding: utf-8 -*-
"""Verify the J-sweep claim for the pure transfer and the two analytic boundaries."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import os, sys, importlib.util
import numpy as np

ENGINE_DIR = _SIMDIR
_spec = importlib.util.spec_from_file_location('v6', os.path.join(ENGINE_DIR, 'double_task_sim_v6.py'))
v6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v6)

CALIB = dict(eta=0.40, mu_TM=0.50, omega=0.05)
CORNER = dict(CALIB, tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0)
PURE_TRANSFER = dict(CALIB, tau=0.16, pE=0.00, phi_max=0.65, retrain=0.0)

print("=== J sweep: pure transfer (claim: triggers for all J in [3,30]) ===")
for j in [3, 5, 10, 15, 20, 25, 30]:
    par = v6.Params(**dict(PURE_TRANSFER, soc_period=j))
    r = v6.simulate_v6(par)
    print(f"  J={j:3d}  social_crash={str(r['social_crash']):5s} "
          f"stress_peak={np.max(r['stress']):.3f}")

print("\n=== J sweep: corner (claim: never triggers) ===")
for j in [3, 10, 30]:
    par = v6.Params(**dict(CORNER, soc_period=j))
    r = v6.simulate_v6(par)
    print(f"  J={j:3d}  social_crash={str(r['social_crash']):5s}")

print("\n=== boundary probes: corner critical varsigma_th ~ 0.33 ===")
for th in [0.32, 0.33, 0.34]:
    r = v6.simulate_v6(v6.Params(**dict(CORNER, stress_thresh=th)))
    print(f"  th={th:.2f}  social_crash={r['social_crash']}")

print("\n=== boundary probes: corner critical alpha_s ~ 0.909 ===")
for a in [0.88, 0.90, 0.92, 0.95]:
    r = v6.simulate_v6(v6.Params(**dict(CORNER, alpha_s=a)))
    print(f"  alpha_s={a:.2f}  social_crash={r['social_crash']}  stress_peak={np.max(r['stress']):.3f}")

print("\n=== boundary probes: pure transfer critical varsigma_th ~ 0.60 ===")
for th in [0.55, 0.60, 0.65]:
    r = v6.simulate_v6(v6.Params(**dict(PURE_TRANSFER, stress_thresh=th)))
    print(f"  th={th:.2f}  social_crash={r['social_crash']}  stress_peak={np.max(r['stress']):.3f}")
