# -*- coding: utf-8 -*-
"""Cross-check every numeric claim in the Calibration chapter and the tables
against a fresh run of the v6 engine (exogenous omega=0.05), plus print the
v12 numbers used in Section 4."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import os, sys, importlib.util
import numpy as np

ENGINE_DIR = _SIMDIR
sys.path.insert(0, ENGINE_DIR)

_spec = importlib.util.spec_from_file_location('v6', os.path.join(ENGINE_DIR, 'double_task_sim_v6.py'))
v6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v6)

CALIB = dict(eta=0.40, mu_TM=0.50, omega=0.05)
POL = {
    'CORNER':   dict(CALIB, tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0),
    'STANDARD': dict(CALIB, tau=0.16, pE=0.45, phi_max=0.65, retrain=0.0),
    'PURE_TRANSFER': dict(CALIB, tau=0.16, pE=0.00, phi_max=0.65, retrain=0.0),
}

print("=== tab:socbase  (v6 exogenous omega, T=80) ===")
print(f"{'policy':10s} {'outcome':11s} {'crash':5s} {'minC':6s} {'peakStress':10s} {'social':5s}")
for k, p in POL.items():
    par = v6.Params(**p)
    r = v6.simulate_v6(par)
    print(f"{k:10s} {r['outcome']:11s} {str(r['crash_type']):5s} "
          f"{np.min(r['C'][10:]):.3f}  {np.max(r['stress']):.3f}      "
          f"{r['social_crash']}")

print("\n=== calibration table claims (P1-P6 reproduction) ===")
par = v6.Params(**POL['CORNER'])
r = v6.simulate_v6(par)
print("corner min C (t>=10):", round(float(np.min(r['C'][10:])), 3))
print("corner T:", len(r['C']))
par2 = v6.Params(**POL['STANDARD'])
r2 = v6.simulate_v6(par2)
print("standard min C:", round(float(np.min(r2['C'][10:])), 3), " outcome:", r2['outcome'])
