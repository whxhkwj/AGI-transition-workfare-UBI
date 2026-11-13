# -*- coding: utf-8 -*-
"""M1 feedback robustness: when social pressure feeds back on output (and hence
on the capital income and tax base), does the workfare-over-transfer ranking
survive? Uses the engine's stress_fb channel; stress_fb=0 reproduces the
baseline path exactly. Deterministic, fixed calibration, fully reproducible.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import sys, importlib.util
import numpy as np
sys.path.insert(0, _SIMDIR)
_spec = importlib.util.spec_from_file_location('v6', _os.path.join(_SIMDIR, 'double_task_sim_v6.py'))
v6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v6)

CALIB = dict(eta=0.40, mu_TM=0.50, omega=0.05)
REGIMES = {
    'corner':        dict(CALIB, tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0),
    'standard':      dict(CALIB, tau=0.16, pE=0.45, phi_max=0.65, retrain=0.0),
    'pure transfer': dict(CALIB, tau=0.16, pE=0.00, phi_max=0.65, retrain=0.0),
}
FB_LEVELS = [0.0, 0.25, 0.50, 1.0]

print("output multiplier from pressure: GDP_t *= 1 - stress_fb * stress_t")
print("%-14s %-10s %-10s %-10s %-9s %-9s" %
      ("regime", "stress_fb", "outcome", "crash type", "min C", "stress pk"))
for name, cfg in REGIMES.items():
    for fb in FB_LEVELS:
        par = v6.Params(**dict(cfg, stress_fb=fb))
        r = v6.simulate_v6(par)
        minC = float(np.min(r['C'][10:]))
        spk = float(np.max(r['stress']))
        print("%-14s %-10.2f %-10s %-10s %-9.4f %-9.4f" %
              (name, fb, r['outcome'], r['crash_type'], minC, spk))
    print()

print("=== ranking under feedback (corner vs pure transfer) ===")
for fb in FB_LEVELS:
    rc = v6.simulate_v6(v6.Params(**dict(REGIMES['corner'], stress_fb=fb)))
    rp = v6.simulate_v6(v6.Params(**dict(REGIMES['pure transfer'], stress_fb=fb)))
    corner_ok = rc['outcome'] == 'smooth'
    transfer_crash = rp['outcome'] == 'crash'
    verdict = "workfare smooth, transfer crashes" if (corner_ok and transfer_crash) else "CHECK"
    print("stress_fb=%.2f: %s" % (fb, verdict))
