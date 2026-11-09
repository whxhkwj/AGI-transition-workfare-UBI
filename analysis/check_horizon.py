# -*- coding: utf-8 -*-
"""核对 §4.6 长视界句（L477）：T=200 三个配置的 minC / sL_end / omega_end，内生 ω 引擎。"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import os, importlib.util
import numpy as np

SKEL = _SIMDIR
sp = importlib.util.spec_from_file_location('v12', os.path.join(SKEL, 'double_task_sim_v12_endogenous_omega.py'))
v12 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v12)

CASES = [
    ("no policy      ", dict(eta=0.40, mu_TM=0.50, omega=0.05, tau=0.05, pE=0.0, phi_max=0.15, retrain=0.0)),
    ("corner phi->0  ", dict(eta=0.40, mu_TM=0.50, omega=0.05, tau=0.15, pE=0.45, phi_max=0.0, retrain=0.0)),
    ("strong floor   ", dict(eta=0.40, mu_TM=0.50, omega=0.05, tau=0.15, pE=0.80, phi_max=0.65, retrain=0.8)),
]
for T in (80, 200):
    print("=== T=%d (endog v12) ===" % T)
    print("%-16s %-11s %-6s %-10s %-10s %s" % ("config", "outcome", "crash", "minC", "sL_end", "om_end"))
    for name, kw in CASES:
        kw = dict(kw); kw['T'] = T
        par = v12.Params(**kw)
        r = v12.simulate_v12(par)
        om = r.get('omega', [None])[-1]
        print("%-16s %-11s %-6s %-10.5f %-10.5f %s" % (
            name, r['outcome'], r['crash_type'], float(np.min(r['C'][10:])), r['sL_end'],
            ("%.3f" % om) if om is not None else "-"))
    print("")
