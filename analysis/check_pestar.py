# -*- coding: utf-8 -*-
"""核对 p_E*=0.4388 在外生(ω=0.05)与内生(v12)两种校准下是否同一阈值。"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import os, importlib.util
import numpy as np

SKEL = _SIMDIR
def load(name, f):
    sp = importlib.util.spec_from_file_location(name, os.path.join(SKEL, f))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

v6 = load('v6', 'double_task_sim_v6.py')
v12 = load('v12', 'double_task_sim_v12_endogenous_omega.py')

for eng, mod, sim in [("exog", v6, v6.simulate_v6), ("endog", v12, v12.simulate_v12)]:
    print("=== %s  tau=0.15 phi=0.02, pE sweep ===" % eng)
    print("%-7s %-12s %-6s %s" % ("pE", "outcome", "crash", "minC"))
    first_smooth = None
    for pe in np.arange(0.40, 0.501, 0.005):
        par = mod.Params(eta=0.40, mu_TM=0.50, omega=0.05, tau=0.15,
                         pE=float(pe), phi_max=0.02, retrain=0.0)
        r = sim(par)
        ok = (r['outcome'] == 'smooth') or (r['outcome'] == 'not_reached' and r['crash_type'] == 'none')
        if ok and first_smooth is None:
            first_smooth = pe
        print("%-7.3f %-12s %-6s %.5f  %s" % (pe, r['outcome'], r['crash_type'],
                                             float(np.min(r['C'][10:])), "<- first non-crash" if ok and abs(pe-first_smooth)<1e-9 else ""))
    print("first pE with no crash: %s\n" % first_smooth)
