# -*- coding: utf-8 -*-
"""核对 §5.1 L500 括号：0.309 与 0.305 的 0.004 差额到底归因于 φ 上限，还是 ω 外生/内生。
两套引擎 × 三个 φ 上限，同一 corner 坐标 (tau=0.15, pE=0.45)。"""
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

BASE = dict(eta=0.40, mu_TM=0.50, omega=0.05, tau=0.15, pE=0.45, retrain=0.0)
PHIS = [0.02, 0.001, 0.0]

print("%-12s %-8s %-11s %-6s %-12s" % ("engine", "phi_max", "outcome", "crash", "minC"))
for eng, mod, sim in [("exog(v6)", v6, v6.simulate_v6), ("endog(v12)", v12, v12.simulate_v12)]:
    for ph in PHIS:
        kw = dict(BASE); kw['phi_max'] = ph
        try:
            par = mod.Params(**kw)
            r = sim(par)
            print("%-12s %-8.3f %-11s %-6s %-12.6f" % (
                eng, ph, r['outcome'], r['crash_type'], float(np.min(r['C'][10:]))))
        except Exception as e:
            print("%-12s %-8.3f ERROR %s" % (eng, ph, e))
