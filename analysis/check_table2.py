# -*- coding: utf-8 -*-
"""核对表 2：CORNER / STANDARD / PURE_UBI 的 minC 全精度与 crash 判定来源。"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import os, importlib.util
import numpy as np

SKEL = _SIMDIR
_spec = importlib.util.spec_from_file_location('v6', os.path.join(SKEL, 'double_task_sim_v6.py'))
v6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v6)

CALIB = dict(eta=0.40, mu_TM=0.50, omega=0.05)
POL = {
    'CORNER':   dict(CALIB, tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0),
    'STANDARD': dict(CALIB, tau=0.16, pE=0.45, phi_max=0.65, retrain=0.0),
    'PURE_UBI': dict(CALIB, tau=0.16, pE=0.00, phi_max=0.65, retrain=0.0),
}

print("%-9s %-11s %-6s %-18s %-9s %-9s %s" % ("policy", "outcome", "crash", "minC(full)", "Cmin", "margin", "sL_end"))
for k, v in POL.items():
    par = v6.Params(**v)
    r = v6.simulate_v6(par)
    minC = float(np.min(r['C'][10:]))
    Cmin = float(r['Cmin'])
    print("%-9s %-11s %-6s %-18.10f %-9.4f %-9.6f %.4f" % (
        k, r['outcome'], r['crash_type'], minC, Cmin, minC - Cmin, r['sL_end']))
