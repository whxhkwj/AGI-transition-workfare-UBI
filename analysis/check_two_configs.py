# -*- coding: utf-8 -*-
"""Verify the two 'standard setting' points quoted in the 'Two configurations' paragraph."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import os, importlib.util
OUT = _SIMDIR
_spec = importlib.util.spec_from_file_location('v6', os.path.join(OUT, 'double_task_sim_v6.py'))
v6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v6)
Params, simulate_v6, welfare = v6.Params, v6.simulate_v6, v6.welfare

def P(**kw):
    base = dict(eta=0.40, mu_TM=0.50)
    base.update(kw)
    return Params(**base)

for pe in (0.30, 0.45):
    p = P(tau=0.16, pE=pe, phi_max=0.65, retrain=0.0, omega=0.05)
    r = simulate_v6(p); w = welfare(r, p)
    print("(0.16, %.2f, 0.65) -> outcome=%-10s minC=%.4f total=%.4f"
          % (pe, r['outcome'], w['minC'], w['total']))
