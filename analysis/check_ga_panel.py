# -*- coding: utf-8 -*-
"""核验 GA 两配置在外生 omega=0.05 下的 minC，看是否等于正文 headline 0.309。"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import sys
import numpy as np

SIM = _SIMDIR
sys.path.insert(0, SIM)
import double_task_sim_v12_endogenous_omega as v12

cases = {
    "corner endogenous": dict(tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0),
    "corner exogenous w=0.05": dict(tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0, omega=0.05),
    "nopolicy exogenous w=0.05": dict(tau=0.05, pE=0.0, phi_max=0.15, retrain=0.0, omega=0.05),
}
for name, kw in cases.items():
    try:
        rr = v12.simulate_v12(v12.Params(eta=0.40, mu_TM=0.50, T=80, **kw))
        print("%-26s outcome=%-12s minC=%.6f  sL_end=%.4f" % (name, rr["outcome"], float(np.min(rr["C"]), ), rr["sL"][-1]))
    except Exception as e:
        print("%-26s FAILED: %s" % (name, e))
