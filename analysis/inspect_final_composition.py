# -*- coding: utf-8 -*-
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "simulation"))
import numpy as np
import double_task_sim_v12_endogenous_omega as v12

configs = {
    "no policy": dict(tau=0.05, pE=0.0, phi_max=0.15, retrain=0.0),
    "corner":    dict(tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0),
    "strong":    dict(tau=0.15, pE=0.80, phi_max=0.65, retrain=0.8),
}
print("config      market_jobs  public_jobs  unemployed   (sum)")
for name, cfg in configs.items():
    p = v12.Params(eta=0.40, mu_TM=0.50, T=200, **cfg)
    r = v12.simulate_v12(p)
    mj, pj, un = r["sL"][-1], r["EP"][-1], r["Urem"][-1]
    print(f"{name:10s}  {mj:.3f}        {pj:.3f}        {un:.3f}    ({mj+pj+un:.3f})")
