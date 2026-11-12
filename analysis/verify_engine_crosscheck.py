# -*- coding: utf-8 -*-
"""引擎原始输出独立交叉验证：恒等式、手工 eff、动态 crossing。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "simulation"))
import numpy as np
import double_task_sim_v12_endogenous_omega as v12

EFF_CRIT = 0.5666666667
configs = {
    "no policy": dict(tau=0.05, pE=0.0, phi_max=0.15, retrain=0.0),
    "corner":    dict(tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0),
    "strong":    dict(tau=0.15, pE=0.80, phi_max=0.65, retrain=0.8),
}
for name, cfg in configs.items():
    p = v12.Params(eta=0.40, mu_TM=0.50, T=200, **cfg)
    r = v12.simulate_v12(p)
    It, Nt, nlab = r["It"], r["Nt"], r["nlab"]
    om, sk = r["omega"], r["skill"]

    # 1) n=N-I 恒等式残余
    resid = np.max(np.abs(nlab - np.maximum(0, Nt - It)))
    # 2) 手工 eff 与动态 crossing
    eff = np.array([om[t] * sk[t - 1] for t in range(1, 200)])
    cross = np.where(eff >= EFF_CRIT)[0]
    cross_t = (cross[0] + 1) if len(cross) else None
    # 3) 手工动态阈值
    dyn = EFF_CRIT / sk[:-1]
    print(f"{name:10s} 恒等式残余={resid:.2e}  eff峰={eff.max():.4f}  "
          f"真实cross={cross_t}  动态阈值范围[{dyn[1:].min():.3f},{dyn[1:].max():.3f}]")
