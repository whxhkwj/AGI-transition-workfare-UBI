# -*- coding: utf-8 -*-
"""C1 支撑：数值目标函数 L(tau,pE,phi) 的权重敏感性。
网格与 v9c_wide 完全一致（8*6*8=384），参数 eta=0.40, mu_TM=0.50, omega=0.05, retrain=0。
可行性判定 min_{t>=10} C_t >= q 与权重无关，故可行格数应在各组相同。
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import os, importlib.util
import numpy as np

SKEL = _SIMDIR
spec = importlib.util.spec_from_file_location('v6', os.path.join(SKEL, 'double_task_sim_v6.py'))
v6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v6)
Params, simulate_v6, welfare = v6.Params, v6.simulate_v6, v6.welfare

taus = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
pes  = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
phis = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.65]

WEIGHT_SETS = [
    ('baseline  lam=0.5 lam_pe=0.6 nu=0.5 delta=0.6', dict(lam=0.5,  lam_pe=0.6, nu=0.5,  delta=0.6)),
    ('flat      lam=1.0 lam_pe=1.0 nu=0.5 delta=1.0', dict(lam=1.0,  lam_pe=1.0, nu=0.5,  delta=1.0)),
    ('half      lam=0.25 lam_pe=0.3 nu=0.25 delta=0.3', dict(lam=0.25, lam_pe=0.3, nu=0.25, delta=0.3)),
    ('double    lam=1.0 lam_pe=1.2 nu=1.0 delta=1.2', dict(lam=1.0,  lam_pe=1.2, nu=1.0,  delta=1.2)),
    ('tax-heavy lam=0.5 lam_pe=0.6 nu=0.5 delta=1.2', dict(lam=0.5,  lam_pe=0.6, nu=0.5,  delta=1.2)),
    ('tax-light lam=0.5 lam_pe=0.6 nu=0.5 delta=0.2', dict(lam=0.5,  lam_pe=0.6, nu=0.5,  delta=0.2)),
]

cache = {}
tot = 0
for tau in taus:
    for pe in pes:
        for pm in phis:
            tot += 1
            p = Params(tau=tau, pE=pe, phi_max=pm, retrain=0.0, omega=0.05,
                       eta=0.40, mu_TM=0.50)
            cache[(tau, pe, pm)] = (p, simulate_v6(p))

print('grid size = %d' % tot)
for name, wv in WEIGHT_SETS:
    best = None
    n_ok = 0
    for (tau, pe, pm), (p, r) in cache.items():
        w = welfare(r, p, **wv)
        ok = (w['minC'] >= p.q) and (r['outcome'] == 'smooth')
        if ok:
            n_ok += 1
            if best is None or w['total'] < best['total']:
                best = dict(tau=tau, pE=pe, pm=pm, total=w['total'],
                            minC=w['minC'], gap=w['gap'], cost=w['cost'], dwl=w['dwl'])
    print('%-50s feasible=%3d  best=(tau=%.2f,pE=%.2f,phi=%.2f) total=%.2f minC=%.3f  [gap=%.2f cost=%.2f dwl=%.2f]'
          % (name, n_ok, best['tau'], best['pE'], best['pm'], best['total'],
             best['minC'], best['gap'], best['cost'], best['dwl']))
