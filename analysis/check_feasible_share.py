# -*- coding: utf-8 -*-
"""Reproduce the feasible-share counts in the recalibration table:
41/64 at pE=0.45, 17/64 at pE=0.30, and 208/384 overall.
Grid = v9c wide grid: 8 tau x 8 phi_max, at fixed pE."""
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

def run(p):
    r = simulate_v6(p); w = welfare(r, p)
    return dict(outcome=r['outcome'], minC=w['minC'], total=w['total'])

taus = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
phis = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.65]
pes  = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

print("feasible-share recount on the v9c wide grid")
print("=" * 60)
n_all = 0; tot_all = 0
for pe in pes:
    n = 0
    for tau in taus:
        for pm in phis:
            c = run(P(tau=tau, pE=pe, phi_max=pm, retrain=0.0, omega=0.05))
            if c['minC'] >= 0.30 and c['outcome'] == 'smooth':
                n += 1
    print("  pE=%.2f -> feasible %2d/64 = %5.1f%%" % (pe, n, 100.0 * n / 64))
    n_all += n; tot_all += 64
print("-" * 60)
print("  overall   -> feasible %3d/%d = %5.1f%%" % (n_all, tot_all, 100.0 * n_all / tot_all))
