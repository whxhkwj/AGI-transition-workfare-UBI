# -*- coding: utf-8 -*-
"""Reproduce the g_A and eta sensitivity sweeps reported at the corner
configuration (tau=0.15, pE=0.45, phi_max=0.02) in the Calibration section."""
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

corner = dict(tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0, omega=0.05)

print("g_A sweep at the corner (tau=0.15, pE=0.45, phi_max=0.02)")
print("=" * 66)
print("  %-8s %-10s %-9s %-9s" % ("g_A", "outcome", "minC", "total"))
for g in [0.010, 0.05, 0.10, 0.20, 0.30]:
    c = run(P(gA=g, **corner))
    print("  %-8.3f %-10s %-9.3f %-9.2f" % (g, c['outcome'], c['minC'], c['total']))

print("\neta sweep at the corner")
print("=" * 66)
print("  %-8s %-10s %-9s %-9s" % ("eta", "outcome", "minC", "total"))
res = []
for e in [0.20, 0.40, 0.60, 0.80]:
    c = run(P(eta=e, **corner))
    res.append(c)
    print("  %-8.2f %-10s %-9.3f %-9.2f" % (e, c['outcome'], c['minC'], c['total']))

lo = min(r['total'] for r in res); hi = max(r['total'] for r in res)
print("\n  loss spread over eta in [0.20, 0.80]: %.2f .. %.2f  (%.2f%% of %.2f)"
      % (lo, hi, 100.0 * (hi - lo) / hi, hi))
