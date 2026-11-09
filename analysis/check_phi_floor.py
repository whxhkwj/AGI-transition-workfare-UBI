# -*- coding: utf-8 -*-
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")
import os, importlib.util, time

SKEL = _SIMDIR
_spec = importlib.util.spec_from_file_location('v6', os.path.join(SKEL, 'double_task_sim_v6.py'))
v6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v6)
Params, simulate_v6, welfare = v6.Params, v6.simulate_v6, v6.welfare


def P(**kw):
    base = dict(eta=0.40, mu_TM=0.50)
    base.update(kw)
    return Params(**base)


def run(tau, pe, pm):
    p = P(tau=tau, pE=pe, phi_max=pm, retrain=0.0, omega=0.05)
    r = simulate_v6(p)
    w = welfare(r, p)
    return r, w


print('=== A. phi lower bound at corner tau=0.15, pE=0.45 ===')
t0 = time.time()
for pm in [0.001, 0.005, 0.010, 0.015, 0.020, 0.050, 0.100]:
    r, w = run(0.15, 0.45, pm)
    print('phi=%.3f outcome=%-6s minC=%.4f total=%.3f tax_sum=%.2f'
          % (pm, r['outcome'], w['minC'], w['total'], r['tax_sum']))
print('elapsed per run: %.2fs' % ((time.time() - t0) / 7.0))
