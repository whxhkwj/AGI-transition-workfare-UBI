# -*- coding: utf-8 -*-
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")
import os, importlib.util

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


taus = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20]
pes = [0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 0.52, 0.55]
phis = [0.001, 0.005, 0.010, 0.020, 0.050, 0.100]

print('=== B. refined grid (%d cells) ===' % (len(taus) * len(pes) * len(phis)))
best = None
ok = 0
tot = 0
for tau in taus:
    b_tau = None
    for pe in pes:
        for pm in phis:
            tot += 1
            r, w = run(tau, pe, pm)
            if w['minC'] >= 0.30 and r['outcome'] == 'smooth':
                ok += 1
                if b_tau is None or w['total'] < b_tau[3]:
                    b_tau = (tau, pe, pm, w['total'], w['minC'])
                if best is None or w['total'] < best[3]:
                    best = (tau, pe, pm, w['total'], w['minC'])
    print(' tau=%.2f -> %s' % (tau, ('pE=%.2f phi=%.3f total=%.3f minC=%.4f' % b_tau[1:]) if b_tau else 'none'))

print('\ngrid=%d feasible=%d' % (tot, ok))
print('global best: tau=%.2f pE=%.2f phi=%.3f total=%.3f minC=%.4f' % best)

# The manuscript additionally reports the scan that fixes the ceiling at
# phi^max = 0.15 and returns (0.18, 0.40, 0.15) with loss 63.53.
print('\n=== C. fixed ceiling phi_max=0.15 (Section 5) ===')
print('  tau    pE     phi    outcome    min C    total loss')
for tau in [0.15, 0.16, 0.18, 0.20]:
    r, w = run(tau, 0.40, 0.15)
    print('  %.2f  %.2f  %.2f   %-8s  %.4f  %9.2f' % (
        tau, 0.40, 0.15, r['outcome'], w['minC'], w['total']))
r, w = run(0.18, 0.40, 0.15)
print('  reported cell (0.18, 0.40, 0.15): total = %.4f  (manuscript 63.53)' % w['total'])
assert abs(w['total'] - 63.53) < 0.02, w['total']
