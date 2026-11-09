# -*- coding: utf-8 -*-
"""v9b：确认修正参数下三维寻优最优是否落在 φmax 边界；做细扫找真实内点"""
import os, importlib.util
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('v6', os.path.join(OUT, 'double_task_sim_v6.py'))
v6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v6)
Params, simulate_v6, welfare = v6.Params, v6.simulate_v6, v6.welfare


def P(**kw):
    base = dict(eta=0.40, mu_TM=0.50)
    base.update(kw)
    return Params(**base)


def run(p):
    r = simulate_v6(p); w = welfare(r, p)
    return dict(outcome=r['outcome'], total=w['total'], minC=w['minC'],
                skill_end=r['skill_end'], tax_sum=r['tax_sum'])


print('v9b 细扫：修正参数 (eta=0.40, mu_TM=0.50) 下三维最优确认')
print('=' * 80)

best = None
# φmax 更低区间 + τ/pE 邻域
taus = [0.18, 0.20, 0.22, 0.25]
pes = [0.30, 0.35, 0.40, 0.45, 0.50]
phis = [0.15, 0.25, 0.35, 0.40, 0.45, 0.55, 0.65]
n_ok = 0
print('\n按 (τ,pE) 汇总：该列在 φmax∈%s 下的可行最优（loss）' % str(phis))
for tau in taus:
    row = []
    for pe in pes:
        best_cell = None
        for pm in phis:
            c = run(P(tau=tau, pE=pe, phi_max=pm, retrain=0.0, omega=0.05))
            if c['minC'] >= 0.30 and c['outcome'] == 'smooth':
                n_ok += 1
                if best_cell is None or c['total'] < best_cell['total']:
                    best_cell = dict(phi_max=pm, **c)
                if best is None or c['total'] < best['total']:
                    best = dict(tau=tau, pE=pe, phi_max=pm, **c)
        row.append('%s@%.2f=%.2f' % ('-' if best_cell is None else 'OK',
                                     best_cell['phi_max'] if best_cell else 0,
                                     best_cell['total'] if best_cell else 0))
    print('  τ=%.2f | ' % tau + ' | '.join('%7s' % x for x in row))

print('\n全局最优（细扫）: τ=%.2f pE=%.2f φmax=%.2f | total=%.2f minC=%.3f skill_end=%.3f tax_sum=%.1f'
      % (best['tau'], best['pE'], best['phi_max'], best['total'],
         best['minC'], best['skill_end'], best['tax_sum']))
print('可行组合数（细扫）: %d' % n_ok)
