# -*- coding: utf-8 -*-
"""v9c：修正参数下宽网格寻优——确认最优区域（τ、φmax 下探到角点）"""
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


print('v9c 宽网格寻优（eta=0.40, mu_TM=0.50, ω=0.05, retrain=0）')
print('=' * 84)
taus = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
pes = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
phis = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.65]
best = None; n_ok = 0; n_crash = 0; tot = 0
# 记录每个 τ 的最优，看趋势
for tau in taus:
    b_tau = None
    for pe in pes:
        for pm in phis:
            tot += 1
            c = run(P(tau=tau, pE=pe, phi_max=pm, retrain=0.0, omega=0.05))
            if c['minC'] >= 0.30 and c['outcome'] == 'smooth':
                n_ok += 1
                if b_tau is None or c['total'] < b_tau['total']:
                    b_tau = dict(pE=pe, phi_max=pm, **c)
                if best is None or c['total'] < best['total']:
                    best = dict(tau=tau, pE=pe, phi_max=pm, **c)
            else:
                n_crash += (c['outcome'] == 'crash')
    print('  τ=%.2f -> 该τ最优 pE=%.2f φmax=%.2f | total=%.2f minC=%.3f'
          % (tau, b_tau['pE'], b_tau['phi_max'], b_tau['total'], b_tau['minC'])
          if b_tau else '  τ=%.2f -> 该τ无可平稳' % tau)

print('\n全局最优: τ=%.2f pE=%.2f φmax=%.2f | total=%.2f minC=%.3f skill_end=%.3f tax_sum=%.1f'
      % (best['tau'], best['pE'], best['phi_max'], best['total'],
         best['minC'], best['skill_end'], best['tax_sum']))
print('网格 %d：可行 %d | 崩溃 %d | 未到终局 %d'
      % (tot, n_ok, n_crash, tot - n_ok - n_crash))
