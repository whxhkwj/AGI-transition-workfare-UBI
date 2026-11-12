# -*- coding: utf-8 -*-
"""M4: stochastic-shock Monte Carlo on the calibrated corner.

Derives a shock-enabled copy of double_task_sim_v6.py in memory (single line
changed: the frontier compound growth), leaving the original file untouched.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import os, numpy as np

SKEL = _SIMDIR
path = os.path.join(SKEL, 'double_task_sim_v6.py')
src = open(path, encoding='utf-8').read()

old = '        cap = par.A0 * (1 + par.gA) ** t'
assert src.count(old) == 1, 'anchor for cap not unique'
src = src.replace(old, '        cap = par.A0 * par._capexp[t]')

anchor = '    b[0] = D[0] / P[0]\n    Cfe = par.w * par.LS\n'
assert src.count(anchor) == 1, 'anchor for Cfe not unique'
inject = (
    "    _g = getattr(par, '_gA_path', None)\n"
    "    if _g is None:\n"
    "        _gg = np.full(T, par.gA)\n"
    "    else:\n"
    "        _gg = np.array(_g, dtype=float)\n"
    "    _gg[0] = 0.0\n"
    "    par._capexp = np.cumprod(1.0 + _gg)\n"
)
src = src.replace(anchor, inject + anchor)

ns = {'__name__': 'v6shock', '__file__': path}
exec(compile(src, path, 'exec'), ns)
Params, simulate, welfare = ns['Params'], ns['simulate_v6'], ns['welfare']


def mc(tau, pe, pm, sigma, rho=0.5, n=None, seed=7, eta_sd=0.0, mu_sd=0.0):
    # 200 draws per cell in the paper. Set MC_DRAWS to a small number for a
    # quick smoke test; the default stays at the published 200.
    if n is None:
        n = int(_os.environ.get("MC_DRAWS", "200"))
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p = Params(eta=0.40 + rng.normal(0, eta_sd), mu_TM=0.50 + rng.normal(0, mu_sd),
                   tau=tau, pE=pe, phi_max=pm, retrain=0.0, omega=0.05)
        T = p.T
        eps = rng.normal(0.0, sigma, T)
        dev = np.zeros(T)
        for t in range(1, T):
            dev[t] = rho * dev[t - 1] + eps[t]
        dev[0] = 0.0
        p._gA_path = 0.010 + dev
        r = simulate(p)
        w = welfare(r, p)
        out.append((r['outcome'], w['minC'], w['total']))
    occ = {}
    for o, _, _ in out:
        occ[o] = occ.get(o, 0) + 1
    minc = np.array([m for _, m, _ in out])
    tot = np.array([t for _, _, t in out])
    return dict(n=n, occ=occ, p05=float(np.percentile(minc, 5)),
                p50=float(np.percentile(minc, 50)),
                mean_total=float(tot.mean()), max_total=float(tot.max()))


print('sigma = sd of the iid innovation in the frontier growth (AR(1), rho=0.5)')
print('%-8s %-10s %-6s %-8s %-8s %-7s %-9s %-9s' %
      ('config', 'sigma', 'n', 'smooth', 'crash', 'minC05', 'minC50', 'meanLoss'))
for cfg, (tau, pe, pm) in [('headline', (0.15, 0.45, 0.020)),
                           ('refined', (0.13, 0.44, 0.001))]:
    for sigma in [0.0, 0.005, 0.010, 0.020, 0.030]:
        d = mc(tau, pe, pm, sigma)
        s = d['occ'].get('smooth', 0)
        c = d['occ'].get('crash', 0)
        print('%-8s %-10.3f %-6d %-8d %-8d %-7.4f %-9.4f %-9.2f' %
              (cfg, sigma, d['n'], s, c, d['p05'], d['p50'], d['mean_total']))

print('\n--- parameter uncertainty on top of sigma=0.010 (eta sd .05, mu_TM sd .05) ---')
for cfg, (tau, pe, pm) in [('headline', (0.15, 0.45, 0.020)),
                           ('refined', (0.13, 0.44, 0.001))]:
    d = mc(tau, pe, pm, 0.010, eta_sd=0.05, mu_sd=0.05, seed=11)
    print('%-8s smooth=%d crash=%d minC05=%.4f meanLoss=%.2f' %
          (cfg, d['occ'].get('smooth', 0), d['occ'].get('crash', 0),
           d['p05'], d['mean_total']))
