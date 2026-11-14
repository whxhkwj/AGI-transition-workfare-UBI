# -*- coding: utf-8 -*-
"""Bisect the standard-setting workfare threshold quoted in the
'Two configurations, not one' paragraph and assert the engine value.

Standard setting (see check_table2.py): tau=0.16, phi_max=0.65, retrain=0,
exogenous omega=0.05. The threshold is the smallest pE that defends the
consumption floor (min_t C_t >= Cmin). Also checks the two fixed points.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import importlib.util
import numpy as np

_spec = importlib.util.spec_from_file_location(
    'v6', _os.path.join(_SIMDIR, 'double_task_sim_v6.py'))
v6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v6)
Params, simulate_v6 = v6.Params, v6.simulate_v6


def minC(pE):
    p = Params(eta=0.40, mu_TM=0.50, omega=0.05, tau=0.16, pE=pE,
               phi_max=0.65, retrain=0.0)
    r = simulate_v6(p)
    return float(np.min(r['C'][10:])), r['Cmin']


lo, hi = 0.05, 0.40
for _ in range(16):
    mid = 0.5 * (lo + hi)
    m, cmin = minC(mid)
    if m - cmin < 0:
        lo = mid
    else:
        hi = mid
pE_star = 0.5 * (lo + hi)
print('standard setting bisection pE* = %.4f' % pE_star)

checks = []
checks.append(('pE* matches manuscript 0.1405', abs(pE_star - 0.1405) < 5e-4))

for pE, expect in ((0.30, 0.333), (0.45, 0.358)):
    m, _ = minC(pE)
    checks.append(('pE=%.2f minC=%.3f' % (pE, expect), abs(m - expect) < 5e-3))

# the threshold itself sits exactly on the floor
m_at, cmin = minC(pE_star)
checks.append(('at pE* minC equals Cmin', abs(m_at - cmin) < 5e-3))

bad = 0
for name, ok in checks:
    print('  %-34s %s' % (name, 'PASS' if ok else 'FAIL'))
    bad += 0 if ok else 1
print('%d/%d checks passed' % (len(checks) - bad, len(checks)))
raise SystemExit(1 if bad else 0)
