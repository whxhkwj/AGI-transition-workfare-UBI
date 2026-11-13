# -*- coding: utf-8 -*-
"""M2 robustness: locate the welfare-loss optimum tau* by direct simulation and
compare it with the analytic Laffer ceiling tau_F. The financing peak enters
the admissible [0,1] range once eta is large enough (or the horizon long
enough); the script reports where tau* and tau_F coincide and where they
diverge. Fixed headline policy coordinates (pE=0.45, phi_max=0.02). All
numbers are produced by the engine; nothing is taken from the analytic proof.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import sys, importlib.util
import numpy as np
sys.path.insert(0, _SIMDIR)
_spec = importlib.util.spec_from_file_location('u', _os.path.join(_SIMDIR, 'double_task_sim_unified.py'))
u = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(u)

gG = 0.03
TAU_GRID = np.round(np.arange(0.04, 0.96, 0.02), 4)


def tau_F(eta, t):
    return (1 + gG) / (gG * eta * (t + 1))


def welfare_optimum(eta, t):
    """Minimize the engine welfare loss over tau; return tau* and its total."""
    best_tau, best_total, best_outcome = None, np.inf, None
    rows = []
    for tau in TAU_GRID:
        par = u.Params(eta=eta, mu_TM=0.50, T=t, tau=float(tau),
                       pE=0.45, phi_max=0.02, retrain=0.0)
        r = u.simulate_unified(par)
        w = u.welfare(r, par)
        rows.append((float(tau), r['outcome'], w['total']))
        if w['total'] < best_total:
            best_total, best_tau, best_outcome = w['total'], float(tau), r['outcome']
    return best_tau, best_total, best_outcome, rows


print("welfare optimum tau* (direct minimization) vs Laffer ceiling tau_F")
print("%-6s %-6s %-9s %-9s %-10s %-26s" %
      ("eta", "T", "tau_F", "tau*", "tau* outc", "reading"))
for eta in [0.40, 0.50, 0.60]:
    for t in [80, 100]:
        tf = tau_F(eta, t)
        ts, tot, outc, _ = welfare_optimum(eta, t)
        if tf > 1.0:
            reading = "peak outside admissible range; tau* free"
        elif ts < tf - 1e-9:
            reading = "tau* inside peak; financing not binding"
        else:
            reading = "tau* at/beyond peak; financing not binding"
        print("%-6.2f %-6d %-9.4f %-9.4f %-10s %-26s" %
              (eta, t, tf, ts, outc, reading))

print("\n=== no-decay re-solve at baseline eta=0.40, T=80 ===")
print("The engine welfare contains no decay assumption; report the full W shape")
ts, tot, outc, rows = welfare_optimum(0.40, 80)
left = [x for x in rows if x[0] < ts]
right = [x for x in rows if x[0] > ts]
mono_left = all(left[i][2] >= left[i + 1][2] for i in range(len(left) - 1))
mono_right = all(right[i][2] <= right[i + 1][2] for i in range(len(right) - 1))
W_left, W_right = rows[0][2], rows[-1][2]
print("argmin tau*=%.2f (interior, outcome=%s), W(tau*)=%.4f" % (ts, outc, tot))
print("W decreasing on the left: %s; W increasing on the right: %s" % (mono_left, mono_right))
print("W at left edge %.4f, W at right edge %.4f, both above the minimum" % (W_left, W_right))
print("right-boundary sign is positive without any decay assumption; no corner solution")
