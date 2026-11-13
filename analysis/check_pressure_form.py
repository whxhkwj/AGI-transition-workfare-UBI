# -*- coding: utf-8 -*-
"""M1 robustness: does the workfare-over-transfer social ranking survive the
FUNCTIONAL FORM of the pressure law?

The economic path is held fixed: each regime is simulated once and its relief
share path r_t = Urem_t / L^S is reused. Pressure is then recomputed under
alternative relief-conversion functions g(r), all endpoint-normalized g(1)=1 so
the regimes stay comparable at full relief, and the social crash is decided by
the same J-period persistence rule as the engine. Fixed seed keeps every path
exactly reproducible. Reproduces the linear (gamma=1) engine result as a check.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import sys, importlib.util
import numpy as np
sys.path.insert(0, _SIMDIR)
_spec = importlib.util.spec_from_file_location('v6', _os.path.join(_SIMDIR, 'double_task_sim_v6.py'))
v6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v6)

CALIB = dict(eta=0.40, mu_TM=0.50, omega=0.05)
REGIMES = {
    'corner':        dict(CALIB, tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0),
    'standard':      dict(CALIB, tau=0.16, pE=0.45, phi_max=0.65, retrain=0.0),
    'pure transfer': dict(CALIB, tau=0.16, pE=0.00, phi_max=0.65, retrain=0.0),
}
# relief-conversion exponent: concave, linear, convex; g(1)=1 in every case.
GAMMAS = [0.5, 1.0, 2.0]
J_BASE = 10
ALPHA_BASE = 0.60
THRESH_BASE = 0.50


def social_crash_from(relief, gamma, alpha_s, thresh, J):
    """Recompute pressure under g(r)=r**gamma and apply the engine's J rule."""
    g = np.power(relief, gamma)
    stress = alpha_s * g
    over = stress[10:] > thresh
    streak = longest = 0
    for x in over:
        streak = streak + 1 if x else 0
        longest = max(longest, streak)
    return float(np.max(stress)), longest >= J


def critical_alpha(relief, gamma, thresh, J):
    """Smallest alpha_s that triggers the J-period crash under this form."""
    lo, hi = 0.0, 20.0
    if not social_crash_from(relief, gamma, hi, thresh, J)[1]:
        return np.inf
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if social_crash_from(relief, gamma, mid, thresh, J)[1]:
            hi = mid
        else:
            lo = mid
    return hi


print("relief-conversion g(r) = r**gamma, endpoint-normalized g(1)=1")
print("alpha_s=%.2f, threshold=%.2f, J=%d; economic path fixed per regime\n" %
      (ALPHA_BASE, THRESH_BASE, J_BASE))
header = "%-14s %-8s %-11s %-9s %-14s" % ("regime", "gamma", "stress pk", "soc crash", "crit alpha_s")
print(header)
crit_table = {}
for name, cfg in REGIMES.items():
    par = v6.Params(**cfg)
    r = v6.simulate_v6(par)
    relief = np.asarray(r['Urem']) / par.LS
    crit_table[name] = {}
    for g in GAMMAS:
        pk, crashed = social_crash_from(relief, g, ALPHA_BASE, THRESH_BASE, J_BASE)
        ca = critical_alpha(relief, g, THRESH_BASE, J_BASE)
        crit_table[name][g] = ca
        print("%-14s %-8.1f %-11.4f %-9s %-14s" %
              (name, g, pk, str(crashed),
               ("%.4f" % ca) if np.isfinite(ca) else "never"))
    print()

# ranking check: under every form, does pure transfer crash while workfare survives?
print("=== ranking (workfare vs pure transfer) at baseline alpha_s ===")
for g in GAMMAS:
    corner_crash = social_crash_from(
        np.asarray(v6.simulate_v6(v6.Params(**REGIMES['corner']))['Urem']) / v6.Params(**REGIMES['corner']).LS,
        g, ALPHA_BASE, THRESH_BASE, J_BASE)[1]
    pt_crash = social_crash_from(
        np.asarray(v6.simulate_v6(v6.Params(**REGIMES['pure transfer']))['Urem']) / v6.Params(**REGIMES['pure transfer']).LS,
        g, ALPHA_BASE, THRESH_BASE, J_BASE)[1]
    order = "workfare survives, transfer crashes" if (not corner_crash and pt_crash) else "CHECK"
    print("gamma=%.1f: %s" % (g, order))
