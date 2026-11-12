# -*- coding: utf-8 -*-
"""Locate the real BEB crossing on the corner path and bisect the public-
employment intensity that defends consumption (min_t C_t >= Cmin).

Under the unified logistic engine the corner path (retrain=0) has no nominal or
real crossing because skill depreciation binds; the script reports that and
keeps the p_E* consumption-floor bisection, which does not require a crossing.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import sys
sys.path.insert(0, _HERE)
sys.path.insert(0, _SIMDIR)
import numpy as np
import double_task_sim_v12_endogenous_omega as v12
import dynamic_threshold as dt

par = v12.Params(eta=0.40, mu_TM=0.50, T=200, tau=0.15, pE=0.45,
                 phi_max=0.02, retrain=0.0)
r = v12.simulate_v12(par)
om, skill, sL, C = r['omega'], r['skill'], r['sL'], r['C']

# nominal crossing: static threshold (skill implicitly 1); real crossing: eff.
nom_t = next((t for t in range(1, 200) if om[t] >= dt.EFF_CRIT), None)
real_t = dt.first_real_crossing(om, skill)
print(f"nominal crossing (static omega_c={dt.EFF_CRIT:.4f}): t={nom_t}")
print(f"real crossing (effective capture)               : t={real_t}")
if nom_t is None:
    print("corner has no crossing at any date: the effective-capture peak is "
          f"{float(np.max(om[1:]*skill[:-1])):.4f} < {dt.EFF_CRIT:.4f}.")

print("\n t    omega   skill(prev)  dynamic_crit   eff     sL       C")
for t in (10, 20, 30, 40, 60, 80):
    dyn_t = dt.dynamic_threshold(skill)[t]
    eff_t = dt.effective_capture(om, skill)[t]
    print(f"{t:3d} {om[t]:7.4f} {skill[t-1]:7.4f}    {dyn_t:.4f}      "
          f"{eff_t:.4f} {sL[t]:.4f} {C[t]:.4f}")

print("\n=== bisection: pE that defends consumption (minC >= Cmin) ===")
def minC_of(pE):
    p = v12.Params(eta=0.40, mu_TM=0.50, tau=0.15, pE=pE,
                   phi_max=0.02, retrain=0.0)
    rr = v12.simulate_v12(p)
    return float(np.min(rr['C'][10:])), rr['Cmin']

lo, hi = 0.30, 0.45
for _ in range(16):
    mid = 0.5 * (lo + hi)
    m, cmin = minC_of(mid)
    if m - cmin < 0:
        lo = mid
    else:
        hi = mid
pE_star = 0.5 * (lo + hi)
print(f"pE* (consumption floor, minC = Cmin) = {pE_star:.6f}")

# at pE*, is the real BEB reached?
p = v12.Params(eta=0.40, mu_TM=0.50, tau=0.15, pE=pE_star,
               phi_max=0.02, retrain=0.0)
rr = v12.simulate_v12(p)
real_at_star = dt.first_real_crossing(rr['omega'], rr['skill'])
print(f"  at pE*: real BEB crossing = {real_at_star} (consumption defended "
      f"without a high-employment restoration)")

print("\n=== outcomes on either side of pE* ===")
for pE in [pE_star - 0.05, pE_star, pE_star + 0.05]:
    p = v12.Params(eta=0.40, mu_TM=0.50, tau=0.15, pE=pE,
                   phi_max=0.02, retrain=0.0)
    rr = v12.simulate_v12(p)
    print(f"pE={pE:6.3f} outcome={rr['outcome']:11s} crash_type={rr['crash_type']:6s} "
          f"minC={np.min(rr['C'][10:]):.4f} sL_end={rr['sL'][-1]:.4f}")
