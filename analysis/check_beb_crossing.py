# -*- coding: utf-8 -*-
"""Locate the exact BEB crossing in the time series and verify that the
employment decline arrests when omega crosses omega_c. Also bisect the pE
critical value."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import sys
sys.path.insert(0, _SIMDIR)
import numpy as np
import double_task_sim_v12_endogenous_omega as v12

OMEGA_C = 0.020 / 0.050 + (1 - 1 / 1.20)

par = v12.Params(eta=0.40, mu_TM=0.50, tau=0.15, pE=0.45,
                 phi_max=0.02, retrain=0.0)
r = v12.simulate_v12(par)
om, sL, C = r['omega'], r['sL'], r['C']

# first period omega crosses omega_c
cross_t = int(np.argmax(om >= OMEGA_C))
print(f"omega first crosses omega_c={OMEGA_C:.4f} at t={cross_t} "
      f"(omega[{cross_t}]={om[cross_t]:.4f}, omega[{cross_t-1}]={om[cross_t-1]:.4f})")

print("\n t    omega     sL       C      d(sL)      sign of employment change")
for t in range(max(1, cross_t - 6), min(80, cross_t + 8)):
    ds = sL[t] - sL[t - 1]
    mark = " <-- BEB crossing" if t == cross_t else ""
    print(f"{t:3d} {om[t]:7.4f} {sL[t]:7.4f} {C[t]:7.4f} {ds:+8.5f}   "
          f"{'falling' if ds<0 else 'rising/flat'}{mark}")

# does sL reach a minimum near the crossing and then arrest?
imin = int(np.argmin(sL))
print(f"\nsL minimum at t={imin}, sL_min={sL[imin]:.4f}; sL_end={sL[-1]:.4f}")
print(f"BEB crossing t={cross_t}, sL minimum t={imin} (gap {imin-cross_t} periods)")

print("\n=== bisection: critical pE for avoiding the crash (minC >= Cmin) ===")
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
print(f"pE* (minC = Cmin) = {pE_star:.6f}")
m, cmin = minC_of(pE_star)
print(f"  at pE*: minC={m:.6f}, Cmin={cmin:.6f}")

# what is omega at that critical policy?
p = v12.Params(eta=0.40, mu_TM=0.50, tau=0.15, pE=pE_star,
               phi_max=0.02, retrain=0.0)
rr = v12.simulate_v12(p)
print(f"  omega trajectory peak at pE*: {np.max(rr['omega']):.4f} (omega_c={OMEGA_C:.4f})")

print("\n=== contrast: full outcomes on either side of pE* ===")
for pE in [pE_star - 0.05, pE_star, pE_star + 0.05]:
    p = v12.Params(eta=0.40, mu_TM=0.50, tau=0.15, pE=pE,
                   phi_max=0.02, retrain=0.0)
    rr = v12.simulate_v12(p)
    print(f"pE={pE:6.3f} outcome={rr['outcome']:11s} crash_type={rr['crash_type']:6s} "
          f"minC={np.min(rr['C'][10:]):.4f} omega_end={rr['omega_end']:.4f}")
