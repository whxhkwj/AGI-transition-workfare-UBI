# -*- coding: utf-8 -*-
"""Confirm the employment reversal in the strong policy: locate the sL minimum
and check it aligns with omega crossing omega_c."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import sys
sys.path.insert(0, _SIMDIR)
import numpy as np
import double_task_sim_v12_endogenous_omega as v12

OMEGA_C = 0.5667
p = v12.Params(eta=0.40, mu_TM=0.50, T=200,
               tau=0.15, pE=0.80, phi_max=0.65, retrain=0.8)
r = v12.simulate_v12(p)
sL, om = r['sL'], r['omega']

imin = int(np.argmin(sL))
cross = int(np.argmax(om >= OMEGA_C))
print(f"strong policy: omega crosses omega_c at t={cross}")
print(f"sL minimum at t={imin}, sL_min={sL[imin]:.4f}; sL_end={sL[-1]:.4f}")
print(f"reversal (sL trough) lags BEB crossing by {imin-cross} periods")

print("\n t    omega     sL      d(sL)")
for t in list(range(cross - 2, cross + 3)) + list(range(imin - 2, imin + 3)):
    if 1 <= t < 199:
        ds = sL[t] - sL[t - 1]
        tag = ("BEB" if t == cross else "") + ("  sL trough" if t == imin else "")
        print(f"{t:3d} {om[t]:7.4f} {sL[t]:7.4f} {ds:+8.5f}  {tag}")

print("\nsL at milestones:",
      " ".join(f"t{t}={sL[t]:.3f}" for t in (0, cross, imin, 120, 160, 199)))
print("omega at milestones:",
      " ".join(f"t{t}={om[t]:.3f}" for t in (0, cross, imin, 120, 160, 199)))
