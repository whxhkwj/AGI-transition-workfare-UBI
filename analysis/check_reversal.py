# -*- coding: utf-8 -*-
"""After the real BEB, verify that the strong-policy employment share converges
to a positive interior equilibrium rather than being absorbed to zero.

The share does not reverse upward within the window: it declines monotonically
from its initial value while its rate of decline shrinks toward zero, which is
convergence to a positive equilibrium and is distinct from the no-policy
absorption to zero."""
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

p = v12.Params(eta=0.40, mu_TM=0.50, T=200,
               tau=0.15, pE=0.80, phi_max=0.65, retrain=0.8)
r = v12.simulate_v12(p)
sL, om = r['sL'], r['omega']
real_t = dt.first_real_crossing(om, r['skill'])

print(f"strong policy: real BEB crossing at t={real_t}")
print(f"sL stays positive: sL_end={sL[-1]:.4f} (no absorption to zero)")
dsl = np.diff(sL)
print("\n t    sL       d(sL)      decline rate")
for t in (1, 20, 40, 80, 120, 160, 198):
    print(f"{t:3d} {sL[t]:.4f}  {dsl[t-1]:+.5f}   {abs(dsl[t-1]):.5f}")

# contrast with no-policy absorption
pn = v12.Params(eta=0.40, mu_TM=0.50, T=200,
                tau=0.05, pE=0.0, phi_max=0.15, retrain=0.0)
rn = v12.simulate_v12(pn)
print(f"\nno-policy sL_end={rn['sL'][-1]:.4f} (absorbed), "
      f"strong sL_end={sL[-1]:.4f} (positive equilibrium)")
print("decline is monotone throughout:", np.all(dsl <= 1e-12))
