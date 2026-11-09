# -*- coding: utf-8 -*-
"""Static bifurcation structure of the v12 fast subsystem and long-run dynamics.
(1) For each fixed omega, locate equilibria of f(s) and verify the BEB:
    omega < omega_c => no physical interior equilibrium;
    omega > omega_c => stable equilibrium s+ in region I.
(2) Long horizon (T=200) to see whether employment eventually arrests."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import sys
sys.path.insert(0, _SIMDIR)
import numpy as np
import sympy as sp
import double_task_sim_v12_endogenous_omega as v12

# ---------- symbolic fast subsystem with omega explicit ----------
s, mu, w, sig, kh, S0, lh, q, kI = sp.symbols(
    's mu omega sigma kappah S0 lambda_h q kappa_I', positive=True)
dN = kh * S0 * (1 - lh * (1 - s)) + q * mu
f = sp.expand(w * sig * dN - kI * mu - s * dN)
C = sp.simplify(-sp.diff(f, s, 2) / 2)
B = sp.simplify(sp.diff(f, s).subs(s, 0))
A = sp.simplify(f.subs(s, 0))

subs0 = {kh: 0.0025, S0: 1.0, lh: 1.20, q: 0.050, kI: 0.020,
         sig: 1.0, mu: 0.05}
sf = 1 - 1 / 1.20
OMEGA_C = 0.020 / 0.050 + sf
Cc = float(C.subs(subs0))
Bf = sp.lambdify(w, B.subs(subs0), 'numpy')
Af = sp.lambdify(w, A.subs(subs0), 'numpy')

print(f"C={Cc}  s_floor={sf:.4f}  omega_c={OMEGA_C:.4f}\n")
print(" omega    Delta        s+        s-      s+ location")
omegas = [0.05, 0.20, 0.40, 0.50, 0.55, 0.5667, 0.60, 0.70, 0.85, 1.0]
static_rows = []
for om in omegas:
    b, a = float(Bf(om)), float(Af(om))
    dd = b * b + 4 * Cc * a
    if dd >= 0:
        r1 = (b + np.sqrt(dd)) / (2 * Cc)
        r2 = (b - np.sqrt(dd)) / (2 * Cc)
        loc = ('IN region I (stable)' if r1 >= sf else
               'below s_floor (virtual)' if r1 > 0 else 'NEGATIVE')
        static_rows.append((om, r1, r2))
        print(f"{om:6.4f} {dd:10.3e} {r1:9.4f} {r2:9.4f}  {loc}")
    else:
        print(f"{om:6.4f} {dd:10.3e}   no real roots (absorbing s=0)")

# verify the BEB condition analytically: f(s_floor) = mu*(omega*q - kappa_I - q*s_floor)
f_sf = sp.simplify(f.subs(s, sp.Rational(1, 6)))
print("\nf(s_floor) =", sp.factor(f_sf.subs(subs0)))
print("=> zero exactly at omega =", sp.solve(sp.Eq(f_sf.subs(subs0), 0), w))

# ---------- long-run dynamics ----------
print("\n=== long horizon T=200, corner policy ===")
for name, cfg in [('no policy', dict(tau=0.05, pE=0.0, phi_max=0.15, retrain=0.0)),
                  ('corner', dict(tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0)),
                  ('strong', dict(tau=0.15, pE=0.80, phi_max=0.65, retrain=0.8))]:
    p = v12.Params(eta=0.40, mu_TM=0.50, T=200, **cfg)
    r = v12.simulate_v12(p)
    sL = r['sL']
    print(f"{name:10s} outcome={r['outcome']:11s} sL_end={sL[-1]:.4f} "
          f"minC={np.min(r['C'][10:]):.4f} omega_end={r['omega_end']:.4f}")
    # find where sL stops decreasing
    diffs = np.diff(sL)
    arrest = np.where(diffs >= -1e-6)[0]
    first_arrest = arrest[0] + 1 if len(arrest) else None
    print(f"           first period sL no longer falls: {first_arrest}  "
          f"(sL there={sL[first_arrest] if first_arrest else float('nan'):.4f})")
