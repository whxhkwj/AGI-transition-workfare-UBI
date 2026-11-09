# -*- coding: utf-8 -*-
"""Check the derived fast equation against the discrete simulator and map the
parameter region in which an interior employment equilibrium can exist at all."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import importlib.util, os
import numpy as np

BASE = _SIMDIR
spec = importlib.util.spec_from_file_location("v6", os.path.join(BASE, "double_task_sim_v6.py"))
v6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v6)

PAP = dict(kappa_N_h=0.005, S0=1.0, lambda_h=1.20, kappa_N_AI=0.050,
           M0=1.0, kappa_I=0.020, omega=0.05)

def f_and_coeffs(s, mu, sig, prm, Ntot):
    """continuous-time f(s)=ds/dt and the A,B,C decomposition"""
    kh = prm['kappa_N_h'] / Ntot
    Sh = max(0.0, prm['S0'] * (1 - prm['lambda_h'] * (1 - s)))
    q = prm['kappa_N_AI'] * prm['M0']
    gN = kh * Sh + q * mu
    return gN * (prm['omega'] * sig - s) - prm['kappa_I'] * mu, kh

# ---- 1) one-step consistency: continuous f vs the simulator's discrete update
par = v6.Params(tau=0.15, pE=0.45, phi_max=0.02, omega=0.05, retrain=0.0,
                eta=0.40, mu_TM=0.50)
r = v6.simulate_v6(par)
print("=== one-step consistency (continuous f vs discrete ds/dt) ===")
Ntot = float(r['Nt'][0]); sig = 1.0
print(" t |   s_t   | discrete ds |  cont f* 1 |  rel err")
for t in (1, 5, 20):
    s_prev = float(r['sL'][t-1]); s_now = float(r['sL'][t])
    mu = float(r['It'][0]) if False else (par.A0 * (1 + par.gA) ** t) / float(r['Nt'][t-1])
    ftot, _ = f_and_coeffs(s_prev, mu, sig, PAP, float(r['Nt'][t-1]))
    print(f"{t:2d} | {s_prev:.4f} | {s_now-s_prev:+.6f} | {ftot:.6f} | "
          f"{abs((s_now-s_prev)-ftot)/max(1e-12,abs(ftot)):.3f}")

# ---- 2) where can an interior equilibrium exist?  A(mu) sign condition
print("\n=== interior-equilibrium condition ===")
print("A(mu) = mu*(omega*sigma*q - kappa_I) + omega*sigma*D0,  D0 = kh*S0*(1-lambda_h) < 0")
for name, prm in (("paper calibration", PAP),):
    kh = prm['kappa_N_h'] / 2.0
    q = prm['kappa_N_AI'] * prm['M0']
    thr_omega = prm['kappa_I'] / q          # omega*sigma must exceed this for A to reach 0
    print(f"  kappa_I={prm['kappa_I']}, q={q} ->需要 omega*sigma > {thr_omega:.3f} 才有内部均衡")
    print(f"  basline omega*sigma = {prm['omega']*1.0:.3f}  (sigma=1.0)")
    print(f"  late   omega*sigma = {prm['omega']*0.30:.3f}  (sigma at the 0.30 floor)")
    A_0 = prm['omega'] * 1.0 * (q * 0.0) + prm['omega'] * 1.0 * (kh * prm['S0'] * (1 - prm['lambda_h']))
    print(f"  A(mu=0) at sigma=1: {A_0:.6f}  -> sign of the quadratic at s=0")

# ---- 3) fold existence across (omega, kappa_N_AI)
print("\n=== fold region scan over (omega, kappa_N_AI), sigma=1, N=2 ===")
kh = PAP['kappa_N_h'] / 2.0
S0, lh, kI = PAP['S0'], PAP['lambda_h'], PAP['kappa_I']
Ccoef = kh * S0 * lh
D0 = kh * S0 * (1 - lh)
print(f"C={Ccoef:.6f}  D0={D0:.6f}")

def fold_count(om, qv, sig=1.0):
    """number of distinct mus with a saddle-node of the quadratic f(s)"""
    b0 = Ccoef * om * sig - D0            # B = b0 - q*mu
    # A = om*sig*(D0+q*mu) - kI*mu = om*sig*D0 + mu*(om*sig*q - kI)
    a0 = om * sig * D0
    a1 = om * sig * qv - kI
    # A = -B^2/(4C) -> a0+a1*mu = -(b0-qv*mu)^2/(4C)
    # 4C(a0+a1 mu) + (b0-qv mu)^2 = 0
    A2 = qv ** 2
    A1 = -2 * b0 * qv + 4 * Ccoef * a1
    A0 = b0 ** 2 + 4 * Ccoef * a0
    disc = A1 ** 2 - 4 * A2 * A0
    if disc < 0:
        return 0, None
    roots = [(-A1 + np.sqrt(disc)) / (2 * A2), (-A1 - np.sqrt(disc)) / (2 * A2)]
    roots = [r for r in roots if (b0 - qv * r) > 0]     # requires B>0 at the fold
    return len(roots), sorted(roots)

print(f"{'omega':>6} {'k_NAI':>7} | {'nfold':>5}   roots(mu)")
for om in (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00):
    for qn in (0.03, 0.05, 0.08):
        n, rr = fold_count(om, qn)
        print(f"{om:6.2f} {qn:7.3f} | {n:5d}   "
              + (" ".join(f"{x:.3f}" for x in rr) if rr else "-"))

print("\nbaseline cell: omega=0.05, kappa_N^AI=0.05 ->", fold_count(0.05, 0.05))
print("interior equilibrium needs omega*sigma*q > kappa_I =", kI)
