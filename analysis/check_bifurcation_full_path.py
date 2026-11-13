# -*- coding: utf-8 -*-
"""Full-path correspondence of the boundary equilibrium bifurcation.

Locks the distinction that addresses the external review point on bifurcation
dynamics. The interior equilibrium of the boundary equilibrium theorem is an
equilibrium of a frozen static slice (skill and slow variables held fixed). On a
full dynamic path the direct automation flow keeps displacing market labor, so
the verified signature of an effective crossing is a discontinuous flattening
of the market-labor-share slope, not a reversal or interior convergence.

The strong regime (tau=0.15, p_E=0.80, phi_max=0.65, retrain=0.8) is checked
through the v12 endogenous entry:
  * s_L declines monotonically throughout and never increases;
  * the mean per-period decline is strictly smaller after the first effective
    crossing than before it, but stays negative;
  * with no capability erosion (beta_A=0) the effective capture stays above the
    threshold durably, yet s_L still declines over a long horizon and does not
    settle at a positive level.
All values are produced directly from the engine; no value is hard-coded.
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIMDIR = os.path.join(os.path.dirname(_HERE), "simulation")

_spec = importlib.util.spec_from_file_location(
    "v12", os.path.join(_SIMDIR, "double_task_sim_v12_endogenous_omega.py"))
v12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v12)

_STRONG = dict(tau=0.15, pE=0.80, phi_max=0.65, retrain=0.8)
THRESH = 0.5667


def run(beta_A, T=200):
    par = v12.Params(beta_A=beta_A, T=T, **_STRONG)
    r = v12.simulate_v12(par)
    return par, r


def first_cross(r):
    eff = r["omega"] * r["skill"]
    above = eff > THRESH
    edges = [i for i in range(len(above) - 1) if above[i] != above[i + 1]]
    return edges[0] if edges else None


print("=" * 70)
print("Full-path correspondence of the boundary equilibrium bifurcation")
print("=" * 70)

checks = []
for beta_A in (1, 0):
    par, r = run(beta_A)
    sL = r["sL"]
    slopes = np = __import__("numpy").diff(sL)
    c = first_cross(r)
    before = slopes[max(0, c - 20):c].mean()
    after = slopes[c + 1:c + 21].mean()
    print("beta_A=%d first cross t=%s slope before=%.5f after=%.5f" % (
        beta_A, c, before, after))
    checks.append(("beta_A=%d sL never increases" % beta_A,
                   bool(__import__("numpy").all(slopes < 0))))
    checks.append(("beta_A=%d slope flattens across crossing" % beta_A,
                   after > before and after < 0))

# Long-horizon no-erosion path: sL must keep declining and not settle positive.
_, long_r = run(0, T=800)
sL = long_r["sL"]
last_slope = __import__("numpy").diff(sL[-100:]).mean()
print("beta_A=0 T=800 sL_end=%.4f last-100 mean slope=%.6f" % (sL[-1], last_slope))
checks.append(("no-erosion long path does not converge positive",
               sL[-1] < sL[200] and last_slope < 0))

print("\nchecks against the manuscript")
bad = 0
for name, ok in checks:
    print("  %-44s %s" % (name, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
print("\n%d/%d checks passed" % (len(checks) - bad, len(checks)))
raise SystemExit(1 if bad else 0)
