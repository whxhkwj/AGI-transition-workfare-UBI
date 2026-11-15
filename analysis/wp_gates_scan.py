# -*- coding: utf-8 -*-
"""Two-gate frontier under the upgraded engine.

For each capital tax, scans the frontier growth gA and reports the critical
value set by (a) the fiscal gate (minimum consumption over t>=10 at the floor)
and (b) the adjustment-speed gate (minimum market employment over the early
window t<10 at s_floor=0.50). The gate that binds first is the operative
boundary. The speed-gate-only column removes the consumption-floor check to
isolate that gate, mirroring the gate-removal experiment.
"""
import os, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIMDIR = os.path.join(ROOT, "simulation")

_spec = importlib.util.spec_from_file_location(
    "U", os.path.join(SIMDIR, "double_task_sim_unified.py"))
U = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(U)

S_FLOOR = 0.50
GRID = np.arange(0.01, 0.251, 0.01)
TAUS = [0.10, 0.15, 0.20]


def critical_gate(tau, gate):
    """Return the largest gA that passes the requested gate."""
    last_pass = None
    for g in GRID:
        par = U.Params(tau=tau, pE=1.0, phi_max=0.65, gA=float(g))
        r = U.simulate_unified(par)
        if gate == "fiscal":
            ok = float(np.min(r["C"][10:])) >= r["Cmin"] - 1e-9
        elif gate == "speed":
            ok = float(np.min(r["sL"][:10])) >= S_FLOOR - 1e-9
        else:  # both gates together
            ok = (float(np.min(r["C"][10:])) >= r["Cmin"] - 1e-9 and
                  float(np.min(r["sL"][:10])) >= S_FLOOR - 1e-9)
        if ok:
            last_pass = float(g)
        else:
            break
    return last_pass


print("tau   fiscal-gate   speed-gate   both (operative)")
for tau in TAUS:
    fg = critical_gate(tau, "fiscal")
    sg = critical_gate(tau, "speed")
    bg = critical_gate(tau, "both")
    print("%.2f   %-12s %-12s %s" % (tau, fg, sg, bg))
