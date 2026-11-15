# -*- coding: utf-8 -*-
"""Controlled comparison: fixed vs dynamic workfare wage under identical accounting.

Runs the three calibrated policy configurations with the exogenous reinstatement
special case (omega=0.05) used in the named check scripts, so the results are
directly comparable to the currently published numbers.
"""
import os, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIMDIR = os.path.join(ROOT, "simulation")

_spec = importlib.util.spec_from_file_location(
    "U", os.path.join(SIMDIR, "double_task_sim_unified.py"))
U = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(U)

CALIB = dict(eta=0.40, mu_TM=0.50, omega_mode="exogenous", omega=0.05)
POL = {
    "CORNER":        dict(CALIB, tau=0.15, pE=0.45, phi_max=0.02),
    "STANDARD":      dict(CALIB, tau=0.16, pE=0.45, phi_max=0.65),
    "PURE_TRANSFER": dict(CALIB, tau=0.16, pE=0.00, phi_max=0.65),
}

for name, cfg in POL.items():
    print("=" * 92)
    print(name)
    for mode in ("fixed", "dynamic"):
        par = U.Params(wP_mode=mode, **cfg)
        r = U.simulate_unified(par)
        minC = float(np.min(r["C"][10:])); Cmin = float(r["Cmin"])
        w, wp = r["w_real"], r["wP"]
        cross = [t for t in range(par.T) if wp[t] >= w[t] + 1e-12]
        first_cross = cross[0] if cross else None
        print("  %-7s outcome=%-11s crash=%-5s minC=%.4f (margin %+.4f) sL_end=%.4f "
              "EP_end=%.4f wP_end=%.3f wP_min=%.3f first_cross=%s"
              % (mode, r["outcome"], r["crash_type"], minC, minC - Cmin,
                 r["sL_end"], float(r["EP"][-1]), float(wp[-1]),
                 float(wp.min()), first_cross))
