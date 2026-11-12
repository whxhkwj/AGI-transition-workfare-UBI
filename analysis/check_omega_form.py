# -*- coding: utf-8 -*-
"""Functional-form sensitivity of the omega saturation, judged by the real BEB.

Three saturation laws share the same endpoints, monotonicity and smoothness:
  exp:      omega_t = 1 - (1-omega0)*exp(-Gamma)
  rational: omega0 + (1-omega0)*Gamma/(1+Gamma)
  clip:     min(1, omega0 + (1-omega0)*Gamma/KAPPA_REF)

A real crossing is dated when effective capture omega*skill reaches EFF_CRIT,
not when the nominal omega crosses the static value.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import importlib.util
import os
import sys
sys.path.insert(0, _HERE)
sys.path.insert(0, _SIMDIR)
import numpy as np
import dynamic_threshold as dt

KAPPA_REF = 0.95 / 0.6005


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_SIMDIR, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v6 = load("v6_omegaform", "double_task_sim_v6.py")
v12 = load("v12_omegaform", "double_task_sim_v12_endogenous_omega.py")

src = open(os.path.join(_SIMDIR, "double_task_sim_v12_endogenous_omega.py"),
           encoding="utf-8").read()
OLD = "omega_t = 1.0 - (1.0 - par.omega0) * np.exp(-Gamma)"
NEW = ("omega_t = (1.0 - (1.0 - par.omega0) * np.exp(-Gamma)) if par.form == 'exp' else "
       "(par.omega0 + (1.0 - par.omega0) * Gamma / (1.0 + Gamma)) if par.form == 'rational' else "
       "min(1.0, par.omega0 + (1.0 - par.omega0) * Gamma / KAPPA_REF)")
assert src.count(OLD) == 1, "omega line anchor not unique"
src = src.replace(OLD, NEW)
src = src.replace("OUT = os.path.dirname",
                  "KAPPA_REF = %r\nOUT = os.path.dirname" % KAPPA_REF, 1)
ns = {"__name__": "v12_omegaform_patch",
      "__file__": os.path.join(_SIMDIR, "double_task_sim_v12_endogenous_omega.py")}
exec(compile(src, "v12_omegaform_patch", "exec"), ns)
Patched = ns["Params"]
simulate = ns["simulate_v12"]


def metrics(form, gamma_pe):
    par = Patched(form=form, gamma_PE=gamma_pe, eta=0.40, mu_TM=0.50, T=200,
                  tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0)
    r = simulate(par)
    gap = np.maximum(0.0, par.w * par.LS - r["C"])
    tm_cost = 0.5 * r["TM"]
    pe_cost = 0.6 * par.wP * r["EP"] + 0.5 * r["EP"] ** 2
    dwl = 0.6 * par.tau ** 2 * r["tax"]
    total = float((gap + tm_cost + pe_cost + dwl).sum())
    tcross = dt.first_real_crossing(r["omega"], r["skill"])
    return dict(form=form, gam=gamma_pe, outcome=r["outcome"], crash=r["crash_type"],
                minC=float(np.min(r["C"][10:])), total=total, tcross=tcross,
                om_end=float(r["omega"][-1]))


cases = [("exp", 3.0), ("rational", 3.0), ("clip", 3.0),
         ("rational", 3.6), ("rational", 4.2), ("rational", 4.8),
         ("clip", 2.4), ("exp", 2.4)]
rows = [metrics(f, g) for f, g in cases]
print("%-9s %5s %-11s %-6s %7s %8s %9s %8s" %
      ("form", "gam", "outcome", "crash", "minC", "total", "real_cross", "om_end"))
for r in rows:
    print("%-9s %5.1f %-11s %-6s %7.4f %8.3f %8s %8.4f" %
          (r["form"], r["gam"], r["outcome"], r["crash"], r["minC"],
           r["total"], "-" if r["tcross"] is None else r["tcross"], r["om_end"]))
