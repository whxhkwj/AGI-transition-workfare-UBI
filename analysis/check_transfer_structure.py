# -*- coding: utf-8 -*-
"""Structural decomposition of why the pure-transfer regime fails.

Reproduces the diagnostic requested in review point 2.6 (its Table 4): holding
the pure-transfer fiscal setting fixed (tau=0.16, p_E=0, phi_max=0.65), vary
the two structural choices that the workfare corner does not share:
  * the transfer consumption MPC mu_TM (workfare wages enter consumption at 1);
  * the social-instability coefficient alpha_s (the independent social channel).
All numbers are produced directly from the unified engine through the v6
exogenous entry; no value is hard-coded.
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIMDIR = os.path.join(os.path.dirname(_HERE), "simulation")

_spec = importlib.util.spec_from_file_location(
    "v6", os.path.join(_SIMDIR, "double_task_sim_v6.py"))
v6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v6)

FLOOR = 0.30


def run(label, mu_TM, alpha_s):
    par = v6.Params(eta=0.40, tau=0.16, pE=0.0, phi_max=0.65,
                    mu_TM=mu_TM, alpha_s=alpha_s)
    r = v6.run_case(par)
    print("%-30s outcome=%-11s crash=%-6s minC=%.4f" % (
        label, r["outcome"], r["crash_type"], r["minC"]))
    return r


print("=" * 70)
print("Pure-transfer structural decomposition (same fiscal setting)")
print("=" * 70)
base = run("baseline (mu=.50, alpha=.60)", 0.50, 0.60)
mpc = run("equal MPC (mu=1.00)", 1.00, 0.60)
soc = run("social disabled (alpha=0)", 0.50, 0.0)
both = run("both (mu=1.00, alpha=0)", 1.00, 0.0)

checks = [
    ("baseline crashes through both channels",
     base["outcome"] == "crash" and base["crash_type"] == "both"),
    ("equal MPC holds consumption, only social crash",
     mpc["crash_type"] == "social" and mpc["minC"] >= FLOOR),
    ("social disabled still crashes economically",
     soc["crash_type"] == "econ" and soc["minC"] < FLOOR),
    ("both relaxed is smooth", both["outcome"] == "smooth"),
]
print("\nchecks against the manuscript")
bad = 0
for name, ok in checks:
    print("  %-44s %s" % (name, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
print("\n%d/%d checks passed" % (len(checks) - bad, len(checks)))
raise SystemExit(1 if bad else 0)
