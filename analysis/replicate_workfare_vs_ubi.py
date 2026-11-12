# -*- coding: utf-8 -*-
"""Reproduce the workfare-versus-transfer row of the results table and the
load-bearing paragraph of Section 5.2.

Six values of omega, two policy settings, two public-employment shares.
The workfare and pure-transfer runs share the same tax rate and transfer
ceiling, differing only in p_E, so the comparison is same-fiscal.
Paper claims (balance-sheet-consistent engine, hard credit ceiling):
  * standard (tau=0.16, phi_max=0.65), p_E=0.45: workfare survives all six
    (min C = 0.3510 at omega=0.02 and 0.3576 at omega=0.05);
  * standard, pure transfer p_E=0.00: all six crash;
  * corner (tau=0.15, phi_max=0.02), p_E=0.45: all six survive, min C = 0.352-0.357;
  * corner, p_E=0.30: all six crash, min C = 0.244-0.265.
"""
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import importlib.util

_SPEC = importlib.util.spec_from_file_location(
    "v6", _os.path.join(_SIMDIR, "double_task_sim_v6.py"))
v6 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v6)

# Calibration used throughout the paper. Both entries MUST be passed
# explicitly: the engine's own defaults (mu_TM=0.70, eta=0.35) are left-overs
# from an earlier experiment and are not the paper's calibration.
CAL = dict(mu_TM=0.50, eta=0.40)
OMEGAS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
FLOOR = 0.30


def sweep(tau, phi_max, pE, label):
    print("\n%s  (tau=%.2f, phi_max=%.2f, pE=%.2f)" % (label, tau, phi_max, pE))
    print("  omega  outcome      crash   min C    total loss")
    rows = []
    for om in OMEGAS:
        r = v6.run_case(v6.Params(omega=om, tau=tau, phi_max=phi_max, pE=pE,
                                  retrain=0.0, **CAL))
        rows.append((om, r))
        print("  %5.2f  %-11s  %-5s  %8.4f  %9.2f" % (
            om, r["outcome"], r["crash_type"], r["minC"], r["total"]))
    survived = sum(1 for _, r in rows if r["outcome"] in ("smooth", "not_reached"))
    print("  -> survived %d of 6" % survived)
    return rows


print("=" * 68)
print("Workfare vs pure transfer, six omega values (same fiscal setting)")
print("calibration: mu_TM=0.50, eta=0.40  (engine defaults 0.70/0.35 are NOT used)")
print("=" * 68)

w_std = sweep(0.16, 0.65, 0.45, "[1] standard setting, workfare")
u_std = sweep(0.16, 0.65, 0.00, "[2] standard setting, pure transfer")
w_cor = sweep(0.15, 0.02, 0.45, "[3] corner, workfare at p_E=0.45")
w_c30 = sweep(0.15, 0.02, 0.30, "[4] corner, workfare at p_E=0.30")

print("\n[5] total loss, workfare (p_E=0.45) vs pure transfer (p_E=0.00), standard setting")
print("  omega  workfare   pure transfer  difference")
for (om, a), (_, b) in zip(w_std, u_std):
    print("  %5.2f  %9.2f  %9.2f  %+10.2f" % (om, a["total"], b["total"],
                                             a["total"] - b["total"]))

# ---- checks against the numbers printed in the paper ----
checks = [
    ("standard workfare survives 6 of 6",
     sum(1 for _, r in w_std if r["outcome"] in ("smooth", "not_reached")) == 6),
    ("standard pure transfer survives 0 of 6",
     sum(1 for _, r in u_std if r["outcome"] in ("smooth", "not_reached")) == 0),
    ("corner p_E=0.45 survives 6 of 6",
     sum(1 for _, r in w_cor if r["outcome"] in ("smooth", "not_reached")) == 6),
    ("corner p_E=0.30 survives 0 of 6",
     sum(1 for _, r in w_c30 if r["outcome"] in ("smooth", "not_reached")) == 0),
    ("workfare omega=0.05 min C = 0.3576", abs(w_std[1][1]["minC"] - 0.3576) < 1e-4),
    ("workfare omega=0.02 min C = 0.3510", abs(w_std[0][1]["minC"] - 0.3510) < 1e-4),
    ("corner p_E=0.45 range 0.352..0.357",
     abs(min(r["minC"] for _, r in w_cor) - 0.3517) < 1e-4
     and abs(max(r["minC"] for _, r in w_cor) - 0.3573) < 1e-4),
    ("corner p_E=0.30 range 0.244..0.265",
     abs(min(r["minC"] for _, r in w_c30) - 0.2445) < 1e-4
     and abs(max(r["minC"] for _, r in w_c30) - 0.2653) < 1e-4),
    ("workfare loss lower at every omega",
     all(a["total"] < b["total"] for (_, a), (_, b) in zip(w_std, u_std))),
]

print("\nchecks against the manuscript")
bad = 0
for name, ok in checks:
    print("  %-46s %s" % (name, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
print("\n%d/%d checks passed" % (len(checks) - bad, len(checks)))
raise SystemExit(1 if bad else 0)
