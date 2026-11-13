# -*- coding: utf-8 -*-
"""Quarterly-to-annual frequency conversion for the nominal rigidities.

Locks the duration-based frequency conversion that addresses the external
review point on time units. The model period is one year and the reduced-form
slopes gamma_p and lambda_w are defined at the annual rate. The quarterly Calvo
reference fixes only the duration: a quarterly no-reset probability maps to an
annual no-reset probability by compounding over four quarters, and the implied
mean annual duration is reported. The quarterly structural slope is not mapped
period-by-period.
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIMDIR = os.path.join(os.path.dirname(_HERE), "simulation")

_spec = importlib.util.spec_from_file_location(
    "v12", os.path.join(_SIMDIR, "double_task_sim_v12_endogenous_omega.py"))
v12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v12)

par = v12.Params()
zeta_q = 0.75
quarters_per_year = 4
zeta_y = zeta_q ** quarters_per_year
annual_duration = 1.0 / (1.0 - zeta_y)
quarterly_duration = 1.0 / (1.0 - zeta_q)

print("=" * 70)
print("Quarterly-to-annual frequency conversion")
print("=" * 70)
print("quarterly zeta_q = %.4f, mean duration = %.2f quarters"
      % (zeta_q, quarterly_duration))
print("annual zeta_y = zeta_q^4 = %.4f, mean duration = %.4f years"
      % (zeta_y, annual_duration))
print("annual reduced-form slopes gamma_p=%.2f lambda_w=%.2f"
      % (par.gamma_p, par.lambda_w))

checks = [
    ("annual zeta matches 0.316", abs(zeta_y - 0.316) < 0.002),
    ("annual duration matches 1.46", abs(annual_duration - 1.46) < 0.01),
    ("quarterly duration is 4 quarters", abs(quarterly_duration - 4.0) < 1e-9),
    ("quarterly duration is one year",
     abs(quarterly_duration / quarters_per_year - 1.0) < 1e-9),
    ("annual zeta is below quarterly zeta", zeta_y < zeta_q),
    ("slopes are positive annual conventions",
     par.gamma_p > 0 and par.lambda_w > 0),
]

print("\nchecks against the manuscript")
bad = 0
for name, ok in checks:
    print("  %-44s %s" % (name, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
print("\n%d/%d checks passed" % (len(checks) - bad, len(checks)))
raise SystemExit(1 if bad else 0)
