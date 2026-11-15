# -*- coding: utf-8 -*-
"""Plan C (tau fiscal recycling) + Plan A (mpc_cap capital-owner consumption).

Baseline corner calibration: pE=0.45, phi_max=0.02, dynamic workfare default.
Everything is computed from the engine; no hard-coded results.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "simulation"))
import numpy as np
import double_task_sim_unified as U

EW = 10  # early window


def run(tau, mpc_cap):
    p = U.Params(tau=tau, pE=0.45, phi_max=0.02, mpc_cap=mpc_cap)
    r = U.simulate_unified(p)
    GDP = float(np.sum(r["GDP"]))
    after_cap = float(np.sum(r["cap_inc"] - r["tax"]))
    uncons = float(np.sum((1.0 - p.mu_TM) * r["TM"]))
    return dict(
        outcome=r["outcome"],
        minC_all=float(np.min(r["C"][1:])),
        minC_late=float(np.min(r["C"][EW:])),
        C_end=float(r["C"][-1]),
        C_sum=float(np.sum(r["C"])),
        cap_leak=after_cap,
        cap_share=100.0 * after_cap / GDP,
        tm_leak=uncons,
        P_end=float(r["P"][-1]),
        sL_end=float(r["sL"][-1]),
        EP_end=float(r["EP"][-1]),
        Urem_end=float(r["Urem"][-1]),
    )


HDR = ("%-22s %-8s %8s %8s %8s %9s %9s %8s %7s %7s"
       % ("scenario", "outcome", "minC", "minC10", "C_end", "C_sum",
          "cap_leak", "cap%", "P_end", "Urem"))


def show(label, tau, mpc_cap):
    d = run(tau, mpc_cap)
    print("%-22s %-8s %8.4f %8.4f %8.4f %9.3f %9.3f %7.1f%% %7.3f %7.3f"
          % (label, d["outcome"], d["minC_all"], d["minC_late"], d["C_end"],
             d["C_sum"], d["cap_leak"], d["cap_share"], d["P_end"],
             d["Urem_end"]))


print("=" * 98)
print("Baseline corner: tau=0.15, pE=0.45, phi_max=0.02")
print("=" * 98)

print("\n--- Plan C: fiscal recycling via higher tau (mpc_cap=0) ---")
print(HDR)
for tau in (0.15, 0.20, 0.25, 0.30, 0.40):
    show("tau=%.2f" % tau, tau, 0.0)

print("\n--- Plan A: capital-owner MPC (tau fixed at 0.15) ---")
print(HDR)
for m in (0.0, 0.025, 0.05, 0.10, 0.15):
    show("mpc_cap=%.3f" % m, 0.15, m)

print("\n--- Plan C + A combined ---")
print(HDR)
for tau in (0.20, 0.25):
    for m in (0.05, 0.10):
        show("tau=%.2f mpc=%.2f" % (tau, m), tau, m)

print("\ndone")
