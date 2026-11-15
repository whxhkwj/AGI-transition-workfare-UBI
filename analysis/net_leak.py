# -*- coding: utf-8 -*-
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "simulation"))
import numpy as np
import double_task_sim_unified as U

print("tau   mpc_cap  after-tax cap  recycled   net cap leak  outcome   minC")
for tau in (0.15, 0.20, 0.25):
    for m in (0.05, 0.10):
        r = U.simulate_unified(U.Params(tau=tau, pE=0.45, phi_max=0.02,
                                        mpc_cap=m))
        after = float(np.sum(r["cap_inc"] - r["tax"]))
        recycled = m * after
        net = after - recycled
        print("%.2f  %.2f     %12.3f  %9.3f  %12.3f  %-8s  %.4f"
              % (tau, m, after, recycled, net, r["outcome"], np.min(r["C"][1:])))

print("\n--- Plan A net leakage at tau=0.15 ---")
for m in (0.0, 0.025, 0.05, 0.10, 0.15):
    r = U.simulate_unified(U.Params(tau=0.15, pE=0.45, phi_max=0.02,
                                    mpc_cap=m))
    after = float(np.sum(r["cap_inc"] - r["tax"]))
    print("mpc=%.3f  after-tax=%.3f  recycled=%.3f  net leak=%.3f"
          % (m, after, m * after, (1 - m) * after))
