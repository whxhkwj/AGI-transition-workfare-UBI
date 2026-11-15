# -*- coding: utf-8 -*-
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "simulation"))
import numpy as np
import double_task_sim_unified as U

tau = 0.10
print("tau=0.10: full gA curves, mpc=0 vs mpc=0.10")
print("gA      mpc0 (outcome,minC,type)            mpc0.10 (outcome,minC,type)")
for ga in np.linspace(0.01, 0.18, 18):
    rows = []
    for m in (0.0, 0.10):
        r = U.simulate_unified(U.Params(eta=0.40, mu_TM=0.50, phi_max=0.02,
                                        pe_policy="full_absorption", tau=tau,
                                        gA=float(ga), mpc_cap=m))
        rows.append((r["outcome"], np.min(r["C"][1:]), r["crash_type"]))
    print("%.4f  %-7s %.4f %-10s   %-7s %.4f %-10s"
          % (ga, rows[0][0], rows[0][1], rows[0][2],
             rows[1][0], rows[1][1], rows[1][2]))

print("\nWhat happens just above the ~0.138 ceiling at high mpc (tau=.20, mpc=.15)?")
for ga in (0.135, 0.139, 0.142, 0.15):
    r = U.simulate_unified(U.Params(eta=0.40, mu_TM=0.50, phi_max=0.02,
                                    pe_policy="full_absorption", tau=0.20,
                                    gA=ga, mpc_cap=0.15))
    print("gA=%.3f %-7s minC=%.4f crash_type=%s"
          % (ga, r["outcome"], np.min(r["C"][1:]), r["crash_type"]))
