# -*- coding: utf-8 -*-
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "simulation"))
import numpy as np
import double_task_sim_unified as U

print("tau   cum_avail  cum_wage(wP*EP)  cum_TM   cum_spent   idle   idle%")
for tau in (0.15, 0.25, 0.40):
    r = U.simulate_unified(U.Params(tau=tau, pE=0.45, phi_max=0.02))
    # available at period t is tax[t-1]
    avail = float(np.sum(r["tax"][:-1]))
    wage = float(np.sum(r["wP"][1:] * r["EP"][1:]))
    tm = float(np.sum(r["TM"][1:]))
    spent = wage + tm
    idle = avail - spent
    print("%.2f  %9.3f  %14.3f  %8.3f  %9.3f  %7.3f  %5.1f%%"
          % (tau, avail, wage, tm, spent, idle, 100.0 * idle / avail))
