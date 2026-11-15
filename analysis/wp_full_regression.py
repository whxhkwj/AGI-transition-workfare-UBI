# -*- coding: utf-8 -*-
"""Full regression under the workfare-wage upgrade.

Crosses the two wage rules (fixed / dynamic) with the two wage-Phillips slack
measures (market / residual) over every policy configuration used in the paper,
under both the exogenous reinstatement special case and the endogenous engine.
"""
import os, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIMDIR = os.path.join(ROOT, "simulation")

_spec = importlib.util.spec_from_file_location(
    "U", os.path.join(SIMDIR, "double_task_sim_unified.py"))
U = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(U)

# Exogenous-reinstatement configurations (omega=0.05), matching the check scripts.
EXO = {
    "No policy":      dict(tau=0.05, pE=0.00, phi_max=0.15),
    "Corner":         dict(tau=0.15, pE=0.45, phi_max=0.02),
    "Standard":       dict(tau=0.16, pE=0.45, phi_max=0.65),
    "Pure transfer":  dict(tau=0.16, pE=0.00, phi_max=0.65),
}
# Endogenous-engine configurations.
ENDO = {
    "No policy":        dict(tau=0.05, pE=0.00, phi_max=0.15),
    "Corner":           dict(tau=0.15, pE=0.45, phi_max=0.02),
    "Strong floor":     dict(tau=0.15, pE=0.80, phi_max=0.65, retrain=0.8),
    "Full absorption":  dict(tau=0.15, pE=1.00, phi_max=0.65),
}


def run(engine, configs):
    for name, cfg in configs.items():
        print("-" * 100)
        print(name)
        for slack in ("market", "residual"):
            for mode in ("fixed", "dynamic"):
                base = dict(eta=0.40, mu_TM=0.50, wP_mode=mode, wage_slack=slack)
                if engine == "exo":
                    base.update(omega_mode="exogenous", omega=0.05)
                base.update(cfg)
                par = U.Params(**base)
                r = U.simulate_unified(par)
                minC = float(np.min(r["C"][10:]))
                cross = [t for t in range(par.T)
                         if r["wP"][t] >= r["w_real"][t] + 1e-12]
                print("  %-8s %-7s outcome=%-11s minC=%.4f sL=%.3f EP=%.3f "
                      "w_end=%.3f wP_end=%.3f cross=%s"
                      % (slack, mode, r["outcome"], minC, r["sL_end"],
                         r["EP"][-1], r["w_real"][-1], r["wP"][-1],
                         cross[0] if cross else "none"))


print("=" * 100)
print("EXOGENOUS REINSTATEMENT (omega=0.05)")
run("exo", EXO)
print("\n" + "=" * 100)
print("ENDOGENOUS ENGINE")
run("endo", ENDO)
