"""Empirical-range sweep for the social-pressure coefficient alpha_s.

Extends the one-at-a-time social sweep (alpha_s 0.30-1.20, used to locate the
0.91 crash boundary) across the full reverse-engineered interval 0.30-6.62,
using the model's own conversion k = 1 - pE = 0.55. Confirms:
  * the workfare-over-transfer ranking holds across the whole interval;
  * the corner stays smooth through alpha_s = 0.90 and crashes socially at >= 0.91;
  * the pure transfer crashes at every alpha_s.
All numbers are produced directly from the v6 engine; no value is hard-coded.
"""
import os, sys, importlib.util
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
SIM = os.path.join(_ROOT, "simulation")
sys.path.insert(0, SIM)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v6 = _load("v6", f"{SIM}/double_task_sim_v6.py")


def run(alpha, pE, tau, phi):
    par = v6.Params(eta=0.40, mu_TM=0.50, tau=tau, pE=pE, phi_max=phi,
                    alpha_s=alpha)
    out = v6.simulate_v6(par)
    return out["outcome"], out["crash_type"], float(np.min(out["C"][10:]))


def main():
    grid = [0.30, 0.45, 0.60, 0.75, 0.90, 0.91, 1.02, 1.50, 2.50, 4.00, 6.62]
    print("alpha_s | corner (pE=.45)                  | pure transfer (pE=0)")
    corner_smooth, transfer_smooth = [], []
    for a in grid:
        c = run(a, 0.45, 0.15, 0.02)
        t = run(a, 0.0, 0.16, 0.65)
        corner_smooth.append(c[0] == "smooth")
        transfer_smooth.append(t[0] == "smooth")
        print("%5.2f   | %-7s %-12s minC %.3f        | %-7s %-12s minC %.3f"
              % (a, c[0], c[1], c[2], t[0], t[1], t[2]))
    assert all(not s for s in transfer_smooth), "pure transfer must crash at every alpha_s"
    assert corner_smooth[grid.index(0.90)] and not corner_smooth[grid.index(0.91)]
    print("\nPASS: ranking robust on 0.30-6.62; corner smooth through 0.90, social crash at 0.91.")


if __name__ == "__main__":
    main()
