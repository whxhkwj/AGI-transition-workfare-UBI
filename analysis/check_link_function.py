# -*- coding: utf-8 -*-
"""Link-function sensitivity: logistic vs probit share law.

Reproduces the functional-form paragraph (manuscript Section 5): the corner is
re-solved with the standard-normal CDF in place of the logistic sigmoid. The
reported effective-capture peaks are the policy-driven mid-horizon response,
excluding the t=0 initial transient. The buffer conclusion is unchanged.

Exit code is non-zero if any reproduced value falls outside the stated value.
"""
import importlib.util
import os
import sys
from math import erf

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "simulation"))
sys.path.insert(0, _HERE)

import double_task_sim_unified as engine  # noqa: E402
import dynamic_threshold as dt  # noqa: E402


def _norm_cdf(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _corner_path(link):
    if link == "probit":
        engine.sigmoid = _norm_cdf
    else:
        engine.sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    par = engine.Params(eta=0.40, mu_TM=0.50, T=80,
                        omega_mode="endogenous", omega=0.05,
                        tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0)
    return engine.simulate_unified(par)


def _mid_horizon_peak(r):
    eff = dt.effective_capture(r["omega"], r["skill"])
    return float(np.max(eff[1:]))  # exclude t=0 initial transient


def main():
    rows = {}
    for link in ("logistic", "probit"):
        r = _corner_path(link)
        rows[link] = (r["outcome"], _mid_horizon_peak(r), float(np.min(r["C"])))

    print("link       outcome   eff peak (t>=1)   min_t C_t")
    for link, (out, peak, mc) in rows.items():
        print("%-9s %-9s %-16.4f %.4f" % (link, out, peak, mc))

    expected = {
        "logistic": (0.121, 0.3562),
        "probit": (0.041, 0.3571),
    }
    ok = True
    for link, (ep, em) in expected.items():
        out, peak, mc = rows[link]
        if out != "smooth":
            ok = False
        if abs(peak - ep) > 0.0005 or abs(mc - em) > 0.0005:
            ok = False
            print("MISMATCH for %s: got peak %.4f minC %.4f" % (link, peak, mc))
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
