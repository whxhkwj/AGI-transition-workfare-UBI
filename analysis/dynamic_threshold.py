# -*- coding: utf-8 -*-
"""Dynamic BEB threshold shared by the bifurcation analysis scripts.

The fast-subsystem boundary equilibrium condition is on the effective labor
capture, not on the nominal reinstatement rate:

    omega_t * skill[t-1] = kappa_I/q + s_floor  (= EFF_CRIT = 0.5667).

Holding skill fixed at one gives the static value omega_c = 0.5667. On the full
dynamic path skill depreciates, so the instantaneous threshold on the nominal
rate is omega_c(t) = EFF_CRIT / skill[t-1] and rises as skill falls.
"""
import numpy as np

# kappa_I / q + s_floor, with kappa_I=0.020, q=0.050, lambda_h=1.20
EFF_CRIT = 0.020 / 0.050 + (1.0 - 1.0 / 1.20)


def _lagged_skill(skill):
    """skill[t-1] used by the engine's within-period capture (t=0 -> 1)."""
    skill = np.asarray(skill, dtype=float)
    return np.concatenate(([1.0], skill[:-1]))


def effective_capture(omega, skill):
    """eff[t] = omega[t] * skill[t-1], matching the engine."""
    return np.asarray(omega, dtype=float) * _lagged_skill(skill)


def dynamic_threshold(skill):
    """Instantaneous BEB threshold omega_c(t) = EFF_CRIT / skill[t-1]."""
    return EFF_CRIT / _lagged_skill(skill)


def first_real_crossing(omega, skill):
    """First period the effective capture reaches EFF_CRIT, else None."""
    eff = effective_capture(omega, skill)
    hit = np.where(eff >= EFF_CRIT)[0]
    hit = hit[hit > 0]
    return int(hit[0]) if len(hit) else None
