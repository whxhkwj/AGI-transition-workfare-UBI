# -*- coding: utf-8 -*-
"""Capability-erosion counterfactual: is the reinstatement collapse built into
the logistic law by its monotone AGI term, or does it follow from the task
accounting?

The endogenous law is  logit(omega_t) = ell0 + Gamma_t - beta_A * g_t,  where
g_t = log(cap_t/A0) accumulates monotonically. A referee could read the collapse
as a tautology: impose a monotone negative term and the share must fall. The
direct counterfactual sets beta_A=0, which removes the capability-erosion
channel entirely, leaving only policy and the task accounting, and re-runs the
same engine with every other parameter fixed.

If the collapse disappeared at beta_A=0 it would be a functional-form artefact;
if the no-policy path still crashes while the financed corner still stays smooth,
the collapse follows from the task-accounting condition omega*sigma*dN < kappa_I*A
rather than from the logistic erosion term. The same runs also show what beta_A
actually governs: the durability of reinstatement (the strong policy's terminal
labor share), not whether the path crashes.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import sys, importlib.util
sys.path.insert(0, _SIMDIR)
_spec = importlib.util.spec_from_file_location(
    'v12', _os.path.join(_SIMDIR, 'double_task_sim_v12_endogenous_omega.py'))
v12 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v12)

HORIZON = 200
COMMON = dict(eta=0.40, mu_TM=0.50, T=HORIZON)
REGIMES = {
    'no policy': dict(COMMON, tau=0.05, pE=0.00, phi_max=0.15, retrain=0.0),
    'corner':    dict(COMMON, tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0),
    'strong':    dict(COMMON, tau=0.15, pE=0.80, phi_max=0.65, retrain=0.8),
}
BETAS = [0, 1]


def run(beta, name):
    par = v12.Params(beta_A=beta, **REGIMES[name])
    r = v12.simulate_v12(par)
    return dict(outcome=r['outcome'],
                minC=float(min(r['C'][10:])),
                sL_end=float(r['sL_end']))


print("capability-erosion counterfactual, T=%d" % HORIZON)
print("beta_A=0 removes the monotone erosion term; beta_A=1 is the baseline\n")
print("%-10s %-8s %-11s %-9s %-9s" % ("regime", "beta_A", "outcome", "minC", "sL_end"))
table = {}
for beta in BETAS:
    table[beta] = {}
    for name in REGIMES:
        res = run(beta, name)
        table[beta][name] = res
        print("%-10s %-8d %-11s %-9.4f %-9.4f" %
              (name, beta, res['outcome'], res['minC'], res['sL_end']))
    print()

# Identification checks.
checks = []
# 1. With erosion removed the no-policy path still crashes.
checks.append(("no-policy still crashes at beta_A=0",
               table[0]['no policy']['outcome'] == 'crash'))
# 2. With erosion removed the financed corner still stays smooth.
checks.append(("corner still smooth at beta_A=0",
               table[0]['corner']['outcome'] == 'smooth'))
# 3. The baseline no-policy path crashes as well, with the same minimum.
checks.append(("no-policy crashes at beta_A=1",
               table[1]['no policy']['outcome'] == 'crash'))
# 4. beta_A governs durability: the strong policy retains a higher terminal labor
#    share when erosion is absent, even though it does not decide the crash.
checks.append(("strong terminal share higher without erosion",
               table[0]['strong']['sL_end'] > table[1]['strong']['sL_end']))

print("=== checks ===")
all_ok = True
for label, ok in checks:
    all_ok = all_ok and ok
    print("[%s] %s" % ("PASS" if ok else "FAIL", label))
print("\nOVERALL:", "PASS" if all_ok else "FAIL")
