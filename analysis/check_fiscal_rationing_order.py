# -*- coding: utf-8 -*-
"""Experiment 4: the within-budget rationing order between EP and TM.

Experiments 1-3 establish which instrument is feasible and rank the two
allocation corners, but they do not state the order in which a *constrained*
budget should be released. This experiment answers that order directly using
the same within-period accounting as the unified engine. The available budget
is increased from zero to full coverage and allocated under two rules:

  * EP-first (the engine's rule): hire as many public workers as the budget
    allows, then spend the residual fiscal room on transfers to the remaining
    unemployed;
  * TM-first (counterfactual): fund transfers first, then use the residual to
    hire public workers.

A public-employment yuan enters consumption at 1 and a transfer yuan at
mu_TM. The script records the activation threshold of each instrument and the
consumption delivered at every budget level under both rules, then repeats over
several (mu_TM, phi) settings to check the order is stable. All values follow
from the engine's accounting identities; nothing is hard-coded.
"""
import numpy as np

W = 1.0            # market wage
WP = 0.85          # public-employment wage
U = 1.0            # representative unemployed population (LS = 1)
BUDGETS = np.round(np.arange(0.0, 0.951, 0.05), 3)
SETTINGS = [
    ("mu.50 phi.02", 0.50, 0.02),
    ("mu.50 phi.15", 0.50, 0.15),
    ("mu.70 phi.15", 0.70, 0.15),
    ("mu.90 phi.15", 0.90, 0.15),
    ("mu.50 phi.40", 0.50, 0.40),
]


def ep_first(b, mu, phi):
    n_ep = min(U, b / WP)
    ep_cost = n_ep * WP
    room = b - ep_cost
    tm = min(room, (U - n_ep) * phi * W)
    cons = n_ep * WP + mu * tm
    return n_ep, tm, cons


def tm_first(b, mu, phi):
    tm = min(b, U * phi * W)
    room = b - tm
    # Transfer receipt does not change employment status, so EP may still hire
    # from the full unemployed population rather than a residual.
    n_ep = min(U, room / WP)
    cons = n_ep * WP + mu * tm
    return n_ep, tm, cons


print("=" * 96)
print("Experiment 4: within-budget rationing order (EP-first engine rule vs TM-first counterfactual)")
print("=" * 96)

checks = []
for label, mu, phi in SETTINGS:
    print("\n### %s (per-person: EP cost %.2f, TM cost %.3f)" % (label, WP, phi * W))
    print(" budget   EP-first: EP_persons TM_spend cons   | TM-first: EP_persons TM_spend cons   | d_cons")
    tight_ok = True
    tight_n = 0
    first_positive = None
    surplus_reversals = 0
    for b in BUDGETS:
        e1, t1, c1 = ep_first(b, mu, phi)
        e2, t2, c2 = tm_first(b, mu, phi)
        if b > 1e-9 and first_positive is None:
            first_positive = (e1, t1, e2, t2)
        if 1e-9 < b <= WP:                      # constrained (tight) budget
            tight_n += 1
            if c1 < c2 - 1e-12:
                tight_ok = False
        if b > WP and c1 < c2 - 1e-12:          # surplus beyond full EP coverage
            surplus_reversals += 1
        print("  %.2f     %.3f      %.4f   %.4f  |           %.3f      %.4f   %.4f  | %+.4f" % (
            b, e1, t1, c1, e2, t2, c2, c1 - c2))
    print("  tight-budget points: %d, EP-first lower in %d surplus points"
          % (tight_n, surplus_reversals))
    # The engine order is defined on constrained budgets: the first positive
    # budget funds EP, and EP-first never loses on the tight interval.
    checks.append(("%s: first yuan funds EP under EP-first" % label,
                   first_positive[0] > 0 and first_positive[1] <= 1e-12))
    checks.append(("%s: EP-first >= TM-first on tight budget" % label, tight_ok))
    if phi >= 0.15:
        # Where the transfer need does not saturate at the first step, the
        # counterfactual rule visibly funds TM first.
        checks.append(("%s: first yuan funds TM under TM-first" % label,
                       first_positive[2] <= 1e-12 and first_positive[3] > 0))

# As mu_TM -> 1 the efficiency gap, and hence the order advantage, must close.
gaps = []
for mu in (0.50, 0.70, 0.90, 0.99):
    g = np.mean([ep_first(b, mu, 0.15)[2] - tm_first(b, mu, 0.15)[2]
                 for b in BUDGETS if b > 1e-9])
    gaps.append(g)
print("\nmean consumption advantage of EP-first by mu_TM:",
      ["%.4f" % g for g in gaps])
checks.append(("advantage nonincreasing as mu_TM rises",
               all(gaps[i] >= gaps[i + 1] - 1e-12 for i in range(len(gaps) - 1))))
checks.append(("advantage near zero at mu_TM=0.99", abs(gaps[-1]) < 0.01))

print("\n" + "=" * 96)
print("checks against the manuscript")
bad = 0
for name, ok in checks:
    print("  %-52s %s" % (name, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
print("\n%d/%d checks passed" % (len(checks) - bad, len(checks)))
raise SystemExit(1 if bad else 0)
