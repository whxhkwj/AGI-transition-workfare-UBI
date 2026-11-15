# -*- coding: utf-8 -*-
"""Same-budget transfer-share experiment under the upgraded engine.

A fixed share s_TM of each period's available tax revenue is reserved for
transfers before public employment is funded, replacing the workfare-first
order. Scans s_TM from zero to one and reports the minimum consumption and the
critical frontier gA, reproducing the robustness paragraph that the workfare
ranking survives when transfers actually spend money.
"""
import os, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIMDIR = os.path.join(ROOT, "simulation")

_spec = importlib.util.spec_from_file_location(
    "U", os.path.join(SIMDIR, "double_task_sim_unified.py"))
U = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(U)
_v4 = U.v4


def simulate_share(par, s_tm):
    """Copy of the engine with the fiscal allocation split by s_tm."""
    T = par.T
    It = np.zeros(T); Nt = np.zeros(T); nlab = np.zeros(T)
    LD = np.zeros(T); sL = np.zeros(T); GDP = np.zeros(T)
    cap_inc = np.zeros(T); tax = np.zeros(T); P = np.zeros(T)
    D = np.zeros(T); C = np.zeros(T); pi = np.zeros(T); phi = np.zeros(T)
    u = np.zeros(T); i_arr = np.zeros(T); rd = np.zeros(T); b = np.zeros(T)
    w_real = np.zeros(T); wP_arr = np.zeros(T); EP = np.zeros(T)
    Urem = np.zeros(T); TM = np.zeros(T); skill = np.zeros(T)
    stress = np.zeros(T); omega_arr = np.zeros(T); net_b_arr = np.zeros(T)

    It[0], Nt[0], nlab[0] = par.I0, par.N0, par.N0 - par.I0
    LD[0] = par.LS * _v4.clamp(nlab[0] / Nt[0], 0, 1)
    sL[0] = LD[0] / par.LS
    GDP[0] = 1.0; P[0], D[0] = 1.0, par.D0
    w_real[0] = 1.0
    wP_arr[0] = max(par.wP_floor, par.wP * w_real[0])
    phi[0] = par.phi_base
    u[0] = 1 - LD[0] / par.LS; EP[0] = 0.0
    Urem[0] = u[0] * par.LS
    TM[0] = phi[0] * par.w * Urem[0]
    cap_inc[0] = max(0.0, GDP[0] - w_real[0] * LD[0])
    tax[0] = par.tau * cap_inc[0]
    skill[0] = 1.0
    stress[0] = par.alpha_s * (Urem[0] / par.LS)
    C[0] = w_real[0] * LD[0] + wP_arr[0] * EP[0] + par.mu_TM * TM[0]
    b[0] = D[0] / P[0]
    ell0 = np.log(par.omega0 / (1.0 - par.omega0))
    Cfe = par.w * par.LS
    g_eff = par.gG * (1 - par.eta * par.tau)
    cap_path = par.A0 * (1.0 + par.gA) ** np.arange(T)

    for t in range(1, T):
        Gamma = (par.gamma_PE * EP[t - 1] / par.LS
                 + par.gamma_TM * TM[t - 1] / (par.w * par.LS)
                 + par.gamma_R * par.retrain * skill[t - 1])
        g_t = np.log(cap_path[t] / par.A0)
        omega_t = U.sigmoid(ell0 + Gamma - par.beta_A * g_t)
        omega_arr[t] = omega_t
        cap = cap_path[t]
        Sh = par.S0 * max(0.0, 1 - par.lambda_h * (It[t - 1] / Nt[t - 1]))
        dN = par.kappa_N_h * Sh + par.kappa_N_AI * cap * par.M0
        dI_auto_gross = par.kappa_I * cap
        eff = _v4.clamp(omega_t * skill[t - 1], 0.0, 1.0)
        dN_L = eff * dN; dN_AI = (1 - eff) * dN
        dI_auto = min(dI_auto_gross, nlab[t - 1] + dN_L)
        Nt[t] = Nt[t - 1] + dN
        It[t] = It[t - 1] + dN_AI + dI_auto
        nlab[t] = max(0.0, Nt[t] - It[t])
        LD[t] = par.LS * _v4.clamp(nlab[t] / Nt[t], 0, 1)
        sL[t] = LD[t] / par.LS
        u[t] = 1 - LD[t] / par.LS

        slack = Urem[t - 1] / par.LS
        w_real[t] = max(0.35, w_real[t - 1] * (1 + par.gW - par.lambda_w * slack))
        wP_arr[t] = max(par.wP_floor, par.wP * w_real[t])

        ud = u[max(0, t - par.d)]
        phi[t] = min(par.phi_max, par.phi_base + par.resp * max(0.0, ud))
        budget = tax[t - 1]
        # Reserve s_tm of the budget for transfers; fund public jobs with the rest.
        tm_budget = s_tm * budget
        ep_budget = budget - tm_budget
        EP[t] = min(par.pE * u[t] * par.LS, u[t] * par.LS,
                    ep_budget / max(wP_arr[t], 1e-12))
        Urem[t] = max(0.0, u[t] * par.LS - EP[t])
        TM[t] = min(phi[t] * par.w * Urem[t], tm_budget)

        dep = par.delta_skill * (1 - par.retrain * par.retain_retrain)
        employed = (LD[t] + EP[t]) / par.LS
        unemployed = Urem[t] / par.LS
        skill[t] = max(0.30, skill[t - 1] * (employed + unemployed * (1 - dep)))
        disp = (It[t] - par.I0) / par.N0
        stress[t] = par.alpha_s * (Urem[t] / par.LS)
        GDP[t] = (1 + g_eff) ** t * (1 + par.gDisp * disp)
        cap_inc[t] = max(0.0, GDP[t] - w_real[t - 1] * LD[t])
        tax[t] = par.tau * cap_inc[t]
        pi[t] = max(par.pi_lb, par.gamma_p * (C[t - 1] / Cfe - 1))
        P[t] = P[t - 1] * (1 + pi[t])
        i_arr[t] = max(0.0, par.rn + pi[t])
        rd[t] = i_arr[t] - pi[t]
        Yd = (w_real[t] * LD[t] + wP_arr[t] * EP[t] + par.mu_TM * TM[t]
              - par.mu * max(0.0, rd[t]))
        D_rolled = D[t - 1] * (1 + i_arr[t])
        desired_net = (par.debt_share * max(0.0, Cfe - Yd)
                       - par.chi_amort * max(0.0, D_rolled - par.bbar * P[t]))
        D[t] = _v4.clamp(D_rolled + desired_net, 0.0, par.b_credit * P[t])
        net_b_arr[t] = D[t] - D_rolled
        b[t] = D[t] / P[t]
        C[t] = max(0.0, Yd + net_b_arr[t] / P[t])

    Cmin = par.q * par.w * par.LS
    return dict(C=C, sL=sL, Cmin=Cmin, minC=float(np.min(C[10:])),
                EP=EP, Urem=Urem, TM=TM)


def critical_gA(s_tm):
    lo, hi = 0.005, 0.25
    for _ in range(20):
        mid = (lo + hi) / 2
        par = U.Params(tau=0.15, pE=1.0, phi_max=0.65, gA=mid)
        r = simulate_share(par, s_tm)
        if r["minC"] >= r["Cmin"] - 1e-9:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


print("s_TM   minC(gA=0.010)   critical gA")
for s in [0.0, 0.25, 0.50, 1.0]:
    par = U.Params(tau=0.15, pE=1.0, phi_max=0.65, gA=0.010)
    r = simulate_share(par, s)
    print("%.2f   %.4f          %.4f" % (s, r["minC"], critical_gA(s)))
