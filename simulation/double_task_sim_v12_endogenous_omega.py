# -*- coding: utf-8 -*-
"""
双层任务模型 v12 —— 内生复职率：公共就业/以工代赈/再培训驱动 omega
=================================================================================
v0.6 把 omega（新任务归劳动的比例）当作外生常数，结论是标定 omega=0.05 远低于
BEB 临界 omega_c = kappa_I/q + s_floor = 0.5667，政策无法介入，崩塌必然。

v12 关闭这个缺口：omega 不再是外生常数，而是政策强度的内生函数。
  公共就业（EP）、以工代赈（TM）、再培训（retrain）把劳动者重新接入新任务，
  从而提高新任务归劳动的比例：

      omega_t = 1 - (1 - omega0) * exp(-Gamma_t)

      Gamma_t = gamma_PE * EP[t-1]/LS
              + gamma_TM * TM[t-1]/(w*LS)
              + gamma_R  * retrain * skill[t-1]

  性质：
    * omega in [omega0, 1]：无政策时回到 omega0，强政策时趋于 1（纯社会福利极限）
    * 对各项政策单调、光滑、有饱和
    * 用上一期政策存量（政策效果有时滞，符合慢-快结构）

  BEB 仍由 omega_t = omega_c 触发；政策把 omega 推过 omega_c，高就业均衡恢复，
  这就是"用分岔理论避免崩塌"的可证机制。
=================================================================================
"""
import os
import numpy as np
import importlib.util

OUT = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('v6', os.path.join(OUT, 'double_task_sim_v6.py'))
v6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v6)
v4 = v6.v4


class Params(v6.Params):
    def __init__(self, **kw):
        super().__init__(**kw)
        # ---- 内生复职率参数 ----
        self.omega0 = 0.05        # 无政策基线（= v6 外生 omega）
        self.gamma_PE = 3.0       # 公共就业 -> omega 弹性
        self.gamma_TM = 2.0       # 以工代赈 -> omega 弹性
        self.gamma_R = 1.5        # 再培训 -> omega 弹性
        for k, v in kw.items():
            setattr(self, k, v)


def simulate_v12(par):
    T = par.T
    It = np.zeros(T); Nt = np.zeros(T); nlab = np.zeros(T)
    LD = np.zeros(T); sL = np.zeros(T)
    GDP = np.zeros(T); cap_inc = np.zeros(T); tax = np.zeros(T)
    P = np.zeros(T); D = np.zeros(T); C = np.zeros(T); pi = np.zeros(T)
    phi = np.zeros(T); u = np.zeros(T); i_arr = np.zeros(T); rd = np.zeros(T)
    b = np.zeros(T); w_real = np.zeros(T); omega_arr = np.zeros(T)
    EP = np.zeros(T); Urem = np.zeros(T); TM = np.zeros(T)
    Eplan = np.zeros(T); shortfall = np.zeros(T); cap_share = np.zeros(T)
    skill = np.zeros(T); stress = np.zeros(T); net_b_arr = np.zeros(T)

    It[0], Nt[0], nlab[0] = par.I0, par.N0, par.N0 - par.I0
    LD[0] = par.LS * v4.clamp(nlab[0] / Nt[0], 0, 1)
    sL[0] = LD[0] / par.LS
    GDP[0] = 1.0
    P[0], D[0] = 1.0, par.D0
    w_real[0] = 1.0
    phi[0] = par.phi_base
    u[0] = 1 - LD[0] / par.LS
    EP[0] = 0.0
    Urem[0] = u[0] * par.LS
    TM[0] = phi[0] * par.w * Urem[0]
    Eplan[0] = par.wP * EP[0] + TM[0]
    cap_inc[0] = max(0.0, GDP[0] - w_real[0] * LD[0])
    tax[0] = par.tau * cap_inc[0]
    cap_share[0] = 1.0
    skill[0] = (LD[0] * 1.0 + EP[0] * 1.0 + Urem[0] * (1 - par.delta_skill)) / par.LS
    stress[0] = par.alpha_s * (Urem[0] / par.LS)
    C[0] = w_real[0] * LD[0] + par.wP * EP[0] + par.mu_TM * TM[0]
    b[0] = D[0] / P[0]
    omega_arr[0] = par.omega0
    Cfe = par.w * par.LS
    g_eff = par.gG * (1 - par.eta * par.tau)

    for t in range(1, T):
        # ---- 内生复职率：上一期政策把劳动者重新接入新任务 ----
        Gamma = (par.gamma_PE * EP[t - 1] / par.LS
                 + par.gamma_TM * TM[t - 1] / (par.w * par.LS)
                 + par.gamma_R * par.retrain * skill[t - 1])
        omega_t = 1.0 - (1.0 - par.omega0) * np.exp(-Gamma)
        omega_arr[t] = omega_t

        cap = par.A0 * (1 + par.gA) ** t
        Sh = par.S0 * max(0.0, 1.0 - par.lambda_h * (It[t - 1] / Nt[t - 1]))
        dN = par.kappa_N_h * Sh + par.kappa_N_AI * cap * par.M0
        dI_auto_gross = par.kappa_I * cap
        # 闭合任务会计：新任务按劳动捕获效率 eff=omega_t·skill 在劳动与 AI 间分配，
        # 劳动得 eff·dN、AI 得 (1-eff)·dN；直接自动化受限于当前劳动任务。
        # 由此 n=N-I 恒等式逐期严格成立，内生复职/技能机制完整保留。
        eff = v4.clamp(omega_t * skill[t - 1], 0.0, 1.0)
        dN_L = eff * dN
        dN_AI = (1.0 - eff) * dN
        dI_auto = min(dI_auto_gross, nlab[t - 1] + dN_L)
        Nt[t] = Nt[t - 1] + dN
        It[t] = It[t - 1] + dN_AI + dI_auto
        nlab[t] = max(0.0, Nt[t] - It[t])
        LD[t] = par.LS * v4.clamp(nlab[t] / Nt[t], 0, 1)
        sL[t] = LD[t] / par.LS
        u[t] = 1 - LD[t] / par.LS

        ud = u[max(0, t - par.d)]
        phi[t] = min(par.phi_max, par.phi_base + par.resp * max(0.0, ud))
        EP_plan = min(par.pE * u[t] * par.LS, u[t] * par.LS)
        Urem_plan = max(0.0, u[t] * par.LS - EP_plan)
        TM_plan = phi[t] * par.w * Urem_plan
        Eplan[t] = par.wP * EP_plan + TM_plan
        Urem[t] = Urem_plan
        if Eplan[t] <= tax[t - 1] + 1e-9:
            EP[t], TM[t] = EP_plan, TM_plan
            shortfall[t] = 0.0
            cap_share[t] = 1.0
        else:
            shrink = tax[t - 1] / Eplan[t]
            EP[t], TM[t] = EP_plan * shrink, TM_plan * shrink
            shortfall[t] = Eplan[t] - tax[t - 1]
            cap_share[t] = shrink

        skill[t] = skill[t - 1] * (1 - par.delta_skill * (Urem[t] / par.LS)
                                   * (1 - par.retrain * par.retain_retrain))
        skill[t] = max(0.30, skill[t])

        disp = (It[t] - par.I0) / par.N0
        GDP[t] = (1 + g_eff) ** t * (1 + par.gDisp * disp)
        cap_inc[t] = max(0.0, GDP[t] - w_real[t - 1] * LD[t])
        tax[t] = par.tau * cap_inc[t]

        stress[t] = par.alpha_s * (Urem[t] / par.LS)

        pi[t] = max(par.pi_lb, par.gamma_p * (C[t - 1] / Cfe - 1.0))
        P[t] = P[t - 1] * (1 + pi[t])
        i_arr[t] = max(0.0, par.rn + pi[t])
        rd[t] = i_arr[t] - pi[t]
        w_real[t] = max(0.35, w_real[t - 1] * (1 + par.gW - par.lambda_w * u[t]))
        # 家庭资产负债表（存量-流量闭环）：净借贷统一决策，受银行授信硬上限约束。
        Yd = (w_real[t] * LD[t] + par.wP * EP[t] + par.mu_TM * TM[t]
              - par.mu * max(0.0, rd[t]))
        D_rolled = D[t - 1] * (1 + i_arr[t])
        desired_net = (par.debt_share * max(0.0, Cfe - Yd)
                       - par.chi_amort * max(0.0, D_rolled - par.bbar * P[t]))
        D[t] = v4.clamp(D_rolled + desired_net, 0.0, par.b_credit * P[t])
        net_b = D[t] - D_rolled
        net_b_arr[t] = net_b
        b[t] = D[t] / P[t]
        C[t] = max(0.0, Yd + net_b / P[t])

    Cmin = par.q * par.w * par.LS
    reached = sL[-1] < par.term_thresh
    C_below = np.min(C[10:]) < Cmin
    neg = pi[10:] < -0.001
    neg_streak = 0; streak = 0
    for x in neg:
        streak = streak + 1 if x else 0
        neg_streak = max(neg_streak, streak)
    debt_up = b[-1] > b[np.argmin(b)] + 1e-6
    spiral = reached and (neg_streak >= par.spiral_period) and debt_up and C_below
    econ_crash = C_below
    over = stress[10:] > par.stress_thresh
    soc_streak = 0; sstreak = 0
    for x in over:
        sstreak = sstreak + 1 if x else 0
        soc_streak = max(soc_streak, sstreak)
    social_crash = soc_streak >= par.soc_period
    crash = econ_crash or social_crash
    smooth = reached and not crash
    outcome = 'smooth' if smooth else ('crash' if crash else 'not_reached')
    if crash:
        crash_type = ('social' if social_crash and not econ_crash else
                      'econ' if econ_crash and not social_crash else 'both')
    else:
        crash_type = 'none'
    return dict(It=It, Nt=Nt, nlab=nlab, LD=LD, sL=sL, C=C, GDP=GDP,
                cap_inc=cap_inc, tax=tax, cap_share=cap_share, shortfall=shortfall,
                P=P, D=D, pi=pi, phi=phi, u=u, b=b, rd=rd, i_arr=i_arr, w_real=w_real,
                EP=EP, Urem=Urem, TM=TM, Eplan=Eplan, skill=skill, stress=stress,
                net_b=net_b_arr, omega=omega_arr,
                Cmin=Cmin, outcome=outcome, sL_end=sL[-1], C_end=C[-1],
                neg_streak=neg_streak, b_end=b[-1],
                crash_type=crash_type, social_crash=social_crash,
                tax_sum=float(tax.sum()), short_sum=float(shortfall.sum()),
                skill_end=float(skill[-1]), omega_end=float(omega_arr[-1]))


def run_case(par):
    r = simulate_v12(par)
    gap = np.maximum(0.0, par.w * par.LS - r['C'])
    tm_cost = 0.5 * r['TM']
    pe_cost = 0.6 * par.wP * r['EP'] + 0.5 * r['EP'] ** 2
    dwl = 0.6 * par.tau ** 2 * r['tax']
    total = float((gap + tm_cost + pe_cost + dwl).sum())
    return dict(outcome=r['outcome'], crash_type=r['crash_type'],
                total=total, minC=float(np.min(r['C'][10:])),
                omega_end=r['omega_end'], skill_end=r['skill_end'])
