# -*- coding: utf-8 -*-
"""
双层任务模型 统一引擎 —— 竞争式内生复职率（外生为特例）
=================================================================================
本模块是全仓库唯一的物理引擎。v6（外生 omega）与 v12（政策内生 omega）不再是
两套独立方程，而是本引擎在同一套任务/人口/财政会计下的两种模式：

  omega_mode="endogenous"（默认，主模型）
      劳动与 AGI 在新任务上按相对竞争力（logistic / 离散选择份额）竞争：

          logit(omega_t) = ell0 + Gamma_t - beta_A * g_t
          omega_t = sigmoid(...)

      其中
          ell0  = log(omega0/(1-omega0))                  基线 log-odds
          Gamma_t = gamma_PE EP[t-1]/L^S
                  + gamma_TM TM[t-1]/(w L^S)
                  + gamma_R retrain skill[t-1]            政策再附着存量
          g_t   = t log(1+g_A)                            AGI 能力的对数增长

      机制：AGI 能力 g_t 上升压低劳动新任务捕获份额（复职引擎内生崩塌）；
      公共就业/以工代赈/再培训通过 Gamma_t 对抗该下降。logistic 保证 omega
      始终落在 (0,1)，无需截断。

  omega_mode="exogenous"（退化特例）
      omega_t = omega（常数），即把复职率当作固定结构参数，用于对照与稳健性。
      此外 beta_A=0 给出"只随政策上升"的内生特例（对应旧 v12 方向）。

任务会计（eff=clamp(omega_t skill,0,1)，dN 分劳动/AI，dI 受可用劳动任务约束，
n=N-I 恒等式逐期严格成立）、人口核算（LD+EP+Urem=L^S）、财政平衡预算与家庭
资产负债表均与 #92-#97 强化后的口径完全一致。
=================================================================================
"""
import os, importlib.util
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location(
    "v4", os.path.join(OUT, "double_task_sim_v4.py"))
v4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v4)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Params(v4.Params):
    def __init__(self, **kw):
        super().__init__(**kw)
        # ---- v6 强化字段（闭合会计 / 净借贷 / 社会通道） ----
        self.pE = 0.30
        self.retrain = 0.0
        self.chi_amort = 0.15        # 超额债务净去杠杆速度
        self.b_credit = 0.80        # 银行授信硬上限（真实债务 b=D/P）
        self.delta_skill = 0.015
        self.retain_retrain = 1.0
        self.alpha_s = 0.60
        self.stress_thresh = 0.50
        self.soc_period = 10
        # ---- 论文权威标定（覆盖 v4 遗留默认 .35/.70）----
        self.eta = 0.40             # 资本税增长抑制
        self.mu_TM = 0.50           # 转移支付的总消费 MPC
        # ---- 统一内生复职率字段 ----
        self.omega_mode = "endogenous"
        self.omega0 = 0.05          # 无政策、初始能力基线（= 外生标定值）
        # 政策 -> 再附着 log-odds。logistic 份额在深基线下的中程响应弱于旧指数
        # 饱和形式，故相对旧 v12 的 (3,2,1.5) 放大 1.5 倍，使强政策在无 AGI
        # 侵蚀（beta_A=0）时仍能把有效捕获推过临界、恢复市场就业。
        self.gamma_PE = 4.5         # 公共就业 -> 再附着 log-odds
        self.gamma_TM = 3.0         # 以工代赈 -> 再附着 log-odds
        self.gamma_R = 2.25         # 再培训 -> 再附着 log-odds
        self.beta_A = 1.0           # AGI 能力 -> 复职份额侵蚀敏感度
        for k, v in kw.items():
            setattr(self, k, v)


def simulate_unified(par):
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
    ell0 = np.log(par.omega0 / (1.0 - par.omega0))
    omega_arr[0] = par.omega0 if par.omega_mode == "endogenous" else par.omega
    Cfe = par.w * par.LS
    g_eff = par.gG * (1 - par.eta * par.tau)

    # 可选外部能力路径（随机冲击 MC 用）；默认确定性复合增长。
    cap_path = getattr(par, "cap_path", None)
    if cap_path is None:
        cap_path = par.A0 * (1.0 + par.gA) ** np.arange(T)
    cap_path = np.asarray(cap_path, dtype=float)

    for t in range(1, T):
        # ---- 统一复职法则：内生竞争 / 外生常数 ----
        if par.omega_mode == "exogenous":
            omega_t = par.omega
        else:
            Gamma = (par.gamma_PE * EP[t - 1] / par.LS
                     + par.gamma_TM * TM[t - 1] / (par.w * par.LS)
                     + par.gamma_R * par.retrain * skill[t - 1])
            # 侵蚀项取实际能力的对数：log(cap/A0)，与 cap 路径一致。
            g_t = np.log(cap_path[t] / par.A0)
            omega_t = sigmoid(ell0 + Gamma - par.beta_A * g_t)
        omega_arr[t] = omega_t

        cap = cap_path[t]
        Sh = par.S0 * max(0.0, 1.0 - par.lambda_h * (It[t - 1] / Nt[t - 1]))
        dN = par.kappa_N_h * Sh + par.kappa_N_AI * cap * par.M0
        dI_auto_gross = par.kappa_I * cap
        # 闭合任务会计：新任务按 eff=omega_t*skill 在劳动与 AI 间分配。
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
        if Eplan[t] <= tax[t - 1] + 1e-9:
            EP[t], TM[t] = EP_plan, TM_plan
            shortfall[t] = 0.0
            cap_share[t] = 1.0
        else:
            # 预算绑定：公共就业与转移按同一比例压缩，未获融资者仍计入失业。
            shrink = tax[t - 1] / Eplan[t]
            EP[t], TM[t] = EP_plan * shrink, TM_plan * shrink
            shortfall[t] = Eplan[t] - tax[t - 1]
            cap_share[t] = shrink
        # 闭合人口核算：剩余失业按实际获融资的公共就业计，LD+EP+Urem=L^S。
        Urem[t] = max(0.0, u[t] * par.LS - EP[t])

        # 显式人口加权技能：在岗者技能保留，仅失业者折旧，再培训抵消。
        dep = par.delta_skill * (1.0 - par.retrain * par.retain_retrain)
        employed = (LD[t] + EP[t]) / par.LS
        unemployed = Urem[t] / par.LS
        skill[t] = skill[t - 1] * (employed + unemployed * (1.0 - dep))
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
        # 家庭资产负债表：净借贷统一决策，受授信硬上限约束；无消费截断。
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
    over = stress[10:] > par.stress_thresh
    soc_streak = 0; sstreak = 0
    for x in over:
        sstreak = sstreak + 1 if x else 0
        soc_streak = max(soc_streak, sstreak)
    social_crash = soc_streak >= par.soc_period
    econ_crash = C_below
    crash = econ_crash or social_crash
    smooth = reached and not crash
    outcome = "smooth" if smooth else ("crash" if crash else "not_reached")
    if crash:
        crash_type = ("social" if social_crash and not econ_crash else
                      "econ" if econ_crash and not social_crash else "both")
    else:
        crash_type = "none"
    return dict(It=It, Nt=Nt, nlab=nlab, LD=LD, sL=sL, C=C, GDP=GDP,
                cap_inc=cap_inc, tax=tax, cap_share=cap_share, shortfall=shortfall,
                P=P, D=D, pi=pi, phi=phi, u=u, b=b, rd=rd, i_arr=i_arr,
                w_real=w_real, EP=EP, Urem=Urem, TM=TM, Eplan=Eplan,
                skill=skill, stress=stress, net_b=net_b_arr, omega=omega_arr,
                Cmin=Cmin, outcome=outcome, sL_end=sL[-1], C_end=C[-1],
                neg_streak=neg_streak, b_end=b[-1], spiral=spiral,
                crash_type=crash_type, social_crash=social_crash,
                tax_sum=float(tax.sum()), short_sum=float(shortfall.sum()),
                skill_end=float(skill[-1]), omega_end=float(omega_arr[-1]))


def welfare(r, par, lam=0.5, lam_pe=0.6, delta=0.6, nu=0.5):
    Cfe = par.w * par.LS
    gap = np.maximum(0.0, Cfe - r["C"])
    tm_cost = lam * r["TM"]
    pe_cost = lam_pe * par.wP * r["EP"] + nu * r["EP"] ** 2
    dwl = delta * par.tau ** 2 * r["tax"]
    loss = gap + tm_cost + pe_cost + dwl
    return dict(total=float(loss.sum()), gap=float(gap.sum()),
                cost=float((tm_cost + pe_cost).sum()), dwl=float(dwl.sum()),
                minC=float(np.min(r["C"][10:])), tax_sum=float(r["tax"].sum()),
                skill_end=float(r["skill_end"]))


def run_case(par):
    r = simulate_unified(par)
    w = welfare(r, par)
    return dict(outcome=r["outcome"], crash_type=r["crash_type"],
                total=w["total"], gap=w["gap"], minC=w["minC"],
                omega_end=r["omega_end"], skill_end=r["skill_end"],
                tax_sum=r["tax_sum"])
