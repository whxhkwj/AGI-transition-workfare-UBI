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
      此外 beta_A=0 给出"只随政策上升"的内生特例（对应 v12 内生方向）。

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
        self.stress_fb = 0.0       # pressure-to-output feedback (0 = channel off)
        # ---- 转型路径约束（最小转型时间 / 早期可行性）----
        self.early_window = 10     # 早期调整窗口长度，与后期统计窗口对齐
        self.early_labor_floor = 0.50  # 早期窗口市场就业份额下限：禁止瞬时清空
        # ---- 论文权威标定（覆盖 v4 遗留默认 .35/.70）----
        self.eta = 0.40             # 资本税增长抑制
        self.mu_TM = 0.50           # 转移支付的总消费 MPC
        # ---- 统一内生复职率字段 ----
        self.omega_mode = "endogenous"
        self.omega0 = 0.05          # 无政策、初始能力基线（= 外生标定值）
        # 政策 -> 再附着 log-odds 系数：logistic 份额对各政策杠杆的结构敏感度。
        self.gamma_PE = 4.5         # 公共就业 -> 再附着 log-odds
        self.gamma_TM = 3.0         # 以工代赈 -> 再附着 log-odds
        self.gamma_R = 2.25         # 再培训 -> 再附着 log-odds
        self.beta_A = 1.0           # AGI 能力 -> 复职份额侵蚀敏感度
        # ---- 动态全覆盖公共就业政策（默认关闭，保持固定 pE 口径）----
        # "fixed"          : EP_plan = pE * u * L^S（现有口径，固定份额）
        # "full_absorption": EP_plan = u * L^S（目标接住全部失业者，有效 pE -> 1），
        #                    财政 within-period 约束照常，税收不足则压缩，刻画崩溃/接管分岔。
        self.pe_policy = "fixed"
        # 全覆盖模式下，财政压缩后是否对剩余失业者保留转移兜底（默认纯公共就业接管）。
        self.full_keep_transfer = False
        # 同预算混合分配：强制把当期可用税收的 transfer_budget_share 预留给失业转移，
        # 其余支付公共就业。0 = 纯公共就业优先（默认），(0,1] 产生正转移支出，用于同预算对照。
        self.transfer_budget_share = 0.0
        # 链接函数截距覆盖：默认 None 用 logit(omega0)。probit 重标定时传入标准
        # 正态分位数 z_0.05，使基准概率仍为 omega0=0.05，保证换链接函数的公平比较。
        self.link_intercept_override = None
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
    unfin_gap = np.zeros(T)   # 零截断处未融资缺口：违约核销或外部资金闭合项
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
    if getattr(par, "link_intercept_override", None) is not None:
        ell0 = float(par.link_intercept_override)
    else:
        ell0 = np.log(par.omega0 / (1.0 - par.omega0))
    # Under a probit re-calibration the intercept is chosen so the baseline CDF
    # still equals omega0; the initial rate is therefore omega0 in both links.
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
        # 同预算混合分配：把当期税收按固定比例拆给公共就业与转移，保证转移有正支出。
        mixed = (par.pe_policy == "full_absorption"
                 and par.transfer_budget_share > 0.0)
        if mixed:
            budget = max(0.0, tax[t - 1])
            b_tm = par.transfer_budget_share * budget
            b_ep = budget - b_tm
            unemployed = u[t] * par.LS
            EP[t] = min(unemployed, b_ep / par.wP)
            Urem[t] = max(0.0, unemployed - EP[t])
            tm_need = phi[t] * par.w * Urem[t]
            TM[t] = min(tm_need, b_tm)
            EP_plan = unemployed
            Eplan[t] = par.wP * unemployed + tm_need
            shortfall[t] = max(0.0, Eplan[t] - budget)
            cap_share[t] = (budget / Eplan[t] if Eplan[t] > 0 else 1.0)
        else:
            if par.pe_policy == "full_absorption":
                # 政策目标：公共就业接住全部失业者（有效 pE -> 1），不留计划内救济。
                EP_plan = u[t] * par.LS
                Urem_plan = 0.0
                TM_plan = phi[t] * par.w * Urem_plan
            else:
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
            # 全覆盖模式可选转移兜底：公共就业优先，剩余财政空间为未接住者融资转移。
            if (par.pe_policy == "full_absorption" and par.full_keep_transfer
                    and Urem[t] > 1e-9):
                fiscal_room = max(0.0, tax[t - 1] - par.wP * EP[t])
                TM_need = phi[t] * par.w * Urem[t]
                TM[t] = min(TM_need, fiscal_room)
                Eplan[t] = par.wP * EP_plan + TM_need
                shortfall[t] = max(0.0, Eplan[t] - tax[t - 1])

        # 显式人口加权技能：在岗者技能保留，仅失业者折旧，再培训抵消。
        dep = par.delta_skill * (1.0 - par.retrain * par.retain_retrain)
        employed = (LD[t] + EP[t]) / par.LS
        unemployed = Urem[t] / par.LS
        skill[t] = skill[t - 1] * (employed + unemployed * (1.0 - dep))
        skill[t] = max(0.30, skill[t])

        disp = (It[t] - par.I0) / par.N0
        stress[t] = par.alpha_s * (Urem[t] / par.LS)
        # Optional pressure feedback: social disruption contracts output and hence
        # the capital income and tax base. Inactive at the default stress_fb=0.
        fb = 1.0 - par.stress_fb * stress[t]
        GDP[t] = (1 + g_eff) ** t * (1 + par.gDisp * disp) * fb
        cap_inc[t] = max(0.0, GDP[t] - w_real[t - 1] * LD[t])
        tax[t] = par.tau * cap_inc[t]

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
        raw_res = Yd + net_b / P[t]
        C[t] = max(0.0, raw_res)
        # 零截断闭合：可用资源为负时，未覆盖部分必须由违约核销或外部资金承担，
        # 显式记账，使家庭资源恒等式 Yd + net_b/P + unfin_gap = C 逐期闭合。
        unfin_gap[t] = max(0.0, -raw_res)

    Cmin = par.q * par.w * par.LS
    ew = par.early_window
    if par.pe_policy == "full_absorption":
        # 成功判据：无遗留失业（市场 + 公共部门接住全部劳动力），而非市场就业消失。
        reached = Urem[-1] < par.term_thresh
    else:
        reached = sL[-1] < par.term_thresh
    # 后期消费可行性（t>=early_window）。
    C_below = np.min(C[ew:]) < Cmin
    # 早期窗口消费可行性：排除 t=0 初始条件，检查 t=1..ew-1，防止早期瞬时崩溃被窗口忽略。
    C_early_below = np.min(C[1:ew]) < Cmin
    neg = pi[ew:] < -0.001
    neg_streak = 0; streak = 0
    for x in neg:
        streak = streak + 1 if x else 0
        neg_streak = max(neg_streak, streak)
    debt_up = b[-1] > b[np.argmin(b)] + 1e-6
    spiral = reached and (neg_streak >= par.spiral_period) and debt_up and C_below
    over = stress[ew:] > par.stress_thresh
    soc_streak = 0; sstreak = 0
    for x in over:
        sstreak = sstreak + 1 if x else 0
        soc_streak = max(soc_streak, sstreak)
    social_crash = soc_streak >= par.soc_period
    # 转型路径约束：早期窗口内市场就业不得瞬时清空（最小转型时间）。
    early_collapse = np.min(sL[1:ew]) < par.early_labor_floor
    econ_crash = C_below or C_early_below
    path_crash = early_collapse
    crash = econ_crash or social_crash or path_crash
    smooth = reached and not crash
    outcome = "smooth" if smooth else ("crash" if crash else "not_reached")
    if crash:
        flags = []
        if social_crash: flags.append("social")
        if econ_crash: flags.append("econ")
        if path_crash: flags.append("path")
        if flags == ["social", "econ"]:
            crash_type = "both"
        elif len(flags) == 1:
            crash_type = flags[0]
        else:
            crash_type = "+".join(flags)
    else:
        crash_type = "none"
    return dict(It=It, Nt=Nt, nlab=nlab, LD=LD, sL=sL, C=C, GDP=GDP,
                cap_inc=cap_inc, tax=tax, cap_share=cap_share, shortfall=shortfall,
                P=P, D=D, pi=pi, phi=phi, u=u, b=b, rd=rd, i_arr=i_arr,
                w_real=w_real, EP=EP, Urem=Urem, TM=TM, Eplan=Eplan,
                skill=skill, stress=stress, net_b=net_b_arr, unfin_gap=unfin_gap,
                omega=omega_arr,
                Cmin=Cmin, outcome=outcome, sL_end=sL[-1], C_end=C[-1],
                neg_streak=neg_streak, b_end=b[-1], spiral=spiral,
                crash_type=crash_type, social_crash=social_crash,
                C_early_below=bool(C_early_below), early_collapse=bool(early_collapse),
                path_crash=bool(path_crash),
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
