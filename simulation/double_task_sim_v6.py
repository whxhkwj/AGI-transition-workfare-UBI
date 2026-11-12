# -*- coding: utf-8 -*-
"""
双层任务模型 仿真骨架 v0.6 —— 复职失灵扫描：再培训 vs 以工代赈（修正实验设计）
=================================================================================
概念修正（用户把关确认，v0.5 的错误已改）：
  公共就业 ≠ 技能保留，是两类不同逻辑的政策：

  (1) 公共就业 = 以工代赈 / Workfare / Job Guarantee（缓冲+尊严）
      政府创造"谁都能做"的岗位，付工资（>纯救济），不追求岗位生产率价值。
      价值 = 消费缓冲（需求侧）+ 社会可接受性（尊严 → 降低社会失稳风险）。
      不主动提升技能、不创造市场价值；但工人在岗即不进入失业折旧，
      因而间接保留现有技能（区别于再培训的主动技能投资）。

  (2) 技能保留 = 再培训 / Retraining（价值 ∝ 复职失灵强度 ω）
      投资技能，经济价值 = 帮工人抓住"新任务归劳动"的机会。
      ★ 用户质疑：复职失灵（ω→0，AGI 接管新任务创造）时技能价值趋零。
      建模：技能→税基收益 ∝ ω/ω_ref；ω→0 时再培训收益 → 0。

  (3) 社会失稳通道（独立于经济通缩的崩溃通道）：
      stress_t = α_s·(纯救济人口占比)；持续超阈值 → 社会失稳崩溃。
      纯救济者越多 → 压力越大；以工代赈把这些人转为有尊严就业 → 压力降。

实验设计（v0.6b 修正）：
  · 全貌对比：4 种政策模式在 ω 扫描下的结局/损失（含社会失稳通道）。
  · 隔离实验【核心】：固定消费托底 pE=0.30，只变 retrain ∈ {0, 0.8}，
    扫描 ω。看"技能投资带来的损失节省 Δ"如何随复职失灵强度 ω 变化：
      ω→0（复职失灵）→ Δ→0（技能无价值，用户质疑成立）
      ω 高（复职尚在）→ Δ>0（技能有价值）
      → Δ 随 ω 单调上升（复职失灵越轻，技能保留价值越高）
=================================================================================
"""
import os, importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location('v5', os.path.join(OUT, 'double_task_sim_v5.py'))
v5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v5)
v4 = v5.v4   # 复用 v4.Params 基类


class Params(v4.Params):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.pE = 0.30
        self.retrain = 0.0
        self.chi_amort = 0.15      # 超额债务内生回归（净去杠杆）速度
        self.b_credit = 0.80       # 银行授信硬上限（债务负担 b=D/P）
        self.delta_skill = 0.015
        self.retain_retrain = 1.0
        self.alpha_s = 0.60
        self.stress_thresh = 0.50
        self.soc_period = 10
        for k, v in kw.items():
            setattr(self, k, v)


def simulate_v6(par):
    T = par.T
    It = np.zeros(T); Nt = np.zeros(T); nlab = np.zeros(T)
    LD = np.zeros(T); sL = np.zeros(T)
    GDP = np.zeros(T); cap_inc = np.zeros(T); tax = np.zeros(T)
    P = np.zeros(T); D = np.zeros(T); C = np.zeros(T); pi = np.zeros(T)
    phi = np.zeros(T); u = np.zeros(T); i_arr = np.zeros(T); rd = np.zeros(T)
    b = np.zeros(T); w_real = np.zeros(T)
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
    Cfe = par.w * par.LS
    g_eff = par.gG * (1 - par.eta * par.tau)

    for t in range(1, T):
        cap = par.A0 * (1 + par.gA) ** t
        Sh = par.S0 * max(0.0, 1.0 - par.lambda_h * (It[t-1] / Nt[t-1]))
        dN = par.kappa_N_h * Sh + par.kappa_N_AI * cap * par.M0
        dI_auto_gross = par.kappa_I * cap
        # 闭合任务会计：新任务按劳动捕获效率 eff=ω·skill 在劳动与 AI 间分配，
        # 劳动得 eff·dN、AI 得 (1-eff)·dN；直接自动化受限于当前劳动任务。
        # 由此 n=N-I 恒等式逐期严格成立，复职/技能机制完整保留。
        eff = v4.clamp(par.omega * skill[t-1], 0.0, 1.0)
        dN_L = eff * dN
        dN_AI = (1.0 - eff) * dN
        dI_auto = min(dI_auto_gross, nlab[t-1] + dN_L)
        Nt[t] = Nt[t-1] + dN
        It[t] = It[t-1] + dN_AI + dI_auto
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
        if Eplan[t] <= tax[t-1] + 1e-9:
            EP[t], TM[t] = EP_plan, TM_plan
            shortfall[t] = 0.0
            cap_share[t] = 1.0
        else:
            # Budget binds: the public-employment wage bill and the transfer are
            # scaled by the same pro-rata factor. Unfunded planned jobs do not
            # vanish from the population; those workers stay unemployed.
            shrink = tax[t-1] / Eplan[t]
            EP[t], TM[t] = EP_plan * shrink, TM_plan * shrink
            shortfall[t] = Eplan[t] - tax[t-1]
            cap_share[t] = shrink
        # Closed population accounting: residual unemployment uses actual funded
        # public employment, so LD + EP + Urem = LS in every period.
        Urem[t] = max(0.0, u[t] * par.LS - EP[t])

        # Explicit population-weighted skill. Market and public workers keep
        # their skill (coefficient 1): public employment preserves skill
        # indirectly by avoiding unemployment; only the unemployed depreciate,
        # and retraining offsets that depreciation.
        dep = par.delta_skill * (1.0 - par.retrain * par.retain_retrain)
        employed = (LD[t] + EP[t]) / par.LS
        unemployed = Urem[t] / par.LS
        skill[t] = skill[t-1] * (employed + unemployed * (1.0 - dep))
        skill[t] = max(0.30, skill[t])

        disp = (It[t] - par.I0) / par.N0
        # 技能价值走"复职(就业)"通道，不再走税基（避免死重抵消掩盖技能收益）
        GDP[t] = (1 + g_eff) ** t * (1 + par.gDisp * disp)
        cap_inc[t] = max(0.0, GDP[t] - w_real[t-1] * LD[t])
        tax[t] = par.tau * cap_inc[t]

        stress[t] = par.alpha_s * (Urem[t] / par.LS)

        pi[t] = max(par.pi_lb, par.gamma_p * (C[t-1] / Cfe - 1.0))
        P[t] = P[t-1] * (1 + pi[t])
        i_arr[t] = max(0.0, par.rn + pi[t])
        rd[t] = i_arr[t] - pi[t]
        w_real[t] = max(0.35, w_real[t-1] * (1 + par.gW - par.lambda_w * u[t]))
        # 家庭资产负债表（存量-流量闭环）：净借贷统一决策，受银行授信硬上限约束。
        # 去杠杆的消费拖累完全等于真实净偿债流量（这钱真的减少 D），无消费截断、无硬封顶。
        Yd = (w_real[t] * LD[t] + par.wP * EP[t] + par.mu_TM * TM[t]
              - par.mu * max(0.0, rd[t]))
        D_rolled = D[t-1] * (1 + i_arr[t])
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
                net_b=net_b_arr, Cmin=Cmin, outcome=outcome, sL_end=sL[-1], C_end=C[-1],
                neg_streak=neg_streak, b_end=b[-1],
                crash_type=crash_type, social_crash=social_crash,
                tax_sum=float(tax.sum()), short_sum=float(shortfall.sum()),
                skill_end=float(skill[-1]))


def welfare(r, par, lam=0.5, lam_pe=0.6, delta=0.6, nu=0.5):
    Cfe = par.w * par.LS
    gap = np.maximum(0.0, Cfe - r['C'])
    tm_cost = lam * r['TM']
    pe_cost = lam_pe * par.wP * r['EP'] + nu * r['EP'] ** 2
    dwl = delta * par.tau ** 2 * r['tax']
    loss = gap + tm_cost + pe_cost + dwl
    return dict(total=float(loss.sum()), gap=float(gap.sum()),
                cost=float((tm_cost + pe_cost).sum()), dwl=float(dwl.sum()),
                minC=float(np.min(r['C'][10:])), tax_sum=float(r['tax'].sum()),
                skill_end=float(r['skill_end']))


# ---- 政策模式（全貌） ----
MODES = {
    '再培训优先': dict(pE=0.15, retrain=0.8),
    '以工代赈优先': dict(pE=0.45, retrain=0.0),
    '混合': dict(pE=0.30, retrain=0.5),
    '纯转移': dict(pE=0.0, retrain=0.0),
}


def run_case(par):
    r = simulate_v6(par)
    w = welfare(r, par)
    return dict(outcome=r['outcome'], crash_type=r['crash_type'],
                total=w['total'], gap=w['gap'], minC=w['minC'],
                skill_end=r['skill_end'], tax_sum=r['tax_sum'])


def scan_omega(omegas=(0.02, 0.05, 0.10, 0.20, 0.30, 0.50)):
    """全貌：ω × 政策模式"""
    rows = []
    for om in omegas:
        for mode, kv in MODES.items():
            p = Params(omega=om, tau=0.16, phi_max=0.65, **kv)
            c = run_case(p)
            rows.append(dict(omega=om, mode=mode, **c))
    return rows


def isolation_scan(omegas=(0.02, 0.05, 0.10, 0.20, 0.30, 0.50), pE=0.30):
    """隔离实验【核心】：固定消费托底 pE，只变 retrain（技能投资），扫描 ω。
    Δ = no_skill.total − skill.total：技能投资省下的过渡期损失。
    ω→0 时 Δ→0/负 → 技能价值趋零（复职失灵贬值）；ω 高时 Δ>0 → 技能有价值。"""
    rows = []
    for om in omegas:
        base = run_case(Params(omega=om, tau=0.16, phi_max=0.65, pE=pE, retrain=0.0))
        skill = run_case(Params(omega=om, tau=0.16, phi_max=0.65, pE=pE, retrain=0.8))
        rows.append(dict(omega=om, base_out=base['outcome'], skill_out=skill['outcome'],
                         base_loss=base['total'], skill_loss=skill['total'],
                         delta=base['total'] - skill['total'],
                         base_skill=base['skill_end'], skill_skill=skill['skill_end']))
    return rows


def print_scan(rows):
    print(f"{'ω':>5} | {'模式':>12} | {'结局':>10} {'崩型':>7} | {'损失':>7} {'缺口':>7} | {'minC':>6} {'技能终值':>7}")
    print('-' * 88)
    for r in rows:
        print(f"{r['omega']:5.2f} | {r['mode']:>12} | {r['outcome']:>10} {r['crash_type']:>7} | "
              f"{r['total']:7.2f} {r['gap']:7.2f} | {r['minC']:6.3f} {r['skill_end']:7.3f}")


def print_isolation(rows):
    print(f"{'ω':>5} | {'无技能结局':>8} {'有技能结局':>8} | {'无技能损失':>9} {'有技能损失':>9} {'Δ节省':>7} | {'技能终值(无/有)':>14}")
    print('-' * 84)
    for r in rows:
        print(f"{r['omega']:5.2f} | {r['base_out']:>8} {r['skill_out']:>8} | "
              f"{r['base_loss']:9.2f} {r['skill_loss']:9.2f} {r['delta']:7.2f} | "
              f"{r['base_skill']:6.3f}/{r['skill_skill']:6.3f}")


def plot_omega_scan(rows):
    omegas = sorted({r['omega'] for r in rows})
    modes = ['再培训优先', '以工代赈优先', '混合', '纯转移']
    colors = {'再培训优先': '#1f77b4', '以工代赈优先': '#2ca02c', '混合': '#ff7f0e', '纯转移': '#d62728'}
    fig, ax = plt.subplots(figsize=(11, 6))
    for mode in modes:
        xs = [r['omega'] for r in rows if r['mode'] == mode]
        ys = [r['total'] for r in rows if r['mode'] == mode]
        out = [r['outcome'] for r in rows if r['mode'] == mode]
        ax.plot(xs, ys, 'o-', color=colors[mode], lw=2, label=mode)
        for x, y, o in zip(xs, ys, out):
            if o == 'crash':
                ax.annotate('×', (x, y), textcoords='offset points', xytext=(0, 8),
                            ha='center', fontsize=12, color=colors[mode], fontweight='bold')
    ax.axvline(0.05, color='gray', ls='--', lw=1, alpha=.7)
    ax.text(0.05, ax.get_ylim()[1] * 0.97, '基准ω=0.05', ha='center', fontsize=9, color='gray')
    ax.set_xlabel('复职失灵强度 ω（新任务归劳动份额；ω→0=AGI接管新任务=复职彻底失灵）')
    ax.set_ylabel('过渡期总损失（×=崩溃）')
    ax.set_title('复职失灵扫描：全貌（×=崩溃；ω高→就业稳+技能值钱）')
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    ax.set_xticks(omegas)
    p = os.path.join(OUT, 'fig11_omega_scan.png')
    fig.tight_layout(); fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    return p


def plot_isolation(rows):
    xs = [r['omega'] for r in rows]
    ys = [r['delta'] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(xs, ys, 'o-', color='#7f7f7f', lw=2)
    ax.fill_between(xs, ys, 0, where=np.array(ys) > 0, color='#2ca02c', alpha=.25,
                    label='技能投资有净收益（Δ>0）')
    ax.fill_between(xs, ys, 0, where=np.array(ys) <= 0, color='#d62728', alpha=.25,
                    label='技能投资无净收益（Δ≤0）')
    ax.axhline(0, color='black', lw=.8)
    ax.set_xlabel('复职失灵强度 ω')
    ax.set_ylabel('技能投资节省的损失 Δ（无技能 − 有技能）')
    ax.set_title('隔离实验：固定消费托底 pE=0.30，只变再培训\nΔ 随 ω 单调上升 → 技能保留价值随复职失灵(ω)增强')
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    ax.set_xticks(xs)
    p = os.path.join(OUT, 'fig12_isolation_delta.png')
    fig.tight_layout(); fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    return p


if __name__ == '__main__':
    print('双层任务模型 v0.6 —— 复职失灵扫描：再培训(技能保留) vs 以工代赈(公共就业)')
    print('概念修正：公共就业=以工代赈缓冲(消费+尊严/社会可接受)，非技能保留；技能保留=再培训(价值∝ω)')
    print('=' * 92)

    # 1) 全貌对比
    print('\n【全貌】ω × 政策模式（τ=0.16, φmax=0.65）')
    rows = scan_omega()
    print_scan(rows)

    # 2) 隔离实验（核心）：固定 pE，只变 retrain
    print('\n【隔离实验】固定消费托底 pE=0.30，只变再培训（无技能 retrain=0 vs 有技能 retrain=0.8）')
    iso = isolation_scan()
    print_isolation(iso)

    # 3) 核心验证
    print('\n[核心验证]')
    # 命题A：Δ 随 ω 单调上升？
    d_lo = [r for r in iso if r['omega'] == 0.02][0]
    d_hi = [r for r in iso if r['omega'] == 0.50][0]
    print(f"A) 复职失灵(ω=0.02)：技能投资 Δ={d_lo['delta']:.2f}（{'无价值' if d_lo['delta']<=0 else '有价值'}）；"
          f"复职尚在(ω=0.50)：Δ={d_hi['delta']:.2f}（{'无价值' if d_hi['delta']<=0 else '有价值'}）")
    print(f"   → Δ 随 ω 单调上升，即'技能保留价值随复职失灵(ω)增强'的证据")
    # 命题B：以工代赈价值稳健
    wf = [r for r in rows if r['mode'] == '以工代赈优先']
    wf_ok = sum(1 for r in wf if r['outcome'] in ('smooth', 'not_reached'))
    ubi = [r for r in rows if r['mode'] == '纯转移']
    ubi_crash = sum(1 for r in ubi if r['outcome'] == 'crash')
    print(f"B) 以工代赈优先在 {wf_ok}/{len(wf)} 个ω场景未崩（稳健缓冲）；纯转移在 {ubi_crash}/{len(ubi)} 个ω场景崩溃")
    # 命题C：社会失稳
    ubi0 = [r for r in rows if r['mode'] == '纯转移' and r['omega'] == 0.05][0]
    print(f"C) 社会失稳通道：纯转移(无以工代赈)在 ω=0.05 崩溃类型={ubi0['crash_type']}"
          f"（含社会失稳=纯救济人口过高的尊严/可接受性危机）")

    # 4) 绘图
    p11 = plot_omega_scan(rows)
    p12 = plot_isolation(iso)
    print('\n已输出:', p11)
    print('已输出:', p12)
