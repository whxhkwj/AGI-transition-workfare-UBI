# -*- coding: utf-8 -*-
"""
双层任务模型 仿真骨架 v0.4 —— 政策组合：公共就业 + 转移支付 + 资本税融资
================================================================
在 v0.2（通缩螺旋 + 政策时滞）+ v0.3（福利损失 + 政策寻优）基础上，补上
现有模型最大的缺陷：转移支付是"免费"的——从不问钱从哪来。

v0.4 引入财政闭环（对应研究框架 2.3 转移支付模块 / 2.4 政策开关）：
  (1) 资本所得流
      资本所得_t = GDP_t − w_real_t·LD_t        （工资账单外的全部产出归资本）
      分配塌陷 s_L→0 时资本所得膨胀 → 税基自动扩大（Korinek–Stiglitz 零和恒等式）
  (2) 资本税融资
      税收_t = τ · 资本所得_t                     （τ=资本税率，政策杠杆）
  (3) 平衡预算 / 财政可行性约束（框架：T ≥ w^u·L̄）
      计划支出 E_plan_t = w_P·公共就业_t + φ_t·w·剩闲失业_t
      实际支出 = min(E_plan_t, 税收_t)；不足→按比例压缩→消费跌破需求→
      ★ 融资约束崩溃（新增崩溃通道：不是政策不管，而是税不够）
  (4) 公共就业
      政府按失业比例 pE 雇佣失业者，工资 w_P（> 救济 φ·w，提供体面工作）；
      剩闲者 U_rem 领转移支付。公共就业把税收转化为消费的更"有效"通道。
  (5) 拉弗式权衡（资本税的供给面代价）
      有效增速 g_eff = g_G·(1 − η·τ)：τ 高→投资放缓→终局 GDP 低→税基收窄。
      → τ 太低税不够（托底失败）、τ 太高增长被扼（税基也缩）：存在内解。

核心故事（平稳过渡的融资缺口）：
  自动化替代发生在 GDP 涨够之前 → 劳动份额先塌、税基后到 → 过渡期融资缺口。
  政策组合的目标：用适中 τ（税够但不扼增长）+ 适量公共就业（提消费、省纯救济）
  平滑这个缺口，从崩溃区推进平稳过渡区。
================================================================
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.dirname(os.path.abspath(__file__))


class Params:
    def __init__(self, **kw):
        # ---- 双层任务（同 v0.2） ----
        self.I0, self.N0, self.LS, self.w = 0.20, 2.00, 1.00, 1.00
        self.kappa_I, self.A0, self.gA = 0.020, 1.00, 0.010
        self.kappa_N_h, self.S0, self.lambda_h = 0.005, 1.00, 1.20
        self.kappa_N_AI, self.omega, self.M0 = 0.050, 0.05, 1.00
        # ---- 名义与价格（同 v0.2，通缩螺旋） ----
        self.gamma_p = 0.15
        self.pi_lb = -0.02
        self.rn = 0.02
        self.gW = 0.01
        self.lambda_w = 0.12
        self.flex = 0.01
        self.debt_share = 0.10
        self.zeta = 0.08
        self.mu = 0.06
        self.bbar = 0.35
        self.D0 = 0.30
        # ---- 总量 / 产出（税率影响增长 → 拉弗） ----
        self.gG = 0.03            # 技术生产率年增速
        self.gDisp = 0.05         # 自动化对总产出额外拉动
        self.eta = 0.35           # ★ 资本税对增长抑制系数：g_eff = gG·(1−η·τ)
        # ---- 政策组合（核心新增） ----
        self.tau = 0.30           # ★ 资本税率（融资杠杆）
        self.phi_base = 0.15      # 基础转移支付替代率
        self.resp = 0.50          # 失业→转移支付响应
        self.phi_max = 0.65       # 转移支付上限
        self.d = 0                # 政策时滞（期）
        self.pE = 0.30            # ★ 公共就业：吸纳失业人口比例上限
        self.wP = 0.85            # ★ 公共就业工资（>φ·w 提供体面工作；<市场 w 略低）
        self.mu_TM = 0.70         # ★ 转移支付消费转化率（救济有漏损：储蓄/非当期，<1）
        # ---- 判定 ----
        self.T = 80
        self.term_thresh = 0.05
        self.q = 0.30
        self.spiral_period = 8
        for k, v in kw.items():
            setattr(self, k, v)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def simulate_v4(par):
    T = par.T
    It = np.zeros(T); Nt = np.zeros(T); nlab = np.zeros(T)
    LD = np.zeros(T); sL = np.zeros(T)
    GDP = np.zeros(T); cap_inc = np.zeros(T); tax = np.zeros(T)
    P = np.zeros(T); D = np.zeros(T); C = np.zeros(T); pi = np.zeros(T)
    phi = np.zeros(T); u = np.zeros(T); i_arr = np.zeros(T); rd = np.zeros(T)
    b = np.zeros(T); w_real = np.zeros(T)
    EP = np.zeros(T); Urem = np.zeros(T); TM = np.zeros(T)
    Eplan = np.zeros(T); shortfall = np.zeros(T)   # 计划支出 / 税收缺口
    cap_share = np.zeros(T)                        # 实际支出占计划的比例(平衡预算压缩)

    It[0], Nt[0], nlab[0] = par.I0, par.N0, par.N0 - par.I0
    LD[0] = par.LS * clamp(nlab[0] / Nt[0], 0, 1)
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
    C[0] = w_real[0] * LD[0] + par.wP * EP[0] + par.mu_TM * TM[0]
    b[0] = D[0] / P[0]
    Cfe = par.w * par.LS
    g_eff = par.gG * (1 - par.eta * par.tau)      # 高税→增长放缓（拉弗）

    for t in range(1, T):
        # ---- 双层任务 / 就业（同 v0.2） ----
        cap = par.A0 * (1 + par.gA) ** t
        Sh = par.S0 * max(0.0, 1.0 - par.lambda_h * (It[t-1] / Nt[t-1]))
        dN = par.kappa_N_h * Sh + par.kappa_N_AI * cap * par.M0
        dI = par.kappa_I * cap
        dnlab = par.omega * dN - dI
        It[t] = It[t-1] + dI
        Nt[t] = Nt[t-1] + dN
        nlab[t] = max(0.0, nlab[t-1] + dnlab)
        LD[t] = par.LS * clamp(nlab[t] / Nt[t], 0, 1)
        sL[t] = LD[t] / par.LS
        u[t] = 1 - LD[t] / par.LS

        # ---- 产出（增长率受资本税抑制）与资本所得、税收 ----
        disp = (It[t] - par.I0) / par.N0
        GDP[t] = (1 + g_eff) ** t * (1 + par.gDisp * disp)
        cap_inc[t] = max(0.0, GDP[t] - w_real[t-1] * LD[t])   # 工资账单外的产出归资本
        tax[t] = par.tau * cap_inc[t]

        # ---- 政策组合：公共就业 + 转移支付，平衡预算 ----
        ud = u[max(0, t - par.d)]
        phi[t] = min(par.phi_max, par.phi_base + par.resp * max(0.0, ud))
        EP_plan = min(par.pE * u[t] * par.LS, u[t] * par.LS)   # 雇佣 pE 比例失业者
        Urem_plan = max(0.0, u[t] * par.LS - EP_plan)          # 剩闲者领救济
        TM_plan = phi[t] * par.w * Urem_plan
        Eplan[t] = par.wP * EP_plan + TM_plan
        Urem[t] = Urem_plan                              # 剩闲失业人数（不受金额压缩影响）
        if Eplan[t] <= tax[t] + 1e-9:
            EP[t], TM[t] = EP_plan, TM_plan
            shortfall[t] = 0.0
            cap_share[t] = 1.0
        else:
            shrink = tax[t] / Eplan[t]                          # 税收不足→按比例压缩
            EP[t], TM[t] = EP_plan * shrink, TM_plan * shrink
            shortfall[t] = Eplan[t] - tax[t]
            cap_share[t] = shrink

        # ---- 名义侧 / 通缩螺旋（同 v0.2） ----
        pi[t] = max(par.pi_lb, par.gamma_p * (C[t-1] / Cfe - 1.0))
        P[t] = P[t-1] * (1 + pi[t])
        i_arr[t] = max(0.0, par.rn + pi[t])
        rd[t] = i_arr[t] - pi[t]
        w_real[t] = max(0.35, w_real[t-1] * (1 + par.gW - par.lambda_w * u[t]))
        D[t] = D[t-1] * (1 + i_arr[t]) + max(0.0, Cfe - C[t-1]) * par.debt_share
        b[t] = D[t] / P[t]

        # ---- 消费：私营工资 + 公共就业工资 + 转移支付 − 债务/利率拖累 ----
        debt_drag = par.zeta * min(max(0.0, b[t] - par.bbar), 0.8)
        C[t] = w_real[t] * LD[t] + par.wP * EP[t] + par.mu_TM * TM[t] \
               - debt_drag - par.mu * max(0.0, rd[t])
        C[t] = max(0.0, C[t])

    # ---- 判定 ----
    Cmin = par.q * par.w * par.LS
    reached = sL[-1] < par.term_thresh
    C_below = np.min(C[10:]) < Cmin                       # 消费曾跌破基本需求
    neg = pi[10:] < -0.001
    neg_streak = 0; streak = 0
    for x in neg:
        streak = streak + 1 if x else 0
        neg_streak = max(neg_streak, streak)
    debt_up = b[-1] > b[np.argmin(b)] + 1e-6
    spiral = reached and (neg_streak >= par.spiral_period) and debt_up and C_below
    # 融资约束崩溃：税收长期不足（存在实质压缩期）且消费跌破需求
    fiscal_crash = reached and C_below and (np.sum(shortfall[10:] > 1e-4) > 3)
    crash = C_below                                  # 任何通道，消费跌破基本需求=崩
    smooth = reached and not crash
    outcome = 'smooth' if smooth else ('crash' if crash else 'not_reached')
    return dict(It=It, Nt=Nt, nlab=nlab, LD=LD, sL=sL, C=C, GDP=GDP,
                cap_inc=cap_inc, tax=tax, cap_share=cap_share, shortfall=shortfall,
                P=P, pi=pi, phi=phi, u=u, b=b, rd=rd, i_arr=i_arr, w_real=w_real,
                EP=EP, Urem=Urem, TM=TM, Eplan=Eplan,
                Cmin=Cmin, outcome=outcome, sL_end=sL[-1], C_end=C[-1],
                neg_streak=neg_streak, b_end=b[-1],
                fiscal_crash=fiscal_crash, spiral=spiral,
                tax_sum=float(tax.sum()), short_sum=float(shortfall.sum()))


def welfare(r, par, lam=0.5, lam_pe=0.6, delta=0.6, nu=0.5):
    """过渡期福利损失 = 消费缺口 + 财政成本(区分救济/公共就业) + 资本税死重
    - 转移支付成本:  λ·TM            （纯救济，便宜但激活弱、有消费漏损 μ_TM）
    - 公共就业成本:  λ_PE·wP·EP + ν·EP² （略贵 + 规模边际成本递增/挤出：雇越多效率越低）
    - 资本税死重:    δ·τ²·tax       （拉弗来源：税越高扭曲越大）
    """
    Cfe = par.w * par.LS
    gap = np.maximum(0.0, Cfe - r['C'])
    tm_cost = lam * r['TM']
    pe_cost = lam_pe * par.wP * r['EP'] + nu * r['EP'] ** 2
    dwl = delta * par.tau ** 2 * r['tax']
    loss = gap + tm_cost + pe_cost + dwl
    return dict(total=float(loss.sum()), gap=float(gap.sum()),
                cost=float((tm_cost + pe_cost).sum()), dwl=float(dwl.sum()),
                minC=float(np.min(r['C'][10:])), tax_sum=float(r['tax'].sum()))


def optimize(lam=0.5, taus=None, pes=None, phis=None):
    """三维扫描 (τ, pE, φ_max)：返回可行(不崩)组合与全局最优"""
    if taus is None:
        taus = [round(x, 2) for x in np.linspace(0.05, 0.70, 7)]
    if pes is None:
        pes = [round(x, 2) for x in np.linspace(0.0, 0.60, 5)]
    if phis is None:
        phis = [0.40, 0.55, 0.65, 0.80]
    best = None
    rows = []
    for tau in taus:
        for pe in pes:
            for pm in phis:
                p = Params(tau=tau, pE=pe, phi_max=pm)
                r = simulate_v4(p)
                w = welfare(r, p, lam)
                ok = (w['minC'] >= p.q) and (r['outcome'] == 'smooth')
                rows.append(dict(tau=tau, pE=pe, phi_max=pm, ok=ok,
                                 total=w['total'], minC=w['minC'],
                                 gap=w['gap'], cost=w['cost'], dwl=w['dwl'],
                                 tax_sum=w['tax_sum'],
                                 fiscal_crash=bool(r['fiscal_crash'])))
                if ok and (best is None or w['total'] < best['total']):
                    best = dict(tau=tau, pE=pe, phi_max=pm, total=w['total'],
                                minC=w['minC'], gap=w['gap'], cost=w['cost'],
                                dwl=w['dwl'], tax_sum=w['tax_sum'])
    return rows, best


def plot_policy_combo(par, best):
    """最优组合 vs 纯转移支付(无融资约束=v0.3最优) vs 无政策 —— 时序对比"""
    p_opt = Params(tau=best['tau'], pE=best['pE'], phi_max=best['phi_max'])
    r_opt = simulate_v4(p_opt)
    # 纯转移支付、无资本税（τ=0，想托底但没钱→融资约束崩）
    p_free = Params(tau=0.0, pE=0.0, phi_max=0.65)
    r_free = simulate_v4(p_free)
    # 无政策：τ 极低 + φ 低且不响应
    p_no = Params(tau=0.05, pE=0.0, phi_max=0.15)
    r_no = simulate_v4(p_no)
    Cfe = par.w * par.LS
    t = np.arange(par.T)
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.8))
    curves = [(r_opt, f'最优组合(τ={best["tau"]:.2f},pE={best["pE"]:.2f},φmax={best["phi_max"]:.2f})', '#2ca02c'),
              (r_free, '纯转移支付(τ=0，无钱托底)', '#d62728'),
              (r_no, '无政策(τ=0.05,φmax=0.15)', '#ff7f0e')]
    for r_, lbl, c in curves:
        axs[0, 0].plot(t, r_['sL'], color=c, label=lbl)
        axs[0, 1].plot(t, r_['C'], color=c, label=lbl)
        axs[0, 2].plot(t, r_['tax'], color=c, label=lbl)
        axs[1, 0].plot(t, r_['EP'], color=c, label=lbl)
        axs[1, 1].plot(t, r_['pi'] * 100, color=c, label=lbl)
        axs[1, 2].plot(t, r_['TM'] + par.wP * r_['EP'], color=c, label=lbl)
    for ax in axs.flat:
        ax.grid(alpha=.3); ax.legend(fontsize=7.5)
    axs[0, 0].set_title('劳动份额 s_L')
    axs[0, 1].set_title('总消费 C（红虚线=基本需求）'); axs[0, 1].axhline(par.q, color='red', ls='--', lw=1)
    axs[0, 2].set_title('资本税收入')
    axs[1, 0].set_title('公共就业人数')
    axs[1, 1].set_title('通胀率 π（%）'); axs[1, 1].axhline(0, color='k', lw=.6)
    axs[1, 2].set_title('财政总支出（转移支付+公共就业工资）')
    fig.suptitle('政策组合寻优：最优(公共就业+转移支付+资本税) vs 纯转移支付 vs 无政策', y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, 'fig7_policy_combo.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return p


def plot_fiscal_account(par, best):
    """最优组合的财政账户：税收 vs 支出 vs 公共就业/转移支付拆分 + 压缩比例"""
    p_opt = Params(tau=best['tau'], pE=best['pE'], phi_max=best['phi_max'])
    r = simulate_v4(p_opt)
    t = np.arange(par.T)
    pe_spend = par.wP * r['EP']
    tm_spend = r['TM']
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.6))
    axs[0, 0].plot(t, r['tax'], color='#1f77b4', lw=1.8, label='资本税收入')
    axs[0, 0].plot(t, pe_spend + tm_spend, color='#2ca02c', lw=1.8, label='财政总支出')
    axs[0, 0].plot(t, r['cap_share'], color='#d62728', lw=1.2, ls='--', label='支出实现比例(≤1)')
    axs[0, 0].set_title('融资平衡：税收 vs 支出 vs 实现比例')
    axs[0, 0].grid(alpha=.3); axs[0, 0].legend(fontsize=8)
    axs[0, 1].plot(t, r['cap_inc'], color='#9467bd', lw=1.6, label='资本所得(税基)')
    axs[0, 1].plot(t, r['GDP'], color='#ff7f0e', lw=1.6, label='GDP')
    axs[0, 1].set_title('税基：资本所得随分配塌陷膨胀（与GDP对比）')
    axs[0, 1].grid(alpha=.3); axs[0, 1].legend(fontsize=8)
    axs[1, 0].plot(t, pe_spend, color='#2ca02c', lw=1.6, label='公共就业工资支出')
    axs[1, 0].plot(t, tm_spend, color='#1f77b4', lw=1.6, label='转移支付支出')
    axs[1, 0].set_title('财政支出结构：公共就业 vs 转移支付')
    axs[1, 0].grid(alpha=.3); axs[1, 0].legend(fontsize=8)
    axs[1, 1].plot(t, r['shortfall'], color='#d62728', lw=1.6, label='税收缺口(计划-实际)')
    axs[1, 1].axhline(0, color='k', lw=.6)
    axs[1, 1].set_title('过渡期融资缺口（>0=税收不足期）')
    axs[1, 1].grid(alpha=.3); axs[1, 1].legend(fontsize=8)
    fig.suptitle(f'财政账户：最优组合 τ={best["tau"]:.2f}, pE={best["pE"]:.2f}, φmax={best["phi_max"]:.2f}', y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, 'fig8_fiscal_account.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return p


def plot_tax_phase(rows, best, lam):
    """(τ, pE) 相图（在最优 φ_max 下）：损失面 + 崩溃/可行区 + 最优点"""
    best_pm = best['phi_max']
    sub = [r for r in rows if r['phi_max'] == best_pm]
    taus = sorted({r['tau'] for r in sub})
    pes = sorted({r['pE'] for r in sub})
    Z = np.full((len(pes), len(taus)), np.nan)
    for r in sub:
        i, j = pes.index(r['pE']), taus.index(r['tau'])
        Z[i, j] = r['total'] if r['ok'] else np.nan
    fig, ax = plt.subplots(figsize=(9.5, 6))
    im = ax.imshow(Z, origin='lower', aspect='auto', cmap='viridis',
                   extent=[taus[0], taus[-1], pes[0], pes[-1]])
    ax.set_xlabel('资本税率 τ')
    ax.set_ylabel('公共就业吸纳比例 pE')
    ax.set_title(f'可行区内过渡期总损失（φmax={best_pm:.2f}；NaN=崩溃/融资不足）')
    cbar = fig.colorbar(im, fraction=0.046, label='总损失（消费缺口+λ·财政支出）')
    # 最优点标记
    bt = best['tau']; bp = best['pE']
    ax.plot(bt, bp, 'r*', ms=15, markeredgecolor='white', label=f'最优(τ={bt:.2f},pE={bp:.2f})')
    ax.legend(fontsize=9, loc='upper right')
    p = os.path.join(OUT, 'fig9_tax_phase.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def laffer_report(lam=0.5, pes=(0.0, 0.30)):
    """拉弗效应诊断：固定 φ_max=0.65、pE∈{0,0.3}，扫描 τ 看税收与福利"""
    taus = [round(x, 2) for x in np.linspace(0.05, 0.80, 12)]
    print('\n拉弗效应诊断（φmax=0.65）：τ → 税收总额 / 总损失 / 结局')
    print('-' * 78)
    print(f"{'τ':>5} | {'pE=0: 税':>10} {'损失':>8} {'结局':>9} | "
          f"{'pE=0.3: 税':>10} {'损失':>8} {'结局':>9}")
    for tau in taus:
        line = f"{tau:5.2f} |"
        for pe in pes:
            p = Params(tau=tau, pE=pe, phi_max=0.65)
            r = simulate_v4(p)
            w = welfare(r, p, lam)
            line += f" {w['tax_sum']:10.1f} {w['total']:8.1f} {r['outcome']:>9} |"
        print(line)


if __name__ == '__main__':
    lam = 0.5
    par = Params()
    print('双层任务模型 v0.4 —— 政策组合：公共就业 + 转移支付 + 资本税融资')
    print('=' * 78)

    # 0) 无资本税演示：想托底但没钱 → 融资约束崩溃
    p0 = Params(tau=0.0, pE=0.0, phi_max=0.65)
    r0 = simulate_v4(p0)
    print(f"\n[0] 纯转移支付、无资本税(τ=0)：outcome={r0['outcome']}  "
          f"minC={np.min(r0['C'][10:]):.3f}(需求{par.q*1.0:.2f})  "
          f"税收缺口期数={int(np.sum(r0['shortfall'][10:] > 1e-4))}  → 融资约束崩溃={r0['fiscal_crash']}")

    # 1) 三维政策寻优
    rows, best = optimize(lam=lam)
    if best is None:
        print('\n无可行的平稳过渡政策组合（全部崩溃）')
    else:
        print(f"\n[1] 三维寻优 (τ, pE, φ_max)："
              f"最优 τ={best['tau']:.2f}, pE={best['pE']:.2f}, φmax={best['phi_max']:.2f}")
        print(f"    总损失={best['total']:.2f}  (消费缺口={best['gap']:.2f}, "
              f"财政支出={best['cost']:.2f}, 资本税死重={best['dwl']:.2f}, minC={best['minC']:.3f})")
        n_ok = sum(1 for r in rows if r['ok'])
        n_fisc = sum(1 for r in rows if r['fiscal_crash'])
        print(f"    可行组合 {n_ok}/{len(rows)}；其中融资约束崩溃 {n_fisc} 个（税不够托不住）")
        # 与 v0.3 无融资约束基准对比
        p3 = Params(tau=0.0, pE=0.0, phi_max=0.65)   # v0.3 最优近似（但这里τ=0会崩，见[0]）
        print(f"\n    对照——v0.3 无融资约束最优损失=54.14（但未设融资纪律）；"
              f"v0.4 有融资约束最优损失={best['total']:.2f}")

    # 2) 拉弗效应诊断
    laffer_report(lam=lam)

    # 3) 绘图
    if best is not None:
        p7 = plot_policy_combo(par, best)
        p8 = plot_fiscal_account(par, best)
        p9 = plot_tax_phase(rows, best, lam)
        print('\n已输出:', p7)
        print('已输出:', p8)
        print('已输出:', p9)
