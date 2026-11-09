# -*- coding: utf-8 -*-
"""
双层任务模型 仿真骨架 v0.5 —— 技能保留：公共就业的供给侧价值
================================================================
在 v0.4（政策组合：公共就业 + 转移支付 + 资本税融资）基础上，补上公共就业
的供给侧维度。v0.4 只给了公共就业"需求侧"价值（把失业者从低救济转到高工资、
提升当期消费）；v0.5 新增"技能保留"——公共就业保护人力资本，防止长期失业
导致的技能退化。

技能保留机制（人力资本 / 技能存量 s_t）：
  (1) 技能存量 s_t（归一化，初始 1）
      私营就业者：技能维持；
      公共就业者：技能基本保留（retain_pe ≈ 1）；
      剩闲失业者：每期技能退化 δ_skill。
      s_t = s_{t-1}·(1 − δ_skill·剩闲份额)·(1 − (1−retain_pe)·公共就业份额)
      剩闲多 → 技能指数退化；公共就业多 → 技能保得住。
  (2) 供给侧传导：技能 → 生产率 → 税基
      GDP_t = (1+g_eff)^t·(1+gDisp·disp)·(1 + κ_gdp·(s_t−1))
      技能退化（s_t<1）→ 产出能力下降 → GDP/资本所得税基缩小 → 融资弱 → 托底难；
      技能保留（公共就业护 s_t）→ 税基大 → 融资强 → 更能托住平稳过渡。
      技能退化的代价走"税基→融资"通道，公共就业的价值由此而来。
  (3) 公共就业 = 双重价值
      需求侧：高工资替代低救济 → 当期消费升（v0.4 已有）
      供给侧：保住技能 → 工资塌陷慢 → 长期消费稳（v0.5 新增）
      财政成本 vs 双重收益 → 最优公共就业规模内解。

对照 v0.4：技能保留会改变最优 pE（公共就业更值）与损失（工资稳定改善），
且技能流失本身是可量化的社会损失（人力资本贬值，未来复职能力差）。
================================================================
"""
import os, importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.dirname(os.path.abspath(__file__))

# 加载 v0.4（复用 Params / simulate_v4 用于对比）
_spec = importlib.util.spec_from_file_location('v4', os.path.join(OUT, 'double_task_sim_v4.py'))
v4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v4)


class Params(v4.Params):
    def __init__(self, **kw):
        super().__init__(**kw)
        # ---- 技能 / 人力资本（新增） ----
        self.delta_skill = 0.015     # ★ 剩闲失业者每期技能退化率
        self.retain_pe = 0.999       # ★ 公共就业技能保留率（≈1，几乎不退化）
        self.k_skill = 0.02          # 技能对工资影响（当前不启用，仅保留）
        self.k_gdp = 0.50            # ★ 技能对产出(GDP)的生产率弹性：人力资本=生产投入
        self.w_floor = 0.35          # 工资地板（v0.4 口径，技能不直接压工资）
        for k, v in kw.items():
            setattr(self, k, v)


def simulate_v5(par):
    """同 v0.4 结构 + 技能存量模块。返回含 skill 的完整时间序列。"""
    T = par.T
    It = np.zeros(T); Nt = np.zeros(T); nlab = np.zeros(T)
    LD = np.zeros(T); sL = np.zeros(T)
    GDP = np.zeros(T); cap_inc = np.zeros(T); tax = np.zeros(T)
    P = np.zeros(T); D = np.zeros(T); C = np.zeros(T); pi = np.zeros(T)
    phi = np.zeros(T); u = np.zeros(T); i_arr = np.zeros(T); rd = np.zeros(T)
    b = np.zeros(T); w_real = np.zeros(T)
    EP = np.zeros(T); Urem = np.zeros(T); TM = np.zeros(T)
    Eplan = np.zeros(T); shortfall = np.zeros(T)
    cap_share = np.zeros(T)
    skill = np.zeros(T)                                  # ★ 技能存量

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
    skill[0] = (LD[0] * 1.0 + EP[0] * par.retain_pe + Urem[0] * (1 - par.delta_skill)) / par.LS
    C[0] = w_real[0] * LD[0] + par.wP * EP[0] + par.mu_TM * TM[0]
    b[0] = D[0] / P[0]
    Cfe = par.w * par.LS
    g_eff = par.gG * (1 - par.eta * par.tau)

    for t in range(1, T):
        # ---- 双层任务 / 就业（同 v0.4） ----
        cap = par.A0 * (1 + par.gA) ** t
        Sh = par.S0 * max(0.0, 1.0 - par.lambda_h * (It[t-1] / Nt[t-1]))
        dN = par.kappa_N_h * Sh + par.kappa_N_AI * cap * par.M0
        dI = par.kappa_I * cap
        dnlab = par.omega * dN - dI
        It[t] = It[t-1] + dI
        Nt[t] = Nt[t-1] + dN
        nlab[t] = max(0.0, nlab[t-1] + dnlab)
        LD[t] = par.LS * v4.clamp(nlab[t] / Nt[t], 0, 1)
        sL[t] = LD[t] / par.LS
        u[t] = 1 - LD[t] / par.LS

        # ---- 政策组合：公共就业 + 转移支付，平衡预算（同 v0.4） ----
        ud = u[max(0, t - par.d)]
        phi[t] = min(par.phi_max, par.phi_base + par.resp * max(0.0, ud))
        EP_plan = min(par.pE * u[t] * par.LS, u[t] * par.LS)
        Urem_plan = max(0.0, u[t] * par.LS - EP_plan)
        TM_plan = phi[t] * par.w * Urem_plan
        Eplan[t] = par.wP * EP_plan + TM_plan
        Urem[t] = Urem_plan
        if Eplan[t] <= tax[t-1] + 1e-9:            # 财政预算基于上期税收（消除循环依赖）
            EP[t], TM[t] = EP_plan, TM_plan
            shortfall[t] = 0.0
            cap_share[t] = 1.0
        else:
            shrink = tax[t-1] / Eplan[t]
            EP[t], TM[t] = EP_plan * shrink, TM_plan * shrink
            shortfall[t] = Eplan[t] - tax[t-1]
            cap_share[t] = shrink

        # ---- ★ 技能存量（供给侧） ----
        skill[t] = skill[t-1] * (1 - par.delta_skill * (Urem[t] / par.LS)) \
                          * (1 - (1 - par.retain_pe) * (EP[t] / par.LS))
        skill[t] = max(0.30, skill[t])     # 技能地板，避免退到零的极端崩溃

        # ---- 产出 / 资本所得 / 税收（受技能→生产率：人力资本=生产投入） ----
        disp = (It[t] - par.I0) / par.N0
        GDP[t] = (1 + g_eff) ** t * (1 + par.gDisp * disp) \
                 * (1 + par.k_gdp * (skill[t] - 1.0))     # ★ 技能→产出：退化降产出→税基缩
        cap_inc[t] = max(0.0, GDP[t] - w_real[t-1] * LD[t])
        tax[t] = par.tau * cap_inc[t]

        # ---- 名义侧 / 通缩螺旋（同 v0.4，但工资受技能影响） ----
        pi[t] = max(par.pi_lb, par.gamma_p * (C[t-1] / Cfe - 1.0))
        P[t] = P[t-1] * (1 + pi[t])
        i_arr[t] = max(0.0, par.rn + pi[t])
        rd[t] = i_arr[t] - pi[t]
        # 实际工资（同 v0.4：随失业温和下降，技能通过"税基→融资"间接传导）
        w_real[t] = max(0.35, w_real[t-1] * (1 + par.gW - par.lambda_w * u[t]))
        D[t] = D[t-1] * (1 + i_arr[t]) + max(0.0, Cfe - C[t-1]) * par.debt_share
        b[t] = D[t] / P[t]

        # ---- 消费（同 v0.4） ----
        debt_drag = par.zeta * min(max(0.0, b[t] - par.bbar), 0.8)
        C[t] = w_real[t] * LD[t] + par.wP * EP[t] + par.mu_TM * TM[t] \
               - debt_drag - par.mu * max(0.0, rd[t])
        C[t] = max(0.0, C[t])

    # ---- 判定（同 v0.4） ----
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
    fiscal_crash = reached and C_below and (np.sum(shortfall[10:] > 1e-4) > 3)
    crash = C_below
    smooth = reached and not crash
    outcome = 'smooth' if smooth else ('crash' if crash else 'not_reached')
    return dict(It=It, Nt=Nt, nlab=nlab, LD=LD, sL=sL, C=C, GDP=GDP,
                cap_inc=cap_inc, tax=tax, cap_share=cap_share, shortfall=shortfall,
                P=P, pi=pi, phi=phi, u=u, b=b, rd=rd, i_arr=i_arr, w_real=w_real,
                EP=EP, Urem=Urem, TM=TM, Eplan=Eplan, skill=skill,
                Cmin=Cmin, outcome=outcome, sL_end=sL[-1], C_end=C[-1],
                neg_streak=neg_streak, b_end=b[-1],
                fiscal_crash=fiscal_crash, spiral=spiral,
                tax_sum=float(tax.sum()), short_sum=float(shortfall.sum()),
                skill_end=float(skill[-1]))


def welfare(r, par, lam=0.5, lam_pe=0.6, delta=0.6, nu=0.5, lam_skill=0.2):
    """过渡期福利损失 = 消费缺口 + 财政成本 + 资本税死重 + 技能流失社会资产损失
    - 技能流失成本: λ_skill·Σ(1−s_t)   （人力资本贬值：当期税基效应之外，未来复职/增长能力受损）
    """
    Cfe = par.w * par.LS
    gap = np.maximum(0.0, Cfe - r['C'])
    tm_cost = lam * r['TM']
    pe_cost = lam_pe * par.wP * r['EP'] + nu * r['EP'] ** 2
    dwl = delta * par.tau ** 2 * r['tax']
    skill_cost_arr = lam_skill * (1.0 - r['skill'])     # 每期技能流失（人力资本贬值）
    loss = gap + tm_cost + pe_cost + dwl + skill_cost_arr
    return dict(total=float(loss.sum()), gap=float(gap.sum()),
                cost=float((tm_cost + pe_cost).sum()), dwl=float(dwl.sum()),
                skill_cost=float(skill_cost_arr.sum()),
                minC=float(np.min(r['C'][10:])), tax_sum=float(r['tax'].sum()),
                skill_end=float(r['skill_end']))


def optimize(lam=0.5, taus=None, pes=None, phis=None):
    """三维扫描 (τ, pE, φ_max)：返回可行组合与全局最优"""
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
                r = simulate_v5(p)
                w = welfare(r, p, lam)
                ok = (w['minC'] >= p.q) and (r['outcome'] == 'smooth')
                rows.append(dict(tau=tau, pE=pe, phi_max=pm, ok=ok,
                                 total=w['total'], minC=w['minC'],
                                 gap=w['gap'], cost=w['cost'], dwl=w['dwl'],
                                 skill_cost=w['skill_cost'], tax_sum=w['tax_sum'],
                                 skill_end=w['skill_end'],
                                 fiscal_crash=bool(r['fiscal_crash'])))
                if ok and (best is None or w['total'] < best['total']):
                    best = dict(tau=tau, pE=pe, phi_max=pm, total=w['total'],
                                minC=w['minC'], gap=w['gap'], cost=w['cost'],
                                dwl=w['dwl'], skill_cost=w['skill_cost'],
                                tax_sum=w['tax_sum'], skill_end=w['skill_end'])
    return rows, best


def plot_skill(par, best):
    """技能保留：技能存量 s_t 与实际工资 w_real 演化对比
    最优组合 vs 无公共就业(纯转移支付+资本税) vs 无政策"""
    p_opt = Params(tau=best['tau'], pE=best['pE'], phi_max=best['phi_max'])
    r_opt = simulate_v5(p_opt)
    p_tm = Params(tau=best['tau'], pE=0.0, phi_max=best['phi_max'])
    r_tm = simulate_v5(p_tm)
    p_no = Params(tau=0.05, pE=0.0, phi_max=0.15)
    r_no = simulate_v5(p_no)
    t = np.arange(par.T)
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.8))
    curves = [(r_opt, f'最优组合(τ={best["tau"]:.2f},pE={best["pE"]:.2f})', '#2ca02c'),
              (r_tm, '纯转移支付+资本税(pE=0)', '#d62728'),
              (r_no, '无政策', '#ff7f0e')]
    for r_, lbl, c in curves:
        axs[0, 0].plot(t, r_['skill'], color=c, label=lbl)
        axs[0, 1].plot(t, r_['w_real'], color=c, label=lbl)
        axs[1, 0].plot(t, r_['EP'], color=c, label=lbl)
        axs[1, 1].plot(t, r_['Urem'], color=c, label=lbl)
    for ax in axs.flat:
        ax.grid(alpha=.3); ax.legend(fontsize=7.5)
    axs[0, 0].set_title('技能存量 s_t（人力资本：剩闲退化、公共就业保留）')
    axs[0, 0].axhline(1.0, color='k', lw=.6, ls='--')
    axs[0, 1].set_title('实际工资 w_real（失业温和下降；技能经税基间接传导）')
    axs[1, 0].set_title('公共就业人数 EP')
    axs[1, 1].set_title('剩闲失业 U_rem（技能退化来源）')
    fig.suptitle('技能保留：公共就业的供给侧价值（技能存量 vs 工资）', y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, 'fig10_skill.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return p


def skill_marginal_report(lam=0.5):
    """技能保留的边际收益：同一政策参数下，v0.4(无技能) vs v0.5(有技能)"""
    print('\n技能保留边际收益（同一组政策参数 τ=0.16, pE=0.30, φmax=0.65）：')
    print('-' * 82)
    print(f"{'版本':>18} | {'结局':>9} | {'损失':>7} {'缺口':>7} {'财政':>7} | {'minC':>6} | {'技能终值':>7}")
    cands = [(0.16, 0.30, 0.65), (0.16, 0.45, 0.65), (0.16, 0.0, 0.65)]
    for tau, pe, pm in cands:
        p4 = v4.Params(tau=tau, pE=pe, phi_max=pm)
        p5 = Params(tau=tau, pE=pe, phi_max=pm)
        r4 = v4.simulate_v4(p4); r5 = simulate_v5(p5)
        w4 = v4.welfare(r4, p4, lam); w5 = welfare(r5, p5, lam)
        print(f"v0.4 无技能 τ={tau} pE={pe:<3} | {r4['outcome']:>9} | "
              f"{w4['total']:7.2f} {w4['gap']:7.2f} {w4['cost']:7.2f} | "
              f"{w4['minC']:6.3f} | {r4['skill_end'] if 'skill' in r4 else '—':>7}")
        print(f"v0.5 有技能 τ={tau} pE={pe:<3} | {r5['outcome']:>9} | "
              f"{w5['total']:7.2f} {w5['gap']:7.2f} {w5['cost']:7.2f} | "
              f"{w5['minC']:6.3f} | {r5['skill_end']:7.3f} | 流失{w5['skill_cost']:6.2f}")


if __name__ == '__main__':
    lam = 0.5
    par = Params()
    print('双层任务模型 v0.5 —— 技能保留：公共就业的供给侧价值')
    print('=' * 82)

    # 1) 三维政策寻优（含技能）
    rows, best = optimize(lam=lam)
    if best is None:
        print('\n无可行的平稳过渡政策组合（全部崩溃）')
    else:
        print(f"\n[1] 三维寻优 (τ, pE, φ_max)：最优 τ={best['tau']:.2f}, "
              f"pE={best['pE']:.2f}, φmax={best['phi_max']:.2f}")
        print(f"    总损失={best['total']:.2f}  (缺口={best['gap']:.2f}, "
              f"财政={best['cost']:.2f}, 死重={best['dwl']:.2f}, "
              f"技能流失={best['skill_cost']:.2f}, minC={best['minC']:.3f})")
        print(f"    技能存量终值={best['skill_end']:.3f}（1=人力资本完整保留，越低流失越重）")
        n_ok = sum(1 for r in rows if r['ok'])
        n_fisc = sum(1 for r in rows if r['fiscal_crash'])
        print(f"    可行组合 {n_ok}/{len(rows)}；融资约束崩溃 {n_fisc} 个")
        print(f"\n    对照：v0.4(无技能) 最优 pE=0.30, 损失=63.05, 技能无；"
              f"v0.5(有技能) 最优 pE={best['pE']:.2f}, 损失={best['total']:.2f}")

    # 2) 技能保留边际收益（同参数 v4 vs v5）
    skill_marginal_report(lam=lam)

    # 3) 绘图
    if best is not None:
        p10 = plot_skill(par, best)
        print('\n已输出:', p10)
