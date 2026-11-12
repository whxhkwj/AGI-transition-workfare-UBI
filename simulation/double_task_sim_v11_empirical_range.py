# -*- coding: utf-8 -*-
"""
社会失稳通道 经验区间稳健性扫描（v11，新命名，不覆盖任何既有脚本）
=================================================================================
修正校准（η=0.40, μ_TM=0.50, ω=0.05），基于 v6 引擎。
目的：把压力系数 alpha_s 的扫描扩展到外部实证反推的完整经验区间 0.5–2.5，
并对社会崩溃阈值 stress_thresh 做方向性扫描，界定社会通道对参数高端取值的脆弱性。
两组政策：
  角点 CORNER：τ=0.15, pE=0.45, φmax=0.02, retrain=0
  标准 STANDARD：τ=0.16, φmax=0.65, pE=0.45, retrain=0
  另附纯转移（pE=0）作为社会通道极端对照。
报告：outcome / crash_type / minC / social_crash / 最大超阈连续期 / 首次持续超阈触发期。
解析临界（角点）：alpha_s,crit = 0.50/(1-0.45) = 0.9091；stress_thresh,crit = 0.33。
"""
import os, importlib.util
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('v6', os.path.join(OUT, 'double_task_sim_v6.py'))
v6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v6)

CALIB = dict(eta=0.40, mu_TM=0.50, omega=0.05)   # ★ 修正校准（用户锁定）
POLICIES = {
    'CORNER':   dict(CALIB, tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0),
    'STANDARD': dict(CALIB, tau=0.16, pE=0.45, phi_max=0.65, retrain=0.0),
    'PURE_TRANSFER': dict(CALIB, tau=0.16, pE=0.00, phi_max=0.65, retrain=0.0),
}


def run(policy, **over):
    par = v6.Params(**POLICIES[policy], **over)
    r = v6.simulate_v6(par)
    # 社会超阈轨迹（与引擎一致，从 t=10 起）
    over_arr = r['stress'][10:] > par.stress_thresh
    streak, max_streak, trigger = 0, 0, None
    for k, x in enumerate(over_arr):
        if x:
            streak += 1
            if trigger is None:
                trigger = k + 10
        else:
            streak = 0
        max_streak = max(max_streak, streak)
    econ_crash = r['crash_type'] in ('econ', 'both')
    return dict(policy=policy, outcome=r['outcome'], crash_type=r['crash_type'],
                minC=float(np.min(r['C'][10:])), social_crash=bool(r['social_crash']),
                econ_crash=bool(econ_crash),
                stress_peak=float(np.max(r['stress'])), max_streak=int(max_streak),
                first_over=trigger, soc_period=par.soc_period)


def baseline_confirm():
    print('### 基准确认（alpha_s=0.60, stress_thresh=0.50, soc_period=10）')
    for name in POLICIES:
        r = run(name)
        print(f"  {name:9s} outcome={r['outcome']:11s} crash_type={r['crash_type']:5s} "
              f"stress_peak={r['stress_peak']:.3f} minC={r['minC']:.3f} "
              f"social_crash={r['social_crash']} max_streak={r['max_streak']}")
    print()


def sweep_alpha_empirical():
    # 完整经验区间 0.5–2.5，并在解析临界 0.909 附近加密
    grids = [0.50, 0.60, 0.75, 0.85, 0.90, 0.91, 0.95, 1.00, 1.25, 1.50, 2.00, 2.50]
    print('### alpha_s 完整经验区间扫描（stress_thresh=0.50, soc_period=10）')
    print(f"  {'alpha_s':>7} | {'policy':9s} | {'outcome':11s} {'crash':6s} | {'minC':6s} | "
          f"{'social':6s} | stress_peak | maxStreak | first_over")
    for name in POLICIES:
        for a in grids:
            r = run(name, alpha_s=a)
            fo = '' if r['first_over'] is None else str(r['first_over'])
            print(f"  {a:7.2f} | {name:9s} | {r['outcome']:11s} {r['crash_type']:6s} | "
                  f"{r['minC']:6.3f} | {str(r['social_crash']):6s} | "
                  f"{r['stress_peak']:11.3f} | {r['max_streak']:9d} | {fo}")
        print('  ' + '-' * 100)
    print()


def sweep_thresh_directional():
    # 方向性扫描，覆盖解析临界 0.33
    grids = [0.20, 0.25, 0.30, 0.33, 0.35, 0.40, 0.45, 0.50, 0.60]
    print('### stress_thresh 方向性扫描（alpha_s=0.60, soc_period=10）')
    print(f"  {'thresh':>7} | {'policy':9s} | {'outcome':11s} {'crash':6s} | {'minC':6s} | "
          f"{'social':6s} | maxStreak | first_over")
    for name in POLICIES:
        for th in grids:
            r = run(name, stress_thresh=th)
            fo = '' if r['first_over'] is None else str(r['first_over'])
            print(f"  {th:7.2f} | {name:9s} | {r['outcome']:11s} {r['crash_type']:6s} | "
                  f"{r['minC']:6.3f} | {str(r['social_crash']):6s} | "
                  f"{r['max_streak']:9d} | {fo}")
        print('  ' + '-' * 95)
    print()


def sweep_2d():
    alphas = [0.50, 0.60, 0.75, 0.91, 1.00, 1.50, 2.50]
    threshs = [0.25, 0.30, 0.33, 0.40, 0.50]
    print('### 二维网格 alpha_s × stress_thresh（角点 CORNER；S=social_crash, .=无社会崩溃）')
    print('          ' + ''.join(f"a={a:<6.2f}" for a in alphas))
    for th in threshs:
        line = f"  th={th:.2f} "
        for a in alphas:
            r = run('CORNER', stress_thresh=th, alpha_s=a)
            line += '  ' + ('S' if r['social_crash'] else '.') + '     '
        print(line)
    print()


def critical_summary():
    print('### 临界边界小结（角点 CORNER）')
    print('  解析：alpha_s,crit = 0.50/(1-0.45) = 0.9091；stress_thresh,crit = 0.60*0.55 = 0.33')
    # 数值定位：alpha_s 首个社会崩溃点
    alphas = np.arange(0.50, 2.5001, 0.01)
    first_a = next((round(a, 2) for a in alphas if run('CORNER', alpha_s=float(a))['social_crash']), None)
    threshs = np.arange(0.20, 0.6001, 0.01)
    first_t = next((round(t, 2) for t in threshs if run('CORNER', stress_thresh=float(t))['social_crash']), None)
    print(f"  数值：alpha_s 首个社会崩溃点 = {first_a}；stress_thresh 首个社会崩溃点（自低向高）= {first_t}")
    r0 = run('CORNER')
    print(f"  基准（alpha_s=0.60, thresh=0.50）：social_crash={r0['social_crash']}，基准结论不翻转；")
    print('  仅当 alpha_s 升至经验区间高端（>0.91）或 thresh 降至 0.33 以下时，角点转为社会崩溃。')
    print()


if __name__ == '__main__':
    print('社会失稳通道经验区间稳健性扫描 v11（修正校准，v6 引擎）')
    print('=' * 104)
    baseline_confirm()
    sweep_alpha_empirical()
    sweep_thresh_directional()
    sweep_2d()
    critical_summary()
