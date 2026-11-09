# -*- coding: utf-8 -*-
"""
社会失稳通道 阈值敏感性扫描（v10，新命名，不覆盖任何既有脚本）
=================================================================================
修正校准（η=0.40, μ_TM=0.50, ω=0.05），基于 v6 引擎。
扫描参数：
  stress_thresh（基准 0.50）、alpha_s（基准 0.60）、soc_period（基准 10）。
两组政策：
  角点 CORNER：τ=0.15, pE=0.45, φmax=0.02, retrain=0
  标准 STANDARD：τ=0.16, φmax=0.65, pE=0.45, retrain=0
  另附纯 UBI（pE=0）作为社会通道极端对照。
报告：outcome / crash_type / minC / social_crash / 最大超阈连续期 / 首次持续超阈触发期。
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
    'PURE_UBI': dict(CALIB, tau=0.16, pE=0.00, phi_max=0.65, retrain=0.0),
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


def baseline_stress():
    print('### 基准参数下 stress 轨迹特征（alpha_s=0.60, stress_thresh=0.50, soc_period=10）')
    for name in POLICIES:
        r = run(name)
        print(f"  {name:9s} outcome={r['outcome']:11s} crash_type={r['crash_type']:5s} "
              f"stress_peak={r['stress_peak']:.3f} minC={r['minC']:.3f} social_crash={r['social_crash']}")
    print()


def sweep_thresh():
    grids = [0.20, 0.25, 0.30, 0.33, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
    print('### 扫描 stress_thresh（其余基准）')
    print(f"  {'thresh':>7} | {'policy':9s} | {'outcome':11s} {'crash':5s} | {'minC':6s} | {'social':6s} | {'maxStreak':9s} | first_over")
    for name in POLICIES:
        for th in grids:
            r = run(name, stress_thresh=th)
            print(f"  {th:7.2f} | {name:9s} | {r['outcome']:11s} {r['crash_type']:5s} | "
                  f"{r['minC']:6.3f} | {str(r['social_crash']):6s} | {r['max_streak']:9d} | {r['first_over']}")
        print('  ' + '-' * 90)
    print()


def sweep_alpha():
    grids = [0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20]
    print('### 扫描 alpha_s（其余基准）')
    print(f"  {'alpha_s':>7} | {'policy':9s} | {'outcome':11s} {'crash':5s} | {'minC':6s} | {'social':6s} | stress_peak | maxStreak")
    for name in POLICIES:
        for a in grids:
            r = run(name, alpha_s=a)
            print(f"  {a:7.2f} | {name:9s} | {r['outcome']:11s} {r['crash_type']:5s} | "
                  f"{r['minC']:6.3f} | {str(r['social_crash']):6s} | {r['stress_peak']:11.3f} | {r['max_streak']}")
        print('  ' + '-' * 90)
    print()


def sweep_period():
    grids = [3, 5, 10, 15, 20, 30]
    print('### 扫描 soc_period（其余基准）')
    print(f"  {'soc_per':>7} | {'policy':9s} | {'outcome':11s} {'crash':5s} | {'social':6s} | maxStreak")
    for name in POLICIES:
        for j in grids:
            r = run(name, soc_period=j)
            print(f"  {j:7d} | {name:9s} | {r['outcome']:11s} {r['crash_type']:5s} | "
                  f"{str(r['social_crash']):6s} | {r['max_streak']}")
        print('  ' + '-' * 80)
    print()


def sweep_2d():
    threshs = [0.25, 0.30, 0.33, 0.35, 0.40, 0.50]
    alphas = [0.45, 0.60, 0.75, 0.90, 1.20]
    print('### 二维网格 stress_thresh × alpha_s（表中 S=social_crash, .=无社会崩溃；角点 CORNER）')
    print('         ' + ''.join(f"alpha={a:<6.2f}" for a in alphas))
    for th in threshs:
        line = f"  th={th:.2f} "
        for a in alphas:
            r = run('CORNER', stress_thresh=th, alpha_s=a)
            line += '   ' + ('S' if r['social_crash'] else '.') + '        '
        print(line)
    print()


if __name__ == '__main__':
    print('社会失稳通道阈值敏感性扫描 v10（修正校准，v6 引擎）')
    print('=' * 96)
    baseline_stress()
    sweep_thresh()
    sweep_alpha()
    sweep_period()
    sweep_2d()
