# -*- coding: utf-8 -*-
"""
双层任务模型 v12（兼容层）—— 内生复职率（主模型入口）
=================================================================================
本文件不再包含独立物理方程。全仓库唯一物理引擎为 double_task_sim_unified.py；
v12 保留为"内生复职率"入口：simulate_v12 强制 omega_mode='endogenous'，调用
统一引擎的竞争式 logistic 复职法则

    logit(omega_t) = ell0 + Gamma_t - beta_A * g_t

其中 AGI 能力 g_t 上升压低劳动新任务捕获份额，政策存量 Gamma_t 对抗该下降。
beta_A=0 退化为只随政策上升的旧 v12 方向；外生常数情形见 v6 兼容层。
=================================================================================
"""
import os, importlib.util

OUT = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "unified", os.path.join(OUT, "double_task_sim_unified.py"))
unified = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(unified)

v4 = unified.v4
Params = unified.Params


def simulate_v12(par):
    """内生复职率路径：AGI 能力与政策在 log-odds 上竞争，调用统一引擎。"""
    par.omega_mode = "endogenous"
    return unified.simulate_unified(par)


def welfare(r, par, lam=0.5, lam_pe=0.6, delta=0.6, nu=0.5):
    return unified.welfare(r, par, lam, lam_pe, delta, nu)


def run_case(par):
    r = simulate_v12(par)
    w = welfare(r, par)
    return dict(outcome=r["outcome"], crash_type=r["crash_type"],
                total=w["total"], gap=w["gap"], minC=w["minC"],
                omega_end=r["omega_end"], skill_end=r["skill_end"],
                tax_sum=r["tax_sum"])
