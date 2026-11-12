# -*- coding: utf-8 -*-
"""
双层任务模型 v6（兼容层）—— 外生复职率
=================================================================================
本文件不再包含独立物理方程。全仓库唯一物理引擎为 double_task_sim_unified.py；
v6 保留为"外生复职率"入口：simulate_v6 强制 omega_mode='exogenous'，把复职率
当作固定常数 omega，调用统一引擎。历史上引用本模块的分析脚本接口不变。

内生复职率（含 AGI 能力侵蚀）为主模型，见 double_task_sim_unified.py 与
double_task_sim_v12_endogenous_omega.py。
=================================================================================
"""
import os, importlib.util

OUT = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "unified", os.path.join(OUT, "double_task_sim_unified.py"))
unified = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(unified)

# 复用统一引擎的基础组件与参数类
v4 = unified.v4
Params = unified.Params


def simulate_v6(par):
    """外生复职率路径：强制 omega 为固定常数，其余会计与统一引擎一致。"""
    par.omega_mode = "exogenous"
    return unified.simulate_unified(par)


def welfare(r, par, lam=0.5, lam_pe=0.6, delta=0.6, nu=0.5):
    return unified.welfare(r, par, lam, lam_pe, delta, nu)


def run_case(par):
    r = simulate_v6(par)
    w = welfare(r, par)
    return dict(outcome=r["outcome"], crash_type=r["crash_type"],
                total=w["total"], gap=w["gap"], minC=w["minC"],
                omega_end=r["omega_end"], skill_end=r["skill_end"],
                tax_sum=r["tax_sum"])


# 向后兼容的常量名
MODES = ["corner", "standard", "pure transfer"]
