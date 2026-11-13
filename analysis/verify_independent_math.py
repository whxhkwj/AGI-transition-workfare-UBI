# -*- coding: utf-8 -*-
"""独立符号验证：从头推导 BEB 临界条件与动态阈值（不引用既有结论）。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "simulation"))
import sympy as sp
import numpy as np

s, mu, om, sig, kh, S0, lh, q, kI = sp.symbols(
    's mu omega sigma kh S0 lambda_h q kappa_I', positive=True)

# 快系统：新任务总量 dN，劳动净捕获 f
dN = kh * S0 * (1 - lh * (1 - s)) + q * mu
f = sp.expand(om * sig * dN - kI * mu - s * dN)

# s_floor = 1 - 1/lambda_h
sf = sp.simplify(1 - 1 / lh)
print("s_floor =", sf)

# 在 s_floor 处，人类任务项的括号
bracket = sp.simplify(1 - lh * (1 - sf))
print("1 - lambda_h(1-s_floor) =", bracket, " （应为 0）")

# dN 在 s_floor
dN_sf = sp.simplify(dN.subs(s, sf))
print("dN(s_floor) =", dN_sf, " （应只剩 q*mu）")

# f(s_floor) 精确形式
f_sf = sp.factor(sp.simplify(f.subs(s, sf)))
print("f(s_floor) =", f_sf)

# 解临界（对 omega）
crit = sp.solve(sp.Eq(f_sf, 0), om)
print("临界 omega（sigma 显式）=", crit)

# 验证符号临界化简为 (kI/q + s_floor)/sigma
target = sp.simplify((kI / q + sf) / sig)
print("\n符号临界 == (kI/q+s_floor)/sigma :",
      sp.simplify(crit[0] - target) == 0)

# 代入数值（直接用 python 浮点，避免 subs 残留符号）
lh_n, q_n, kI_n = 1.20, 0.050, 0.020
sf_n = (lh_n - 1) / lh_n
eff_crit = kI_n / q_n + sf_n
print(f"\ns_floor 数值 = {sf_n:.6f}")
print(f"omega*sigma 临界 = {eff_crit:.6f}")
print(f"sigma=1 时 omega_c = {eff_crit:.6f}")

# 动态阈值：omega_c(sigma) = (kI/q+sf)/sigma
print("\n动态阈值 omega_c(t) = %.4f / skill[t-1]" % (0.020/0.050 + sf_n))

# 独立核对：第38期 skill=0.8785 时动态阈值
sk37 = 0.8785
print("第38期动态阈值 = %.4f (skill=0.8785)" %
      ((0.020/0.050 + sf_n) / sk37))

# 验证 transversality：df(s_floor)/d omega
trans = sp.simplify(sp.diff(f_sf, om))
print("\ntransversality d f(s_floor)/d omega =", trans, " (>0)")
