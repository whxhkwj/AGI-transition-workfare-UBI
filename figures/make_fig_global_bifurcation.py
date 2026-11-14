# -*- coding: utf-8 -*-
"""Global outcome bifurcation under the full-absorption public-employment policy.

Two figures:
  fig_global_bifurcation : crash basin vs smooth public-employment absorption
                           basin over the (capital tax, automation speed) plane.
  fig_absorption_path    : time paths of market employment, public employment,
                           residual unemployment and consumption on the smooth
                           full-absorption path.

All numbers come from the unified engine; nothing is hard-coded.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, _SIMDIR)
import double_task_sim_unified as U

plt.rcParams.update({
    "font.size": 8, "font.family": "DejaVu Sans",
    "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 6.8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": False, "mathtext.default": "regular",
})

BLUE = "#0072B2"; ORANGE = "#E69F00"; GREEN = "#009E73"
RED = "#D55E00"; PURPLE = "#CC79A7"; GREY = "#666666"; FILLGR = "#D8D8D8"
OUT = _os.path.join(_ROOT, "paper", "figures")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(_os.path.join(OUT, stem + "." + ext), bbox_inches="tight")
    plt.close(fig)
    print("saved", stem + ".pdf")


def run_full(tau, ga, keep_transfer=False):
    p = U.Params(eta=0.40, mu_TM=0.50, phi_max=0.02,
                 pe_policy="full_absorption",
                 full_keep_transfer=keep_transfer, tau=tau, gA=ga)
    r = U.simulate_unified(p)
    return r


# ---------- Figure: global bifurcation basin over tau x gA ----------
taus = np.linspace(0.05, 0.40, 15)
gas = np.linspace(0.01, 0.20, 20)
Smooth = np.zeros((len(gas), len(taus)), dtype=bool)
MinC = np.zeros_like(Smooth, dtype=float)
for i, ga in enumerate(gas):
    for j, tau in enumerate(taus):
        r = run_full(tau, ga)
        Smooth[i, j] = (r["outcome"] == "smooth")
        MinC[i, j] = np.min(r["C"][1:])

xi, yi = np.meshgrid(taus, gas)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.pcolor(xi, yi, np.ma.array(np.ones_like(MinC), mask=Smooth),
          cmap=ListedColormap([FILLGR]), shading="auto", zorder=0)
cm = ax.contourf(taus, gas, np.ma.array(MinC, mask=~Smooth),
                 levels=14, cmap="cividis", zorder=1)
cs = ax.contour(taus, gas, np.ma.array(MinC, mask=~Smooth),
                levels=7, colors="white", linewidths=0.45, alpha=0.7, zorder=2)
ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
fig.colorbar(cm, ax=ax, label="minimum consumption (smooth basin)",
             shrink=0.9, pad=0.02)

# critical boundary: for each tau, bisection on gA at the consumption floor.
crit = []
for tau in taus:
    lo, hi = 0.005, 0.25
    for _ in range(12):
        mid = (lo + hi) / 2
        r = run_full(tau, mid)
        if np.min(r["C"][1:]) >= 0.30 and r["outcome"] == "smooth":
            lo = mid
        else:
            hi = mid
    crit.append((lo + hi) / 2)
ax.plot(taus, crit, color="#8B1A1A", lw=1.6, zorder=5,
        label="critical boundary (floor)")
ax.text(0.06, 0.185, "grey = crash basin", fontsize=6.6, color=GREY, zorder=6)
ax.text(0.22, 0.035, "smooth public-employment absorption",
         fontsize=6.6, color="#0a4d68", zorder=6)
ax.set_title("Global outcome bifurcation: crash vs smooth public-employment "
             "absorption")
ax.set_xlabel(r"capital tax $\tau$")
ax.set_ylabel(r"automation frontier growth $g_A$")
ax.legend(loc="upper right")
fig.tight_layout()
save(fig, "fig_global_bifurcation")

# ---------- Figure: smooth full-absorption dynamic paths ----------
r = run_full(0.15, 0.01)
t = np.arange(len(r["sL"]))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
a = ax[0]
a.plot(t, r["sL"], color=BLUE, lw=1.4, label=r"market employment $s_L$")
a.plot(t, r["EP"], color=GREEN, lw=1.4, label=r"public employment $E^P/L^S$")
a.plot(t, r["Urem"], color=RED, lw=1.3, label=r"residual unemployment $U^{rem}/L^S$")
a.plot(t, r["sL"] + r["EP"] + r["Urem"], color="black", lw=0.8, ls=":",
       label="population identity (=1)")
a.set_title("(a) Employment reallocation under full absorption")
a.set_xlabel("period $t$"); a.set_ylabel("share of labor force"); a.legend()
a = ax[1]
a.plot(t, r["C"], color=BLUE, lw=1.4, label=r"aggregate consumption $C_t$")
a.axhline(0.30, color=RED, ls="--", lw=1.0, label=r"consumption floor $C^{min}$")
a.fill_between(t, 0, 0.30, color=RED, alpha=0.07)
a.set_title("(b) Consumption stays above the floor")
a.set_xlabel("period $t$"); a.set_ylabel("consumption"); a.legend()
fig.tight_layout()
save(fig, "fig_absorption_path")

print("global bifurcation figures done")
