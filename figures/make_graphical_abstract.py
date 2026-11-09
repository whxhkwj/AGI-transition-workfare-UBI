# -*- coding: utf-8 -*-
"""Graphical abstract for the JEDC submission.

Canvas equals the Elsevier printed size (13 x 5.3 cm = 5.12 x 2.15 in). Drawing
on a large canvas and scaling it down later would shrink the labels: a 13 in
canvas printed at 13 cm turns 7 pt into 2.7 pt. PNG is exported at 280 dpi,
giving 1433 x 602 px, above the required 1328 x 531 (w x h).

Panels (b)(c) come from the same v6 engine and the same corner configuration as
Figure 3 (omega = 0.05), so the printed buffer equals the manuscript's headline
min_t C_t = 0.309 rather than the endogenous-omega variant 0.305.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_SIMDIR = _os.path.join(_ROOT, "simulation")

import os, sys, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIM = _SIMDIR
OUT = _os.path.join(_ROOT, "paper")
_spec = importlib.util.spec_from_file_location("v6", os.path.join(SIM, "double_task_sim_v6.py"))
v6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v6)

plt.rcParams.update({
    "font.size": 6.8,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 7.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 5.6,
    "axes.grid": True, "grid.alpha": 0.20, "grid.linewidth": 0.4,
    "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": False,
    "mathtext.default": "regular",
})

BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
RED, PURPLE, GREY = "#D55E00", "#CC79A7", "#666666"

OMEGA_C = 0.5667
SF = 1 - 1 / 1.20
Cc = 0.003

fig, ax = plt.subplots(1, 3, figsize=(5.12, 2.15),
                       gridspec_kw=dict(width_ratios=[0.92, 1.04, 1.04]))
fig.subplots_adjust(left=0.075, right=0.985, top=0.845, bottom=0.15, wspace=0.40)

# ---------------- (a) the stable branch is destroyed ----------------
om = np.linspace(0.05, 1.0, 700)
sp = []
for o in om:
    B = 0.003 * o + 0.0005 - 0.0025
    A = o * (-0.0005 + 0.0025) - 0.001
    D = B * B + 4 * Cc * A
    sp.append((B + np.sqrt(D)) / (2 * Cc) if D >= 0 else np.nan)
sp = np.array(sp)
phys = sp >= SF
ax[0].axvspan(0.05, OMEGA_C, color=GREY, alpha=0.16, lw=0, zorder=0)
ax[0].plot(np.ma.array(om, mask=~phys), np.ma.array(sp, mask=~phys),
           color=BLUE, lw=1.7, zorder=3)
ax[0].axhline(SF, color="black", lw=0.6, alpha=0.5)
ax[0].axvline(OMEGA_C, color=PURPLE, ls="--", lw=0.9, zorder=2)
ax[0].plot(OMEGA_C, SF, marker="*", ms=6.5, color="black", zorder=5)
ax[0].text(0.075, 0.66, "no interior\nequilibrium", fontsize=6.0,
           color="#444444", ha="left", va="center", zorder=4)
ax[0].annotate(r"$\omega_c$", (OMEGA_C, SF), xytext=(OMEGA_C - 0.19, 0.44),
               fontsize=7.0, color=PURPLE,
               arrowprops=dict(arrowstyle="-", lw=0.5, color=PURPLE))
ax[0].set_xlim(0.05, 1.02)
ax[0].set_ylim(0.06, 0.82)
ax[0].set_xlabel(r"$\omega$")
ax[0].set_ylabel(r"$s$")
ax[0].set_title("(a) the branch is destroyed")

# ---------------- (b)(c) two regimes ----------------
configs = {
    "No policy": dict(tau=0.05, pE=0.0, phi_max=0.15, retrain=0.0, omega=0.05),
    "Workfare corner": dict(tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0, omega=0.05),
}
st = {"No policy": (RED, "--"), "Workfare corner": (BLUE, "-")}
res = {}
for name, cfg in configs.items():
    res[name] = v6.simulate_v6(v6.Params(eta=0.40, mu_TM=0.50, T=80, **cfg))
t = np.arange(len(res["Workfare corner"]["C"]))
MINC = float(np.min(res["Workfare corner"]["C"]))

for name, rr in res.items():
    c, ls = st[name]
    ax[1].plot(t, rr["EP"], color=c, ls=ls, lw=1.5, label=name)
    ax[2].plot(t, rr["C"], color=c, ls=ls, lw=1.5, label=name)

ax[1].axhline(0.45, color=GREY, ls=":", lw=0.9)
ax[1].legend(loc="upper left", fontsize=5.2, handlelength=1.6,
             borderpad=0.2, labelspacing=0.25)
ax[1].set_xlabel(r"$t$")
ax[1].set_ylabel(r"$e^{P}_t$")
ax[1].set_title("(b) policy moves the system")
ax[1].set_ylim(-0.02, 0.58)
ax[1].set_xlim(0, 79)

ax[2].axhline(0.30, color=RED, ls="--", lw=0.9)
ax[2].fill_between(t, 0, 0.30, color=RED, alpha=0.08)
ax[2].text(76, 0.225, r"$C^{\min}=0.30$", fontsize=6.3, color=RED, ha="right")
ax[2].text(24, 0.40, r"$\min_t C_t=%.3f$" % MINC, fontsize=6.3, color=BLUE)
ax[2].set_xlabel(r"$t$")
ax[2].set_ylabel(r"$C_t$")
ax[2].set_title("(c) the buffer holds")
ax[2].set_ylim(0.0, 0.95)
ax[2].set_xlim(0, 79)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, "JEDC_graphical_abstract." + ext),
                dpi=(280 if ext == "png" else None), facecolor="white")

for name in ("Workfare corner", "No policy"):
    rr = res[name]
    print("%-18s outcome=%s  minC=%.6f" % (name, rr["outcome"],
                                           float(np.min(rr["C"]))))
print("printed buffer label: %.3f" % MINC)
