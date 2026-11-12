# -*- coding: utf-8 -*-
"""Republish all six paper figures as vector PDF at near-print size.

Why the size matters: a matplotlib PDF keeps its own inch dimensions, so a
figure drawn 11 in wide and later scaled down to 0.98\\textwidth (~6.5 in)
shrinks 9 pt labels to roughly 5 pt. Every canvas below is therefore sized
close to its final printed width, with base font 8 pt.

Other changes:
  * vector PDF output (pdf.fonttype 42) instead of 150-dpi PNG
  * one muted colour-blind-safe palette for every figure
  * Figure 5: no hatch (vector hatch prints as a dense black mesh at this
    scale); infeasible cells are flat light grey instead
  * Figure 5 axis label shortened to "transition loss"; the phi bound is
    printed as a roman superscript rather than the italic mathtext default
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
from matplotlib.colors import ListedColormap


def _restore_native_findfont():
    """Use matplotlib's own font resolver instead of the sandbox CJK bridge.

    The AIO sandbox installs a startup patch (_aio_mpl_fontconfig, via a
    .pth file) whose CJK fallback injection replaces every Western-only
    font request with Noto Sans CJK SC, even when the requested family is
    explicitly DejaVu Sans. These figures contain no CJK text, so the
    injection is unwanted and changes the embedded font from DejaVu Sans
    (CID TrueType) to Noto Sans CJK (CID Type 0C), which poppler flags as
    a font-type mismatch. Re-load the font manager from its own source to
    recover the unpatched class-level findfont and bind it back. This
    affects only the current process; no system file is modified.
    """
    import matplotlib.font_manager as _fm
    if not getattr(_fm.fontManager.findfont, "_aio_mpl_fontconfig_patched", False):
        return
    _spec = importlib.util.spec_from_file_location("_fm_clean", _fm.__file__)
    _clean = importlib.util.module_from_spec(_spec)
    sys.modules["_fm_clean"] = _clean
    _spec.loader.exec_module(_clean)
    _native = _clean.FontManager.findfont
    _fm.FontManager.findfont = _native
    _bound = _native.__get__(_fm.fontManager, _fm.FontManager)
    _fm.fontManager.findfont = _bound
    _fm.findfont = _bound
    # Some submodules (notably matplotlib._mathtext) bind the module-level
    # findfont at import time and keep using the patched reference even after
    # font_manager is restored. Replace every such captured reference.
    for _name, _mod in list(sys.modules.items()):
        if _mod is None or not _name.startswith("matplotlib"):
            continue
        if getattr(_mod, "findfont", None) is not None:
            _mod.findfont = _bound


_restore_native_findfont()

# ---------- one shared style ----------
plt.rcParams.update({
    "font.size": 8,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.5,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": False,
    "mathtext.default": "regular",
})

BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
RED    = "#D55E00"
PURPLE = "#CC79A7"
GREY   = "#666666"
FILLGR = "#D8D8D8"

SIM = _SIMDIR
OUT = _os.path.join(_ROOT, "paper", "figures")
_spec = importlib.util.spec_from_file_location("v6", os.path.join(SIM, "double_task_sim_v6.py"))
v6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v6)


def P(**kw):
    base = dict(eta=0.40, mu_TM=0.50)
    base.update(kw)
    return v6.Params(**base)


def run(p):
    r = v6.simulate_v6(p)
    w = v6.welfare(r, p)
    return r, w


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, stem + "." + ext), bbox_inches="tight")
    plt.close(fig)
    print("saved", stem + ".pdf")


CORNER = dict(tau=0.15, pE=0.45, phi_max=0.02, omega=0.05, retrain=0.0)
r, w = run(P(**CORNER))
t = np.arange(len(r["sL"]))

# ---------- Figure 3: baseline smooth transition ----------
fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.8))
a = ax[0, 0]
a.plot(t, r["It"], color=BLUE, lw=1.2, label="automation frontier $I_t$")
a.plot(t, r["Nt"], color=ORANGE, lw=1.2, label="new-task bound $N_t$")
a.plot(t, r["nlab"], color=GREEN, lw=1.2, label="labor tasks $n^{lab}_t$")
a.set_title("(a) Task structure: displacement vs reinstatement")
a.set_xlabel("period $t$"); a.set_ylabel("task index"); a.legend()
a = ax[0, 1]
a.plot(t, r["sL"], color=BLUE, lw=1.2, label="labor share $s_t=L^D/L^S$")
a.axhline(0.05, color=RED, ls=":", lw=0.9, label="absorbing value 0.05")
a.set_title("(b) Labor share $s_t$")
a.set_xlabel("period $t$"); a.set_ylabel("share"); a.set_ylim(-0.02, 0.95); a.legend()
a = ax[1, 0]
a.plot(t, r["C"], color=BLUE, lw=1.2, label="aggregate consumption $C_t$")
a.axhline(r["Cmin"], color=RED, ls="--", lw=0.9, label="consumption floor $C^{min}=0.30$")
a.fill_between(t, 0, r["Cmin"], color=RED, alpha=0.07)
a.set_title("(c) Demand: consumption vs floor (outcome = smooth)")
a.set_xlabel("period $t$"); a.set_ylabel("consumption"); a.set_ylim(0, 0.75); a.legend()
a = ax[1, 1]
a.plot(t, r["GDP"], color=GREEN, lw=1.2, label="total output $GDP_t$")
a.set_title("(d) Aggregate: output growth (price effect)")
a.set_xlabel("period $t$"); a.set_ylabel("output"); a.legend()
fig.tight_layout()
save(fig, "fig_baseline")

# ---------- Figure 4: policy regime comparison ----------
regimes = {
    "No policy": dict(tau=0.05, pE=0.0, phi_max=0.15),
    "Workfare (corner)": dict(tau=0.15, pE=0.45, phi_max=0.02),
    "Pure transfer": dict(tau=0.0, pE=0.0, phi_max=0.65),
}
styles = {"No policy": (RED, "--"), "Workfare (corner)": (BLUE, "-"),
          "Pure transfer": (PURPLE, "-.")}
res = {}
for name, kv in regimes.items():
    rr, _ = run(P(omega=0.05, retrain=0.0, **kv))
    res[name] = rr
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
for name, rr in res.items():
    c, ls = styles[name]
    ax[0].plot(t, rr["sL"], ls=ls, color=c, lw=1.3, label=name)
    ax[1].plot(t, rr["C"], ls=ls, color=c, lw=1.3, label=name)
ax[0].axhline(0.05, color=GREY, ls=":", lw=0.9)
ax[0].set_title("(a) Labor share $s_t$ by regime")
ax[0].set_xlabel("period $t$"); ax[0].set_ylabel("labor share"); ax[0].legend()
ax[1].axhline(0.30, color=RED, ls="--", lw=0.9)
ax[1].set_title("(b) Consumption by regime (floor = 0.30)")
ax[1].set_xlabel("period $t$"); ax[1].set_ylabel("consumption"); ax[1].legend()
fig.tight_layout()
save(fig, "fig_policy_compare")

# ---------- Figure 5: loss surface over tau x pE ----------
taus = np.linspace(0.05, 0.30, 11)
pes = np.linspace(0.0, 0.60, 13)
Z = np.zeros((len(pes), len(taus)))
Feas = np.zeros_like(Z, dtype=bool)
for i, pe in enumerate(pes):
    for j, ta in enumerate(taus):
        rr, ww = run(P(tau=ta, pE=pe, phi_max=0.02, omega=0.05, retrain=0.0))
        Z[i, j] = ww["total"]
        Feas[i, j] = (rr["outcome"] == "smooth")
fig, ax = plt.subplots(figsize=(6.0, 4.4))
xi, yi = np.meshgrid(taus, pes)
ax.pcolor(xi, yi, np.ma.array(np.ones_like(Z), mask=Feas),
          cmap=ListedColormap([FILLGR]), shading="auto", zorder=0)
Zm = np.ma.array(Z, mask=~Feas)
cf = ax.contourf(taus, pes, Zm, levels=14, cmap="cividis", zorder=1)
cs = ax.contour(taus, pes, Zm, levels=7, colors="white", linewidths=0.45, alpha=0.7, zorder=2)
ax.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
fig.colorbar(cf, ax=ax, label="transition loss", shrink=0.9, pad=0.02)
bi, bj = np.argwhere(Feas)[np.argmin(Zm[Feas])]
ax.plot(taus[bj], pes[bi], marker="*", ms=10, color="#8B1A1A",
        markeredgecolor="black", markeredgewidth=0.4, zorder=6)
ax.annotate("minimum (wide-grid corner)", (taus[bj], pes[bi]),
            textcoords="offset points", xytext=(7, -12), color="#8B1A1A",
            fontsize=6.5, zorder=6)
# denser re-solve (phi_max -> 0.001) cannot be marked on this surface: at phi_max=0.02
# that (tau,pE) cell is infeasible, so mark it in words in the empty grey panel instead
ax.text(0.056, 0.27,
        "denser re-solve " + r"($\varphi^{\mathrm{max}}=0.001$): 66.70 at (0.13, 0.44)",
        fontsize=6.2, color="#8B1A1A", ha="left", zorder=6,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
ax.text(0.155, 0.050, "grey = infeasible or crash", fontsize=6.5, color=GREY, zorder=6)
ax.set_title("Loss surface over the capital-tax and workfare-share grid "
             r"($\varphi^{\mathrm{max}}=0.02$, $\omega=0.05$)")
ax.set_xlabel(r"capital tax $\tau$"); ax.set_ylabel(r"workfare share $p_E$")
fig.tight_layout()
save(fig, "fig_loss_surface")

# ---------- Figure 6: retraining value vs omega ----------
omegas = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
deltas = []
for om in omegas:
    _, wb0 = run(P(omega=om, tau=0.16, phi_max=0.65, pE=0.30, retrain=0.0))
    _, ws = run(P(omega=om, tau=0.16, phi_max=0.65, pE=0.30, retrain=0.8))
    deltas.append(wb0["total"] - ws["total"])
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(omegas, deltas, "o-", color=GREY, lw=1.3, ms=4)
ax.fill_between(omegas, deltas, 0, where=np.array(deltas) > 0, color=GREEN, alpha=0.20,
                label=r"net gain $\Delta>0$")
ax.fill_between(omegas, deltas, 0, where=np.array(deltas) <= 0, color=RED, alpha=0.20,
                label=r"no net gain $\Delta\leq0$")
ax.axhline(0, color="black", lw=0.7)
ax.set_title("Isolation experiment: net gain $\\Delta$ across $\\omega$")
ax.set_xlabel(r"reinstatement rate $\omega$ (labor-accruing share of new tasks)")
ax.set_ylabel(r"loss saved by retraining $\Delta$"); ax.legend()
fig.tight_layout()
save(fig, "fig_skill_omega")

# ---------- Figures 1-2: BEB (v12 endogenous omega) ----------
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.join(_ROOT, "analysis"))
import double_task_sim_v12_endogenous_omega as v12
import dynamic_threshold as dth

OMEGA_C = 0.5667
SF = 1 - 1 / 1.20
Cc = 0.003


def coeffs(om):
    B = 0.003 * om + 0.0005 - 0.0025
    A = om * (-0.0005 + 0.0025) - 0.001
    return B, A


fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
om_grid = np.linspace(0.05, 1.0, 600)
s_plus, s_minus = [], []
for om in om_grid:
    B, A = coeffs(om)
    D = B * B + 4 * Cc * A
    if D >= 0:
        s_plus.append((B + np.sqrt(D)) / (2 * Cc))
        s_minus.append((B - np.sqrt(D)) / (2 * Cc))
    else:
        s_plus.append(np.nan); s_minus.append(np.nan)
s_plus, s_minus = np.array(s_plus), np.array(s_minus)
a = ax[0]
phys = s_plus >= SF
a.plot(np.ma.array(om_grid, mask=~phys), np.ma.array(s_plus, mask=~phys),
       color=BLUE, lw=1.6, label="stable equilibrium $s^+$ (region I)")
virt = (s_plus >= 0) & (s_plus < SF)
a.plot(np.ma.array(om_grid, mask=~virt), np.ma.array(s_plus, mask=~virt),
       color=ORANGE, lw=1.3, ls="--", label="virtual branch (region II)")
neg = s_plus < 0
a.plot(np.ma.array(om_grid, mask=~neg), np.ma.array(s_plus, mask=~neg),
       color=GREY, lw=1.1, ls=":", label="non-physical ($s^+<0$)")
a.plot(om_grid, s_minus, color=RED, lw=1.0, ls="-.", label="unstable branch $s^-$")
a.axhline(SF, color="black", ls="-", lw=0.7, alpha=0.6)
a.text(0.07, SF + 0.03, r"switching manifold $s_{floor}$", fontsize=6.5)
a.set_xlim(0.04, 1.02)
a.axvline(OMEGA_C, color=PURPLE, ls="--", lw=1.1)
a.plot(OMEGA_C, SF, marker="*", ms=9, color="black")
a.annotate("BEB\n" + r"$\omega_c=0.567$", (OMEGA_C, SF),
           xytext=(OMEGA_C - 0.26, 0.30), fontsize=6.5, color=PURPLE)
a.set_xlabel(r"reinvestment rate $\omega$")
a.set_ylabel(r"equilibrium employment share $s$")
a.set_title("(a) Fast-subsystem BEB, static slice (skill = 1)")
a.set_ylim(-0.5, 0.85); a.legend(loc="upper left")

a = ax[1]
f_sf = np.array([0.0025 * (om - OMEGA_C) for om in om_grid])
a.plot(om_grid, f_sf, color=GREEN, lw=1.4)
a.axhline(0, color="black", lw=0.7)
a.axvline(OMEGA_C, color=PURPLE, ls="--", lw=1.1)
a.fill_between(om_grid, f_sf, 0, where=f_sf < 0, color=RED, alpha=0.12,
               label="absorbing: no interior equilibrium")
a.fill_between(om_grid, f_sf, 0, where=f_sf >= 0, color=BLUE, alpha=0.10,
               label="stable interior equilibrium exists")
a.set_xlabel(r"reinvestment rate $\omega$")
a.set_ylabel(r"vector field on the manifold $f(s_{floor})$")
a.set_title(r"(b) Transversality at $\omega_c$, static slice (skill = 1)")
a.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none",
         framealpha=0.9)
fig.tight_layout()
save(fig, "fig_beb_static")

configs = {
    "No policy": dict(tau=0.05, pE=0.0, phi_max=0.15, retrain=0.0),
    "Workfare (corner)": dict(tau=0.15, pE=0.45, phi_max=0.02, retrain=0.0),
    "Strong social floor": dict(tau=0.15, pE=0.80, phi_max=0.65, retrain=0.8),
}
styles12 = {"No policy": (RED, "--"), "Workfare (corner)": (BLUE, "-"),
            "Strong social floor": (GREEN, "-.")}
res = {}
for name, cfg in configs.items():
    p = v12.Params(eta=0.40, mu_TM=0.50, T=80, **cfg)
    res[name] = v12.simulate_v12(p)
t12 = np.arange(80)

fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.9))
for name, rr in res.items():
    c, ls = styles12[name]
    ax[0].plot(t12, rr["omega"], color=c, ls=ls, lw=1.3, label=name)
    # instantaneous BEB threshold for this path: EFF_CRIT / skill[t-1]
    dyn = dth.dynamic_threshold(rr["skill"])
    ax[0].plot(t12, dyn, color=c, lw=0.7, ls=(0, (2, 2)), alpha=0.65)
    real_t = dth.first_real_crossing(rr["omega"], rr["skill"])
    if real_t is not None:
        ax[0].plot(real_t, rr["omega"][real_t], marker="*", ms=8,
                   color="black", zorder=6)
    ax[1].plot(t12, rr["sL"], color=c, ls=ls, lw=1.3, label=name)
    ax[2].plot(t12, rr["C"], color=c, ls=ls, lw=1.3, label=name)
ax[0].set_title(r"(a) Endogenous $\omega_t$ and dynamic $\omega_c(t)$")
ax[0].set_xlabel("period $t$")
ax[0].set_ylabel("reinvestment rate")
ax[0].annotate("dotted: dynamic threshold; $*$ = real crossing",
               xy=(0.03, 0.96), xycoords="axes fraction",
               fontsize=5.8, color=GREY)
ax[1].set_title(r"(b) Employment share $L^D/L^S$"); ax[1].set_xlabel("period $t$")
ax[1].set_ylabel("employment share")
ax[2].axhline(0.30, color=RED, ls="--", lw=0.9)
ax[2].fill_between(t12, 0, 0.30, color=RED, alpha=0.07)
ax[2].set_title("(c) Consumption (floor = 0.30)"); ax[2].set_xlabel("period $t$")
ax[2].set_ylabel("consumption")
h, l = ax[0].get_legend_handles_labels()
fig.tight_layout(rect=[0, 0.10, 1, 1])
fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5, frameon=False,
           bbox_to_anchor=(0.5, 0.0))
save(fig, "fig_beb_policy")

print("all figures republished as PDF:", sorted(f for f in os.listdir(OUT) if f.endswith(".pdf")))
