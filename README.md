# Replication package

**Anatomy of a Smooth Transition: A Task-Based Macroeconomic Model of AGI
Displacement with Fiscal and Social Safety Nets**

DUAN Fang (Huaqiao University) — prepared for submission to the *Structural
Change and Economic Dynamics*.

This repository reproduces every number that appears in the manuscript: the
simulation engine, the analysis scripts behind the tables and the sensitivity
paragraphs, and the figure code. Running `analysis/replicate_workfare_vs_ubi.py`
takes about half a minute and prints a pass/fail check of nine headline claims
against the values printed in the paper.

---

## Layout

```
simulation/   the unified engine (one physical law); v6/v12 are thin wrappers
analysis/     one script per number in the paper; each prints the value it checks
figures/      the six manuscript figures and the graphical abstract
paper/        LaTeX source, compiled PDF, cover letter, graphical abstract
```

Nothing in the repository depends on an absolute path: every script resolves
`simulation/` relative to its own location, so a fresh clone runs as is.

## Requirements

Python 3.9 or later, and

```
pip install -r requirements.txt      # numpy, matplotlib, sympy
```

`paper/` additionally needs pdfLaTeX. `elsarticle.cls` is shipped in `paper/`,
so the build also works on a TeX distribution that does not carry it:

```bash
cd paper
pdflatex manuscript.tex
pdflatex manuscript.tex      # second pass for cross-references
```

The result is the submitted PDF: 55 pages, no errors, no undefined references.
This was verified from a clean copy of the repository on TeX Live 2022.

## Quick check

```bash
python analysis/replicate_workfare_vs_ubi.py
```

Expected tail:

```
checks against the manuscript
  standard workfare survives 6 of 6              PASS
  standard pure transfer survives 0 of 6         PASS
  corner p_E=0.45 survives 6 of 6                PASS
  corner p_E=0.30 survives 0 of 6                PASS
  ...
9/9 checks passed
```

The script exits non-zero if any check fails.

## Calibration: read this before running anything by hand

The engine files are the ones the paper was produced with, and their `Params`
defaults already carry the paper calibration:

| parameter | engine default | paper calibration |
|---|---|---|
| `mu_TM` | 0.50 | **0.50** |
| `eta`   | 0.40 | **0.40** |

(`tau=0.3`, `pE=0.30`, `phi_max=0.65` and `phi_base=0.15` are scenario
parameters and are always set explicitly by the callers.)

A fresh `Params()` therefore matches the paper without overriding these two.
Every script in `analysis/` still passes them explicitly for clarity; do the
same in any experiment of your own:

```python
CAL = dict(mu_TM=0.50, eta=0.40)
r = v6.run_case(v6.Params(omega=0.05, tau=0.15, phi_max=0.02, pE=0.45,
                          retrain=0.0, **CAL))
```

## Engine chain

`double_task_sim_unified.py` is the single physical engine and depends only on
`v4`, which supplies the base `Params` and the accounting primitives. The
reinstatement rate follows one competitive logistic law,
`logit(omega_t) = ell0 + Gamma_t - beta_A * g_t`, where `Gamma_t` is the policy
reattachment stock and `g_t = log(A_t/A_0)` is the AGI capability gain.
`double_task_sim_v6.py` and `double_task_sim_v12_endogenous_omega.py` are thin
compatibility wrappers over the same engine: `simulate_v6` forces the exogenous
special case, `omega_t = omega`, and `simulate_v12` forces the endogenous law.
There is no separate physics in either wrapper. Setting `beta_A=0` gives the
policy-only limit. Keep the three files together; the targeted sweep scripts
cited in the robustness passages run against this same engine.

```python
import importlib.util
spec = importlib.util.spec_from_file_location("uni", "simulation/double_task_sim_unified.py")
uni = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uni)          # spec.loader.exec_module, not module_from_spec alone
```

The endogenous defaults are `omega_0=0.05`, `gamma_PE=4.5`, `gamma_TM=3.0`,
`gamma_R=2.25`, `beta_A=1.0`, with a skill floor of `0.30`.

`uni.run_case(par)` returns a dict with `outcome` (`smooth` / `crash` /
`not_reached`), `crash_type` (`econ` / `both`), `minC`, `total` and `sL_end`.
`uni.simulate_unified(par)` returns the full path; `uni.welfare(path, par)`
returns `minC`, `total` and the decomposition.

## Where each number comes from

### Results table, Section 5

| Manuscript | Script | Key values |
|---|---|---|
| Table 2, Standard / Pure transfer / corner | `analysis/check_table2.py` | 0.3576, 0.2589, 0.3568 |
| Workfare vs transfer, same fiscal, 6/6 not crashed vs 6/6 | `analysis/replicate_workfare_vs_ubi.py` | six omega values, both settings |
| Section 5.2, load-bearing paragraph | `analysis/replicate_workfare_vs_ubi.py` | corner 0.352–0.357 at p_E=0.45, 0.244–0.265 at p_E=0.30 |
| 0.357 vs 0.353 attribution | `analysis/check_phi_gap.py` | phi_max floor |
| phi_max lower bound | `analysis/check_phi_floor.py` | 0.02 boundary |
| Horizon, T = 80 vs 200 | `analysis/check_horizon.py` | exogenous and endogenous |

### Robustness and structural sections

| Manuscript | Script | Key values |
|---|---|---|
| Grid minima and identification | `analysis/check_grid_refine.py` | (0.18, 0.40, 0.15) = 63.53; dense grid (0.10, 0.40, 0.001) = 62.82, min C 0.3070 |
| Link-function sensitivity (logistic vs probit) | `analysis/check_link_function.py` | min C 0.3562 vs 0.3571; link does not change the outcome |
| Deflation nesting (consumption-smooth) | `analysis/check_deflation_nesting.py` | longest negative run 79, real debt capped at 0.80, min C 0.3562 |
| Shock Monte Carlo (absorption counts) | `analysis/check_shock_mc.py` | headline crash counts 0/0/0, minima 0.3540, 0.3348 |
| g_A and eta sensitivity | `analysis/check_ga_eta.py` | 0.344 / 0.294 / 0.237 / 0.222 |
| Feasible share of the parameter box | `analysis/check_feasible_share.py` | 41/64 = 64.1%, 17/64 = 26.6%, 208/384 = 54.2% |
| Social-threshold bounds, alpha_s crit | `analysis/check_soc_bounds.py` | varsigma_th 0.32/0.33 boundary, alpha_s 0.90/0.92 boundary (crit 0.909) |
| Empirical-range alpha_s sweep (0.30–6.62) | `analysis/empirical_range_alpha_sweep.py` | ranking robust on full interval; corner smooth through 0.90, social crash at 0.91 |
| Two calibrated configurations | `analysis/check_two_configs.py` | 0.3576 at p_E=0.45 (smooth), 0.3326 at p_E=0.30 (smooth) |
| Welfare weights (384 combinations) | `analysis/check_welfare_weights.py` | wide-grid best (0.10, 0.40, 0.02) = 62.84 under baseline weights |
| BEB and the policy that avoids the crash | `analysis/check_v12_beb.py` | endogenous re-employment |
| p_E* crossing | `analysis/check_beb_crossing.py` | 0.3661, omega_c = 0.5667 |
| Static bifurcation diagram, long horizon | `analysis/check_static_bifurcation.py` | sympy, no path dependence |
| Reversal of the policy ranking | `analysis/check_reversal.py` | parameter region |
| Fold region diagnostics | `analysis/check_fold_region.py` | local fold locus |
| Cross-checks between tables | `analysis/check_tables_cross.py` | internal consistency |
| Graphical-abstract annotation | `analysis/check_ga_panel.py` | corner 0.357, no-policy 0.057 |

### Figures

```bash
python figures/republish_figures.py        # six figures -> paper/figures (about 2 min)
python figures/make_graphical_abstract.py  # -> paper/, 1433 x 602 px
```

Both read the same engine and the same calibration as the manuscript, and both
write into `paper/`, so `paper/manuscript.tex` picks the new figures up
directly through `\graphicspath{{figures/}}`.

## Runtime

A single simulation is about 1.5 s, so the cheap scripts finish in seconds and
the grid sweeps take minutes:

| Script | Cost |
|---|---|
| `check_table2`, `check_pestar`, `check_phi_gap`, `check_beb_crossing`, ... | < 10 s |
| `check_feasible_share` (384 cases) | ~10 min |
| `check_grid_refine` (540 cells) | ~15 min |
| `check_shock_mc` (10 cells x 200 draws) | ~50 min |

`check_shock_mc` accepts a reduced draw count for a smoke test; the published
results use the default of 200:

```bash
MC_DRAWS=20 python analysis/check_shock_mc.py
```

## Licence

MIT, see `LICENSE`. The manuscript text and figures remain under the author's
copyright; the MIT licence covers the code in `simulation/`, `analysis/` and
`figures/`.

## Contact

DUAN Fang — duan.fang@hqu.edu.cn, College of Information Science and
Engineering, Huaqiao University, Xiamen, China.
