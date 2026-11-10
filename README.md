# Replication package

**Anatomy of a Smooth Transition: A Task-Based Macroeconomic Model of AGI
Displacement with Fiscal and Social Safety Nets**

DUAN Fang (Huaqiao University).

This repository reproduces every number that appears in the manuscript: the
simulation engine, the analysis scripts behind the tables and the sensitivity
paragraphs, and the figure code. Running `analysis/replicate_workfare_vs_ubi.py`
takes about half a minute and prints a pass/fail check of nine headline claims
against the values printed in the paper.

---

## Layout

```
simulation/   the model, one file per refinement step (v4 -> v5 -> v6 -> v12)
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

The result is the compiled PDF: 47 pages, no errors, no overfull boxes. This
was verified from a clean copy of the repository on TeX Live 2022.

## Quick check

```bash
python analysis/replicate_workfare_vs_ubi.py
```

Expected tail:

```
checks against the manuscript
  standard workfare survives 4 of 6              PASS
  standard pure UBI survives 0 of 6              PASS
  corner p_E=0.45 survives 6 of 6                PASS
  corner p_E=0.30 survives 0 of 6                PASS
  ...
9/9 checks passed
```

The script exits non-zero if any check fails.

## Calibration: read this before running anything by hand

The engine files are the ones the paper was produced with, and their
`Params` defaults carry two values left over from an earlier exploratory run:

| parameter | engine default | paper calibration |
|---|---|---|
| `mu_TM` | 0.70 | **0.50** |
| `eta`   | 0.35 | **0.40** |

(`tau=0.3`, `pE=0.30`, `phi_max=0.65` and `phi_base=0.15` are scenario
parameters and are always set explicitly by the callers.)

If you instantiate `Params()` without passing `mu_TM=0.50, eta=0.40` you will
get numbers that differ from the paper in the third digit. Every script in
`analysis/` passes them explicitly; do the same in any experiment of your own:

```python
CAL = dict(mu_TM=0.50, eta=0.40)
r = v6.run_case(v6.Params(omega=0.05, tau=0.15, phi_max=0.02, pE=0.45,
                          retrain=0.0, **CAL))
```

## Engine chain

`double_task_sim_v6.py` loads `v5` from disk, and `v5` loads `v4`. Keep the
three files together. `v12_endogenous_omega.py` loads `v6` the same way and adds
the endogenous re-employment channel used in Section 4; `v9`, `v10` and `v11`
are the targeted sweeps cited in the robustness passages.

```python
import importlib.util
spec = importlib.util.spec_from_file_location("v6", "simulation/double_task_sim_v6.py")
v6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v6)          # spec.loader.exec_module, not module_from_spec alone
```

`v6.run_case(par)` returns a dict with `outcome` (`smooth` / `crash` /
`not_reached`), `crash_type` (`econ` / `both`), `minC`, `total` and `sL_end`.
`v6.simulate_v6(par)` returns the full path; `v6.welfare(path, par)` returns
`minC`, `total` and the decomposition.

## Where each number comes from

### Results table, Section 5

| Manuscript | Script | Key values |
|---|---|---|
| Table 2, Standard / Pure UBI / corner | `analysis/check_table2.py` | 0.2999, 0.2100, 0.309266 |
| "Workfare vs UBI", 4/6 not crashed vs 6/6 | `analysis/replicate_workfare_vs_ubi.py` | six omega values, both settings |
| Section 5.2, load-bearing paragraph | `analysis/replicate_workfare_vs_ubi.py` | corner 0.304–0.310 at p_E=0.45, 0.197–0.218 at p_E=0.30 |
| 0.309 vs 0.305 attribution | `analysis/check_phi_gap.py` | phi_max floor |
| phi_max lower bound | `analysis/check_phi_floor.py` | 0.02 boundary |
| Horizon, T = 80 vs 200 | `analysis/check_horizon.py` | exogenous and endogenous |

### Robustness and structural sections

| Manuscript | Script | Key values |
|---|---|---|
| Grid minima and identification | `analysis/check_grid_refine.py` | (0.18, 0.40, 0.15) = 67.20; dense grid (0.13, 0.44, 0.001) = 66.70, min C 0.3004 |
| Alternative omega functional forms | `analysis/check_omega_form.py` | min C 0.3046-0.3058 across exp / rational / clip; no form changes the outcome |
| Shock Monte Carlo (absorption counts) | `analysis/check_shock_mc.py` | crash counts 0/1/6, minima 0.3046, 0.2738 |
| g_A and eta sensitivity | `analysis/check_ga_eta.py` | 0.283 / 0.228 / 0.166 / 0.150 |
| Feasible share of the parameter box | `analysis/check_feasible_share.py` | 29/64 = 45.3%, 6/64 = 9.4%, 122/384 = 31.8% |
| Social-threshold bounds, alpha_s crit | `analysis/check_soc_bounds.py` | varsigma_th 0.32/0.33 boundary, alpha_s 0.90/0.92 boundary (crit 0.909) |
| Two calibrated configurations | `analysis/check_two_configs.py` | 0.2999 at p_E=0.45, 0.2758 at p_E=0.30 |
| Welfare weights (384 combinations) | `analysis/check_welfare_weights.py` | wide-grid best (0.15, 0.45, 0.02) = 66.86 under baseline weights |
| BEB and the policy that avoids the crash | `analysis/check_v12_beb.py` | endogenous re-employment |
| p_E* crossing | `analysis/check_beb_crossing.py` | 0.4388, omega_c = 0.5667 |
| Static bifurcation diagram, long horizon | `analysis/check_static_bifurcation.py` | sympy, no path dependence |
| Reversal of the policy ranking | `analysis/check_reversal.py` | parameter region |
| Fold region diagnostics | `analysis/check_fold_region.py` | local fold locus |
| Cross-checks between tables | `analysis/check_tables_cross.py` | internal consistency |
| Graphical-abstract annotation | `analysis/check_ga_panel.py` | 0.309 |

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
| `check_table2`, `check_pestar`, `check_phi_gap`, `check_omega_form`, ... | < 10 s |
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

DUAN Fang — duan.fang@hqu.edu.cn, School of Information Science and
Engineering, Huaqiao University, Xiamen, China.
