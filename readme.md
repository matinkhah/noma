```markdown
# Hybrid GWO–GA for Energy-Efficient Resource Allocation in NOMA Fog Computing IoT

Reproduction package for the paper:

**Hybrid Grey Wolf and Genetic Algorithm for Energy-Efficient Resource Allocation in NOMA Fog Computing-Based IoT Systems**

This repository contains a **single, self-contained** experiment script that:

- implements the NOMA fog offloading model and all optimisers (RS, GA, GWO, PSO, DE, WOA, GWOGA, plus equal-allocation and all-local baselines);
- runs the full experimental suite (E1–E13) with identical settings across algorithms (same population size \(K\), same true fitness-evaluation budget, same random instances per seed);
- reports mean ± SD / 95 % CI, paired Wilcoxon tests with Holm correction, energy breakdown, population diversity, convergence curves, and sequential wall-clock timing;
- is **resumable** (finished experiments are skipped on restart) and produces a ready-to-download results bundle.

## Requirements

```bash
pip install numpy scipy matplotlib
```

Python ≥ 3.8 recommended. SciPy is optional but required for Wilcoxon / Holm tests.

## Quick start

```bash
# Smoke test (~2 min)
python run_experiments.py --quick --workers 2

# Reviewer-oriented run (E1, E5, E9–E12 + timing) — default
python run_experiments.py --only new --workers 2

# Full suite including E2–E8
python run_experiments.py --only all --workers 4

# Specific experiments
python run_experiments.py --only E1,E9,E11 --seeds 30
```

### Google Colab

```python
!pip -q install numpy scipy matplotlib
# upload run_experiments.py, then:
!python run_experiments.py --only new --workers 2
```

Runtime on a free Colab is roughly 1–2 hours for `--only new`. The run is **resumable**: if the runtime disconnects, re-execute the same command; finished experiments (saved as `results/raw_*.json`) are skipped. Download `colab_results_bundle.zip` from the file browser when finished.

## Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--seeds` | 30 | Independent random seeds |
| `--budget` | 10000 | True fitness evaluations per run |
| `--workers` | CPU count | Parallel processes |
| `--only` | `new` | `new` / `all` / comma list (`E1,E9,…`) |
| `--quick` | off | Few seeds, small budget (smoke test) |
| `--force` | off | Ignore saved `raw_*.json` and recompute |
| `--no-timing` | off | Skip sequential timing (E13) |
| `--long-seeds` | 10 | Seeds for long-run experiment E10 |
| `--scale-seeds` | 10 | Seeds for large-scale experiment E12 |
| `--timing-seeds` | 10 | Seeds for sequential timing E13 |
| `--outdir` | `images/media` | Figure directory |
| `--resdir` | `results` | Results directory |

## Experiments

| ID | Description |
|----|-------------|
| **E1** | Default scenario (convergence curves + final metrics) |
| **E2** | Number of tasks \(N\) |
| **E3** | Fog-node computing capacity \(F\) |
| **E4** | Population size \(K\) (fixed evaluation budget) |
| **E5** | Ablation (GWO, GWO+crossover, GWO+mutation, GWOGA) + diversity |
| **E6** | NOMA vs OMA |
| **E7** | Energy/delay weight trade-off |
| **E8** | GWOGA parameter sensitivity (\(p_c\), \(p_m\), \(\mu\)) |
| **E9** | Evaluation-budget sensitivity (2 k–50 k) |
| **E10** | Long-run reference (10× budget) |
| **E11** | Number of fog nodes \(M = 1,2,4\) |
| **E12** | Larger scale (\(N = 60,100\), scaled blocks/nodes) |
| **E13** | Sequential wall-clock timing + hardware/library versions |

Default configuration values appear in Table I of the paper and in the `DEFAULT` dict inside the script.

## Output

After a successful run you will find:

```
results/
  summary.txt          # human-readable numbers + statistical tests
  output_v2.csv        # flat per-run table
  curves_E1.csv        # convergence curves
  stats_tests.csv      # Wilcoxon + Holm-corrected p-values
  timing.json          # sequential timing (E13)
  raw_E*.json          # raw per-experiment results (used for resume)
  table_main.tex       # LaTeX body of the main results table

images/media/
  image1.png … image14.png   (+ matching .pdf)
  # image1  system model
  # image2  convergence
  # image3  energy vs tasks
  # image4  energy vs fog capacity
  # image5  delay vs tasks
  # image6  population size + run time
  # image7  ablation + diversity
  # image8  NOMA vs OMA
  # image9  weight trade-off
  # image10 p_c / p_m / μ sensitivity
  # image11 budget sensitivity
  # image12 long-run gap
  # image13 several fog nodes
  # image14 larger scale

colab_results_bundle.zip   # ready-to-share archive (summary, CSVs, figures)
```

## Reproducibility notes

- All algorithms share the same population size \(K\), the same true fitness-evaluation budget, and the same problem instances (seeded identically).
- Random number generators are deterministic given the seed and algorithm ID.
- Parallel runs use `ProcessPoolExecutor`; sequential timing (E13) is performed single-process to avoid interference.
- Constraint feasibility, energy breakdown (transmission vs local), and worst-case delay are reported alongside the primary normalised cost \(J\).

## Licence

Code released for research reproducibility of the associated paper. Please cite the paper if you use this implementation.
```

The file is already saved at:

**`/home/workdir/artifacts/README.md`**"# noma" 
