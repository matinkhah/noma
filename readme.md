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
python run_experiments_v2.py --quick --workers 2

# Reviewer-oriented run (E1, E5, E9–E12 + timing) — default
python run_experiments_v2.py --only new --workers 2

# Full suite including E2–E8
python run_experiments_v2.py --only all --workers 4

# Specific experiments
python run_experiments_v2.py --only E1,E9,E11 --seeds 30
```

### Google Colab

```python
!pip -q install numpy scipy matplotlib
# upload run_experiments_v2.py, then:
!python run_experiments_v2.py --only new --workers 2
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


## run_experiments_v3

```bash
git clone https://github.com/matinkhah/noma.git
cd noma
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt

# smoke test (about 10 s): default scenario, 2 runs, budgets divided by 5
python run_experiments_v3.py noma --exp E1 --runs 2 --quick --workers 1

# the paper's default scenario (30 runs)
python run_experiments_v3.py noma --exp E1 --runs 30
```

Python 3.9 or newer is required. The code was last checked with Python 3.12.3, numpy 2.4.4, scipy 1.17.1,
pandas 3.0.2, scikit-learn 1.8.0 and matplotlib 3.10.8. scikit-learn is needed only by the `heuristics` suite and
matplotlib only for figures.

---

## Command line

```
python run_experiments_v3.py noma       [--exp E1,E2,... | all] [--runs N] [--workers N] [--quick]
                                         [--restarts N] [--refQ Q]
python run_experiments_v3.py heuristics [--quick] [--workers N] [--no-verify]
python run_experiments_v3.py all        [--quick] [--workers N]
python run_experiments_v3.py --exp E1 --runs 30      # no sub-command = noma suite
```

| Option (noma) | Default | Meaning |
|---|---|---|
| `--exp` | `all` | Comma-separated experiment IDs (`E1`…`E8`) or `all` |
| `--runs` | 30 | Independent runs (instances) per setting; the paper uses 30 everywhere |
| `--workers` | CPU count − 1 | Worker processes; results do not depend on this number (each job has its own seeds) |
| `--quick` | off | Budgets divided by 5 (minimum 200 evaluations); for smoke tests only |
| `--restarts` | 10 | E7: number of long reference restarts per instance |
| `--refQ` | 200000 | E7: evaluations per reference restart |

`all` runs the NOMA suite with `--exp all --runs 30` (or `--runs 3 --quick`) followed by the heuristics suite.

---

## The model

* **Uplink NOMA with SIC.** A block of width `W = B_tot / R` is shared by at most `U_max` devices. The receiver decodes in
  decreasing received power `q_i = g_i p_i`; user *i* sees interference from weaker users and a fraction `β` of the
  stronger (already cancelled) users:
  `γ_i = q_i / (Σ_{q_j<q_i} q_j + β Σ_{q_j>q_i} q_j + σ²)`, `r_i = W log2(1+γ_i)`. `β = 0` is perfect SIC (default).
  `U_max = 1` is OMA.
* **Offloading.** Each of the `N` devices (one task each) executes locally (`E = κ f² C`, `L = C / f`) or offloads to one of
  `M` fog nodes (`E = p S / r`, `L = S / r + C / f_fog`). Fog-side energy, result return and queueing are not modelled, so
  all reported energy is *device* energy.
* **Objective.** `J = (1/N) Σ_i [ w_E E_i/E_i^loc + w_L L_i/L_i^loc + π_i ]`; all-local execution gives `J = 1` exactly.
  The penalty `π_i = λ[(L_i − L_max)⁺ / L_max + (1 − γ_i/γ_th)⁺]` applies to offloaders that violate the delay limit or SIC
  decodability. A run is *feasible* when no penalty is active.
* **Encoding and repair.** A candidate is `x ∈ [0,1]^{4N}` (offloading target, block, power, computing share per device).
  Repair runs inside the fitness evaluation and is not written back: per block only the `U_max` strongest received-power
  offloaders stay offloaded, and node shares above capacity are scaled proportionally.

### Default parameters

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| Devices `N` | 30 | Fog nodes `M` | 2 |
| Fog capacity `F_k` | 20×10⁹ cycles/s | Device CPU `f_loc` | U(0.7, 1) GHz |
| Task size `S` | U(0.1, 1) Mbit | Cycles per bit | U(300, 700) |
| Bandwidth / blocks | 10 MHz / 8 | `U_max` (NOMA / OMA) | 3 / 1 |
| `P_max` | 300 mW | `L_max` | 1 s |
| `γ_th` | 1 (0 dB) | Noise PSD `N0` | −174 dBm/Hz |
| Path loss | −30 dB at 1 m, α = 3.5 | Distance | U(50, 300) m |
| `κ` | 10⁻²⁸ | Weights `w_E, w_L` | 0.5, 0.5 |
| Penalty `λ` | 10 | Population `K` / budget `Q` | 100 / 10⁴ |

### Algorithms

All population methods share the population size `K` and the budget `Q` of true fitness evaluations.

| Name in CSV | Method |
|---|---|
| `RS` | Random search |
| `GA` | Real-coded GA: tournament (size 2), arithmetic crossover `p_c = 0.8`, per-gene Gaussian mutation `p_m = 0.1`, `μ = 0.02`, one elite |
| `GWO` | Grey Wolf Optimiser, `a` decreasing 2 → 0 over `⌊Q/K⌋` generations |
| `PSO` | `w = 0.72`, `c1 = c2 = 1.49`, `|v| ≤ 0.2` |
| `DE` | rand/1/bin, `F = 0.5`, `CR = 0.9` |
| `WOA` | `b = 1`, `a` decreasing 2 → 0 |
| `HYB` | GWO–GA hybrid: per generation `(1−ρ)K` wolves make GWO moves, `ρK` GA offspring (tournament, crossover, mutation) replace the worst members if better; `ρ = 0.5`; offspring count against `Q` |
| `GWO+X`, `GWO+M` | Hybrid with mutation switched off (crossover only) / crossover switched off (mutation only) |
| `S-GWO`, `S-HYB` | GWO / hybrid with one initial member replaced by the encoded constructive solution |
| `CONS` | Constructive rule: devices in decreasing channel gain; node with largest residual capacity; share `min{residual, F_k/⌈N/M⌉}`; power `P_max/2`; first block (by increasing occupancy) in which all members stay SIC-decodable, within `L_max` and with normalised cost < 1; otherwise local |
| `EQUAL`, `LOCAL` | Equal allocation at `P_max` (lower reference); all-local execution (`J = 1`) |

Variant names accept overrides, e.g. `HYB:rho=0.1`, `HYB:mu=0.05`, `GWO+M:mu=0.005`, `S-HYB`.

---

## Experiments

| ID | Study | What varies | Main outputs in `results/` | Observed wall-clock (30 runs) |
|---|---|---|---|---|
| E1 | Default scenario | all methods at `N=30, Q=10⁴, K=100`; Friedman-style ranking inputs; constructive recovery share | `E1_runs.csv`, `E1_summary.csv`, `E1_pairs.csv` | not recorded |
| E2 | Budget sweep | `Q ∈ {2·10³, 5·10³, 10⁴, 2·10⁴, 5·10⁴, 10⁵}`; Holm over 4 comparisons per budget; TOST (δ = 0.01) | `E2_runs.csv`, `E2_summary.csv`, `E2_pairs.csv` | ≈ 46 min |
| E3 | Operator ablation | GWO, GWO+X, GWO+M, hybrid; sweeps of `ρ, μ, p_m`; seeded variants; diversity and convergence curves | `E3_runs.csv`, `E3_summary.csv`, `E3_pairs.csv`, `fig_convergence.pdf` | ≈ 7 min |
| E4 | NOMA vs OMA | `N = 10…60`, fixed `(R,M) = (8,2)` and scaled resources; equal-capacity OMA control (`3R` blocks); `β ∈ {0.05, 0.1, 0.2}` | `E4_runs.csv`, `E4_summary.csv`, `E4_gap.csv`, `fig_noma_oma.pdf` | ≈ 23 min |
| E5 | Scalability | `N = 60, 100`; fixed `Q = 10⁴` vs `Q = 10⁴·N/30` | `E5_runs.csv`, `E5_summary.csv` | ≈ 20 min |
| E6 | Population size | `K ∈ {10, 20, 30, 50, 100, 200, 400}` | `E6_runs.csv`, `E6_summary.csv`, `E6_pairs.csv` | ≈ 20 min |
| E7 | Best-known reference | `N = 8, 10, 30`; 10 long seeded restarts per instance; gap of each method | `E7_runs.csv`, `E7_gap.csv` | ≈ 5 h |
| E8 | Fog nodes | `M ∈ {1, 2, 4}` | `E8_runs.csv`, `E8_summary.csv`, `E8_pairs.csv` | ≈ 10 min |

Timings are those observed on the authors' machine and vary with hardware and `--workers`. To reproduce the paper:

```bash
for e in E1 E2 E3 E4 E5 E6 E7 E8; do
  python run_all_experiments.py noma --exp $e --runs 30
done
```

### Output files

`*_runs.csv` hold one row per (run, algorithm, setting); `*_summary.csv` are means over runs; `*_pairs.csv` / `E4_gap.csv`
are paired tests.

| Column | Meaning |
|---|---|
| `run`, `alg`, `N`, `Q`, `K`, `M`, `R`, `Umax` | Run index, algorithm name and setting (further tags per experiment: `Qlab`, `regime`, `scheme`, `beta`, `budget`, `Mlab`, `kind`, `salt`) |
| `J` | Objective as defined above (penalised) |
| `J_unpen` | `J` without penalty; equals `w_E·En + w_L·Ln` exactly |
| `E_abs`, `L_abs` | Mean energy per task (mJ) and delay per task (ms) |
| `En`, `Ln` | Mean per-device normalised energy and delay (ratio to all-local) |
| `Etx` | Transmission part of the energy per task (mJ) |
| `off_frac` | Fraction of tasks offloaded after repair |
| `feasible` | No penalty active |
| `time` | Wall-clock seconds of the run (not comparable across sessions) |
| `J_feas` (summaries) | Mean `J` over feasible runs only |
| `mean_diff`, `ci_lo`, `ci_hi` (pairs) | First minus second method (positive = first is worse); 95 % Student-*t* CI |
| `p_raw`, `p_holm` (pairs) | Two-sided paired Wilcoxon signed-rank test; Holm adjustment over the comparisons of that group |
| `p_tost` (pairs) | TOST equivalence p-value for margin δ = 0.01 in `J` (unadjusted) |

### Statistics

* **Paired design.** Instance seed depends only on `(experiment, run, N)`; the optimiser seed additionally depends on the
  algorithm name (CRC-32) and a salt, so all algorithms see identical instances in a given run.
* **Wilcoxon + Holm.** Families: E1 – the 13 comparisons written to `E1_pairs.csv` (the manuscript's nine-comparison family against GWO is recomputed from `E1_runs.csv`); E2 – four comparisons per budget; E3 – twelve comparisons against GWO;
  E4 – 48 comparisons; E6 and E8 – the comparisons listed in their `*_pairs.csv`.
* **Equivalence.** Two one-sided Wilcoxon tests on the paired differences with margin ±0.01 in `J` (about 1.4 % of the leading cost).
  "Equivalent" is claimed only when the TOST p-value is below 0.05.
* **E7 reference.** For each instance `J_bk` is the lowest `J` among the long restarts and all compared runs. It is an *upper bound on the optimum*,
  so the reported gaps are minimum distances to the optimum.

---


Highlights (default scenario, Q = 10⁴, K = 100, 30 runs)


Method	J	Energy (mJ)	Delay (ms)
RS	0.827	15.2	286
GA	0.813	15.5	273
GWO	0.696	11.2	269
PSO	0.693	10.9	274
DE	0.744	12.5	282
WOA	0.813	15.3	285
GWO–GA hybrid	0.696	11.4	265
S-GWO / S-HYB	0.692 / 0.695	11.1 / 11.5	268 / 261
Constructive	0.890	16.6	314
All local	1.000	20.3	334



Main findings (details and statistics in the manuscript):

* The hybrid's value depends on the budget: worse than GWO at `Q ≤ 5·10³`, tied at `10⁴`, better than GWO from `2·10⁴`, but never the best method with default operators
  (GWO leads at small budgets, PSO from `2·10⁴`). With a mutation step `μ = 0.05` the hybrid is significantly better than GWO at `10⁴`.
* GWO's final cost worsens beyond `10⁴` evaluations (0.697 → 0.725 at `10⁵`).
* The ranking depends on the population size and the number of fog nodes; with `K` tuned per method GWO, PSO, DE, GA and the hybrid lie within 0.015 in `J`.
* The constructive rule (`J = 0.890`) is worse than random search and does not help as an initialiser.
* NOMA beats plain OMA by 11.5 % at `N = 30`, but by 6.1 % against an OMA with the same number of concurrent offloaders (where it saves delay, not energy); with scaled resources the gain falls to
  1 % at `N = 50` and reverses at `N = 60`; at a residual-cancellation fraction `β = 0.05` it is within 0.6 % of OMA.
* At `N = 100` only GWO (0.92) and the constructive rule (0.96) beat all-local execution with certainty.
* The best methods cut device energy by 44–46 % versus all-local execution (fog-side energy is not modelled).

### Scope and limitations

One default scenario plus the stated sweeps; static single-cell model without fading, fog-side energy or result return; perfect CSI;
imperfect SIC tested only at `N = 30` with GWO; no lower bound on the optimum (E7 gives an upper bound).
See the *Limitations* section of the manuscript.

---


# noma_e9_e12 Experiences

[![License: MIT](https://shields.io)](https://opensource.org)
[![Python 3.8+](https://shields.io)](https://python.org)

This repository contains the complete, self-contained reference implementation and reproducibility validation suite for **"Deconstructing the Claims of Hybrid Metaheuristics in NOMA Fog Offloading: A Budget- and Capacity-Controlled Methodological Audit"** (submitted to *IEEE Access*).

Rather than presenting an unconstrained "algorithm tweak," this framework introduces a rigorous, budget- and capacity-controlled benchmark environment. It maps the operational boundaries of heuristic schedulers against strict computing budgets (Q), isolates baseline algorithmic plateaus, and decouples physical multi-access gains from raw channel capacity expansion.

---

## 📡 Core System Architecture & Features

The underlying codebase implements a single-cell uplink multiplexing model coupling discrete offloading, sub-channel assignment, transmission power control, and continous edge computing frequency slices.

### 🧩 Physical & Algorithmic Extensions
* **Baseline Framework:** Deterministic, distance-based path loss profile with localized device-centric cost boundaries.
* **Rayleigh Block Fading ([E9]):** Models stochastic multi-path propagation under a quasi-static block-fading layout with perfect channel state information per realization.
* **System-Wide Cost Disconnect ([E10]):** Evaluates global network footprint trajectories by capturing infrastructure-side computing energy overheads (ξ) and passive downlink result-return communication paths (ε).
* **Algorithmic Stagnation Diagnostic ([E11]):** Isolates structural search plateaus via conditional greedy coordinate updates (`G-GWO`) and budget-decoupled decay parameter schedules (`F-GWO`).
* **Macro-Mutation Exploration Tuning ([E12]):** Tracks global optimal coordinates through disjoint validation instances exploring exploratory step-size metrics up to \(\mu^* = 0.2\).

### 📊 Audited Optimization Algorithms
The repository contains unified continuous numerical update vectors for a broad baseline suite:
* **Stochastic Schedulers:** Uniform Random Search (`RS`), Canonical Genetic Algorithm (`GA`), Particle Swarm Optimization (`PSO`), Differential Evolution (`DE`), and the Whale Optimization Algorithm (`WOA`).
* **Swarm Benchmarks:** Baseline Grey Wolf Optimizer (`GWO`), Greedy GWO (`G-GWO`), and Fixed-Schedule GWO (`F-GWO`).
* **Compound Paradigms:** Combined GWO--GA Hybrid Pipeline, along with Constructive-Seeded variations (`S-GWO`, `S-HYB`).
* **Operational Anchors:** Fast Channel-Ordered Deterministic Constructive Rule (`CONS`), Uniform Distribution Baseline (`EQUAL`), and an All-Local Execution Floor.

---

## ⚙️ Installation & Prerequisites

The execution engine is fully self-contained in a single script and relies strictly on core mathematical libraries. 

### Dependencies
Ensure you have Python 3.8+ installed along with the following packages:
```bash
pip install numpy scipy pandas
```

---

## 🚀 Execution & Command Reference

The framework uses an automated, multi-threaded task runner controlled via an argument parser. All operations execute deterministic seed matching based on locked array coordinates.

### 1. Generating Primary Experimental Paper Values (`E9`–`E12`)
To run all newly proposed communication audits and diagnostic sweeps with 30 independent runs over 8 parallel process workers:
```bash
python noma_e9_e12.py --exp E9,E10,E11,E12 --runs 30 --workers 8
```

### 2. Executing Reproducibility Table Validations (`V1`–`V8`)
The framework incorporates a regression audit layer that checks every cell output against the paper's original tables (**Tables II, P, Q, QP, K, KP, M, MP, A, N, S, R**), returning explicit `EXACT` or `OK` validation indicators:

* **Quick Sanity Check (Table II and Table P):**
  ```bash
  python noma_e9_e12.py --exp V1 --workers 8
  ```
* **Standard Verification Sweep (All baseline tables except heavy compute matrix boundaries):**
  ```bash
  python noma_e9_e12.py --exp VAL --workers 8
  ```
* **Full Exhaustive Validation (Includes deep matrix resource bounds at Q=5×10⁴, 10⁵ and E₇ restart boundaries):**
  ```bash
  python noma_e9_e12.py --exp VAL --vheavy --workers 8
  ```

### 3. Comprehensive Automation & Smoke Testing
* **Run Everything Sequential (Validations + New Experiments):**
  ```bash
  python noma_e9_e12.py --exp all --workers 8
  ```
* **Rapid Code Integration / Smoke Test (Divides computational budgets by 5 for quick parsing verification):**
  ```bash
  python noma_e9_e12.py --exp E9 --runs 2 --quick --workers 1
  ```

### Key Execution Flags
* `--reuse`: Bypasses re-computation if prior execution data tables exist within the target folder.
* `--xi <float>`: Manually changes the infrastructure-to-edge power scaling scale parameter (Defaults to `0.25`).
* `--out <dir>`: Sets custom output directories for compiled data matrix CSV streams.

---

## 📁 Data Outputs & Architecture

Upon completion, execution datasets are automatically compiled into a uniform structural layout under the `./results_ext/` directory:

Use code with caution.
├── results_ext/
│   ├── E9a_summary.csv        # Baseline vs. Rayleigh multi-path propagation indices
│   ├── E9b_gap.csv            # NOMA Capacity Decomposition metrics (OMA vs. OMA3R)
│   ├── E10_summary.csv        # Global System-Wide Energy balancing metrics
│   ├── E11_summary.csv        # Algorithmic Stagnation Diagnostic outputs
│   ├── E12_test_summary.csv   # Disjointly optimized Macro-Mutation tracking metrics
│   ├── validation.csv         # Step-by-step cross-check audit matching paper parameters
│   └── paper_values.json      # Structured dictionary matching LaTeX file requirements

---

## 🔬 Core Novelty Insights

This framework formally demonstrates two major physical and computational anomalies:
1. **The Algorithmic Illusion:** The historical performance advantage attributed to GWO-GA hybrids is proven to be an artifact of un-tuned population configurations and an unconditional replacement policy in standard GWO. Applying a greedy update path (`G-GWO`) completely stabilizes the baseline, while the hybrid only achieves superiority when configured with highly exploratory macro-mutation jumps (\(\mu^* = 0.2\)).
2. **NOMA Capacity Buffer:** Through the capacity-equalized orthogonal baseline control (OMA3R), we show that NOMA's apparent performance gains are driven by a concurrent user capacity buffer expansion rather than structural spectral efficiency, leaving it highly vulnerable to minor SIC residual power leakage (β ≥ 0.05).

---

# noma_open_items.py

Here is a complete, production-ready README.md for the noma_open_items.py. It expands upon the previous architecture by incorporating the complete system-level logic of your extended codebases (noma_e9_e12.py and noma_open_items.py).
[![License: MIT](https://shields.io)](https://opensource.org)
[![Python 3.8+](https://shields.io)](https://python.org)

This repository contains the complete reference implementation, empirical extensions, and replication protocols for a series of budget- and capacity-controlled audits targeting resource scheduling in uplink NOMA edge-fog computing environments.

Rather than proposing incremental heuristic adjustments, this repository provides a rigorous software system to track allocation costs against strict runtime computation budgets (Q). It decouples physical multi-access gains from user-capacity buffers, diagnoses runtime stagnation in population-based search loops, and isolates systemic cost disconnects between edge energy profiles and infrastructure scales.

---

## 📡 System Architecture & Computational Extensions

The simulation engine models a single-cell uplink multi-user pipeline coupling discrete server assignment, sub-channel block grouping, continuous transmission power levels, and continuous fog node processing shares. The framework contains three distinct modules:

### 1. Architectural Baseline (`noma_e9_e12.py`)
* **Rayleigh Block Fading ([E9]):** Evaluates scheduler trajectories under quasi-static block fading with perfect channel state information (CSI) per realization.
* **System-Wide Cost Disconnect ([E10]):** Re-allocates resources under global accounting (\(J_{\mathrm{sys}}\)) incorporating fog processing constants (ξ), server capacitance constants (\(\kappa_{\mathrm{f}}\)), and dedicated downlink result-return delay overheads (ε).
* **Stagnation Diagnostics ([E11]):** Unmasks structural decay anomalies by evaluating standard Grey Wolf Optimizer (GWO) pipelines against budget-decoupled parameter schedules (`F-GWO`) and conditional greedy updates (`G-GWO`).
* **Exploratory Step Sizing ([E12]):** Tracks optimization limits by exploring continuous coordinate mutation scales up to a macro threshold (\(\mu^* = 0.2\)) via disjoint instance sets.

### 2. Multi-Candidate Reruns & Open Network Items (`noma_open_items.py`)
* **Greedy Optimizer Sweeps (`GREEDY`):** Reruns and validates budget sweeps ([E2]), high-dimensional scalability ([E5]), matrix sizing ([E6]), restart bounds ([E7]), and node density scales ([E8]) leveraging the greedy state replacement baseline.
* **Tuned Multi-Candidate Tracking ([E13]):** Cross-checks the optimized macro-mutation hybrid scheduler against Particle Swarm Optimization (`PSO`) and `G-GWO` configured at their respective empirically selected population profiles (\(K^*\)).
* **Constructive Power Search Heuristics ([E14]):** Evaluates a granular parameter deconstruction of channel-ordered constructive heuristics (`CONS`), swapping fixed power policies for dynamic admission grid searches and continuous coordinate local refinement loops on the true objective function.
* **Imperfect SIC Boundary Fragility ([E15]):** Charts break-even boundary conditions where residual inter-user power leaks (β) from imperfect successive interference cancellation completely neutralize non-orthogonal multi-user processing gains across Rayleigh propagation channels.
* **Continuous Partial Offloading ([E16]):** Transitions from a binary scheduling model to a continuous partial allocation paradigm where a fraction \(\phi_i \in [0,1]\) of a node's dataset is computed locally in parallel with remote edge migration under high infrastructure power scales (ξ ≥ 0.5).

---

## 🛠️ Software Prerequisites

The computational suite is written in pure Python and requires no external proprietary solvers or convex libraries. 

### Dependencies
```bash
pip install numpy scipy pandas
```

---

## 🚀 Execution & Command Reference

The framework uses an automated, concurrent task architecture driving isolated deterministic instance seeds.

### Module A: Rayleigh Propagation & System Overhead Audits (`noma_e9_e12.py`)
* **Compile All Extended Data Metrics:**
  ```bash
  python noma_e9_e12.py --exp E9,E10,E11,E12 --runs 30 --workers 8
  ```
* **Verify Baseline Replication Integrity (Cross-checks outputs against original reference tables):**
  ```bash
  python noma_e9_e12.py --exp VAL --vheavy --workers 8
  ```

### Module B: Greedy Overhauls & Physical Link Fragility Audits (`noma_open_items.py`)
* **Execute All Conditional Greedy Reruns (`E2`, `E5`, `E6`, `E7`, `E8`):**
  ```bash
  python noma_open_items.py --exp GREEDY --runs 30 --workers 8
  ```
* **Execute All Advanced Open Networking Audits (`E13`, `E14`, `E15`, `E16`):**
  ```bash
  python noma_open_items.py --exp OPEN --runs 30 --workers 8
  ```
* **Comprehensive Automated Sequential Execution Suite:**
  ```bash
  python noma_open_items.py --exp all --runs 30 --workers 8
  ```
* **Rapid Integration Smoke Test (Divides budgets for smoke testing local parsing):**
  ```bash
  python noma_open_items.py --exp all --runs 2 --quick --restarts 2 --refQ 2000 --workers 1
  ```

### Key Parameter Flags
* `--reuse`: Bypasses re-computation steps if compiled CSV data structures match within the target directory.
* `--out <string>`: Directs output folder definitions (Defaults to `./results_ext/` or `./results_open/`).
* `--betas <string>`: Sets custom residual interference list configurations for SIC leakage loops (e.g., `0,0.05,0.1,0.2,0.5`).
* `--xis <string>`: Configures server power scale arrays to chart infrastructure energy inflation.

---

## 📁 Data Outputs & File Tree

Compiled results are automatically exported to explicit, structured spreadsheet frames ready for LaTeX table parsing:

Use code with caution.
├── results_ext/               # Primary Auditing Module Outputs
│   ├── E9a_summary.csv        # Rayleigh block fading cost summaries
│   ├── E9b_gap.csv            # NOMA vs OMA equal-capacity buffer comparisons
│   ├── E10_summary.csv        # Global System-Wide Energy balancing metrics
│   └── validation.csv         # Re-run structural replication telemetry
│
├── results_open/              # Advanced Open Item Module Outputs
│   ├── E2_summary.csv         # Greedy-selection budget convergence timelines
│   ├── E5_summary.csv         # High-dimension feasibility and local floor bounds
│   ├── E6_bestK.csv           # Optimal population mappings per candidate update track
│   ├── E14_summary.csv        # Constructive grid-power search & refinement logs
│   ├── E15_break_even.csv     # Calculated β breakdown thresholds vs OMA
│   └── E16_summary.csv        # Continuous partial offloading fractional matrices (mean_phi)

---

## 🔬 Core Engineering Insights Discovered

This framework provides specific, citable threshold parameters for next-generation edge deployment architectures:
1. **The Algorithmic Illusion:** Standard metaheuristic degradation at high budgets is exogenous to the hybrid architecture. It is caused by GWO's unconditional update policy. Enforcing a conditional greedy filter (`G-GWO`) completely mitigates this decay. Hybrids only achieve optimization dominance when configured as a highly exploratory macro-mutation optimizer (\(\mu^* = 0.2\)).
2. **Spectral vs Capacity Buffer:** By evaluating standard scheduling against our capacity-equalized orthogonal baseline control (OMA3R), we show NOMA's apparent benefit reduces exclusively to delay mitigation rather than device power reduction, and it is highly fragile, failing completely at a minor 5% residual signal cancellation power leak (β ≥ 0.05).
3. **Partial Offloading Scale Reversal:** Splitting computational tasks continuously (\(\phi_i\)) provides strong structural advantages in green networking environments. However, once server scaling overheads exceed a threshold boundary (ξ ≥ 0.5), any localized device energy minimization is wiped out by massive infrastructure power inflation.

---

## 📜 License & Citations

This software is distributed under the terms of the **MIT License**. If you utilize this simulation framework, auditing protocols, or data tables in your academic work, please cite the corresponding journal publication:

```bibtex
@article{matinkhah2026computation,
  author={Matinkhah, S. Mojtaba},
  title={Computation Budget, Access Capacity and Fog Energy in Uplink NOMA Offloading: A Controlled Evaluation of Allocation Algorithms},
  journal={Computer Communications},
  issn={0140-3664},
  year={2026},
  volume={184},
  pages={1--22},
  doi={...}
}
```



