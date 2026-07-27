# RL-Based Dynamic Voltage and Frequency Scaling (DVFS) Governor for Batteryless Intermittent IoT Edge Nodes

A research-oriented implementation of a **reinforcement learning (RL) driven DVFS governor** designed for **energy-harvesting, batteryless intermittent IoT systems**.  
The project integrates simulation, training, baseline benchmarking, and Renode-based hardware co-simulation for ARM Cortex-M4-class targets.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Technical Highlights](#technical-highlights)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation and Benchmarking](#evaluation-and-benchmarking)
- [Renode Hardware Co-Simulation](#renode-hardware-co-simulation)
- [Artifacts](#artifacts)
- [License](#license)

## Overview
Batteryless edge nodes powered by harvested ambient energy face highly variable power availability and frequent brownouts. This repository explores adaptive control of voltage/frequency operating points using RL so the system can maximize useful work while respecting strict energy constraints.

The implementation includes:
- A custom environment modeling supercapacitor-backed intermittent operation.
- RL training pipelines (PPO and DQN variants).
- Deterministic and Monte Carlo evaluation utilities.
- Comparison against conventional DVFS baseline policies.
- Renode-driven firmware co-simulation for architecture-faithful validation.

## Repository Structure
```text
.
├── firmware/
│   ├── main.c
│   ├── build_elf.py
│   └── firmware.elf
├── models/
│   ├── ppo_dvfs_model.zip
│   ├── ppo_dvfs_seed_0.zip ... ppo_dvfs_seed_4.zip
│   └── dqn_dvfs_model.zip
├── src/
│   ├── environment.py
│   ├── train.py
│   ├── evaluate_and_plot.py
│   ├── baselines.py
│   └── export_html_with_embedded_images.py
├── renode/
│   ├── stm32f4_dvfs.repl
│   ├── stm32f4_dvfs.resc
│   ├── renode_server.py
│   └── arm_cortex_m4_co_sim.py
├── results/
│   ├── benchmark_performance_comparison.png
│   ├── benchmark_performance_comparison.pdf
│   ├── benchmark_raw_results.csv
│   ├── sensitivity_raw_results.csv
│   ├── capacitance_sweep_raw_results.csv
│   └── renode_cosim_telemetry.json
├── Predictive_RL_DVFS_Research_Paper.md
├── Predictive_RL_DVFS_Research_Paper.pdf
├── IEEE_Predictive_RL_DVFS_Research_Paper.html
├── requirements.txt
└── README.md
```

## Technical Highlights
- **Energy-aware control objective** under intermittent harvested power.
- **Physics-grounded environment** with capacitor energy dynamics.
- **Multiple policy families** (PPO and DQN) and seed-wise reproducibility.
- **Baseline governors** (e.g., fixed/max/powersave/threshold-style) for fair comparison.
- **Statistical evaluation** including significance testing and sensitivity studies.
- **Firmware + Renode loop** enabling software/hardware co-simulation workflows.

## Installation
### Prerequisites
- Python 3.10+ recommended
- `pip`
- Renode (for co-simulation path)

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
# Windows (PowerShell): .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Renode (Windows example)
```bash
winget install --id Renode.Renode --accept-source-agreements --accept-package-agreements --silent
```

## Quick Start
### 1) Build firmware ELF (if regeneration is needed)
```bash
python firmware/build_elf.py
```

### 2) Train RL models
```bash
python src/train.py
```

### 3) Evaluate and generate plots/results
```bash
python src/evaluate_and_plot.py
```

## Training
`src/train.py` orchestrates RL training runs across configured seeds and algorithm variants.

Expected outputs include model checkpoints in `models/`, such as:
- `ppo_dvfs_model.zip`
- `ppo_dvfs_seed_*.zip`
- `dqn_dvfs_model.zip`

## Evaluation and Benchmarking
`src/evaluate_and_plot.py` performs policy evaluation against baseline governors and produces:
- Aggregate comparisons (`results/benchmark_performance_comparison.*`)
- Raw trial data (`results/*_raw_results.csv`)
- Additional sensitivity/capacitance analyses.

## Renode Hardware Co-Simulation
For hardware-aware validation:
1. Use `renode/stm32f4_dvfs.repl` and `renode/stm32f4_dvfs.resc` to define/load the platform.
2. Ensure firmware ELF exists at `firmware/firmware.elf`.
3. Run:
```bash
python renode/arm_cortex_m4_co_sim.py
```

This flow enables trajectory exchange and telemetry generation (e.g., `results/renode_cosim_telemetry.json`).

## Artifacts
The repository includes publication-oriented artifacts:
- Research manuscript in Markdown/PDF/HTML formats.
- Plots and benchmark summaries under `results/`.
- Pretrained model archives under `models/`.

## License
This project is distributed under the **MIT License**. See [LICENSE](LICENSE) for details.
