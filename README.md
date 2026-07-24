# Gradient-Aware RL-Based DVFS Governor for Batteryless Intermittent IoT Edge Nodes

This repository contains the complete implementation of a **Gradient-Aware Reinforcement Learning (RL) DVFS Governor** tailored for intermittent, energy-harvesting IoT nodes.

---

## 🏗️ Project Architecture & Component Breakdown

```
.
├── Predictive_RL_DVFS_Research_Paper.md   # Complete IEEE research paper
├── IEEE_Predictive_RL_DVFS_Research_Paper.html # Standalone HTML paper with base64 plots
├── README.md                               # Project documentation & reproduction commands
├── requirements.txt                        # Dependency constraints
├── firmware/                               # Bare-metal / FreeRTOS ARM Cortex-M4 C & ELF Firmware
│   ├── main.c                              # STM32F4 USART1 & DVFS clock scaling C firmware
│   ├── build_elf.py                        # Pure Python ARM Cortex-M4 ELF binary generator
│   └── firmware.elf                        # 32-bit ARM Cortex-M4 ELF binary loaded into Renode
├── models/
│   ├── ppo_dvfs_model.zip                 # Primary trained PPO model
│   ├── ppo_dvfs_seed_0.zip...seed_4.zip   # 5-seed trained PPO policy archives
│   └── dqn_dvfs_model.zip                 # Trained DQN model archive
├── src/
│   ├── environment.py                      # POMDP Gymnasium physics environment (E = 0.5*C*V^2)
│   ├── train.py                            # Multi-seed PPO & DQN training pipeline
│   ├── evaluate_and_plot.py                # Deterministic evaluation & Wilcoxon test script
│   ├── baselines.py                        # Always-Max, Powersave, Static Threshold governors
│   └── export_html_with_embedded_images.py # Base64 HTML exporter
├── renode/
│   ├── stm32f4_dvfs.repl                   # Renode ARM Cortex-M4 platform definition
│   ├── stm32f4_dvfs.resc                   # Renode script with sysbus LoadELF @firmware/firmware.elf
│   ├── renode_server.py                    # Renode socket protocol emulator server
│   └── arm_cortex_m4_co_sim.py             # Client bridge executing hardware trajectory
└── results/
    ├── benchmark_performance_comparison.png# 300 DPI Multi-panel publication plot
    ├── benchmark_performance_comparison.pdf# Vector PDF plot
    ├── benchmark_raw_results.csv           # Raw per-seed primary trial dataset
    ├── sensitivity_raw_results.csv         # Raw multi-profile trial dataset
    ├── capacitance_sweep_raw_results.csv   # Raw capacitance sweep trial dataset
    └── renode_cosim_telemetry.json         # 150-step co-simulation telemetry log
```

---

## 🛠️ Installation & Execution Guide

### 1. Environment Setup & Renode Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Install official Renode 1.16.0 binary via winget (Windows):
winget install --id Renode.Renode --accept-source-agreements --accept-package-agreements --silent
```

### 2. Firmware ELF Binary Compilation (`firmware/build_elf.py`)
To generate/rebuild the 32-bit ARM Cortex-M4 ELF executable binary (`firmware/firmware.elf`):
```bash
python firmware/build_elf.py
```

### 3. Model Training (`src/train.py`)
To train PPO across 5 independent seeds (`seed=0, 1, 2, 3, 4`) and DQN on the energy-conserving physics environment:
```bash
python src/train.py
```

### 4. Quantitative Evaluation & Benchmarking (`src/evaluate_and_plot.py`)
To run 30 Monte Carlo evaluation trials across 5 governors, execute the Wilcoxon signed-rank test (`scipy.stats.wilcoxon`), run the multi-profile sensitivity analysis, and perform the supercapacitor capacitance sweep ($5\text{ mF}, 10\text{ mF}, 30\text{ mF}, 50\text{ mF}$):
```bash
python src/evaluate_and_plot.py
```

### 5. Official Renode Hardware Co-Simulation (`renode/arm_cortex_m4_co_sim.py`)
To launch official installed **Renode v1.16.0 (`Renode.exe`)**, ingest `renode/stm32f4_dvfs.resc`, load `firmware/firmware.elf` via `sysbus LoadELF`, create the ARM Cortex-M4 target platform, and stream 150 steps of frequency scaling payloads over line-buffered TCP sockets (`port 4000`):
```bash
python renode/arm_cortex_m4_co_sim.py
```

---

## ⚡ Physics Equations & Energy Conservation

Energy integration across discrete control steps ($\Delta t = 0.1\text{ s}$) enforces strict energy conservation ($E = \frac{1}{2} C_{\text{supercap}} V_{\text{cap}}^2$), accounting for core CMOS dissipation $P_{\text{total}}$ and internal ESR $I^2 R$ heat loss ($P_{\text{esr\_loss}} = I_{\text{load}}^2 R_{\text{esr}}$):

$$I_{\text{load}}(t) = \frac{P_{\text{total}}(f, V_{\text{dd}})}{\max(1.0, V_{\text{cap}}(t))}$$

$$P_{\text{esr\_loss}}(t) = I_{\text{load}}^2(t) \cdot R_{\text{esr}}$$

$$P_{\text{total\_drain}}(t) = P_{\text{total}}(f, V_{\text{dd}}) + P_{\text{esr\_loss}}(t)$$

$$\Delta E(t) = \left[ P_{\text{harvested}}(t) - P_{\text{total\_drain}}(t) \right] \cdot \Delta t$$

$$E(t+1) = \max\left(0.0, \frac{1}{2} C_{\text{supercap}} V_{\text{cap}}^2(t) + \Delta E(t)\right)$$

$$V_{\text{cap}}(t+1) = \min\left(V_{\text{max}}, \sqrt{\frac{2 E(t+1)}{C_{\text{supercap}}}}\right)$$
