# Predictive RL-Based DVFS Governor for Batteryless Intermittent IoT Edge Nodes

This repository contains the official implementation of the **Predictive Reinforcement Learning (RL) DVFS Governor** for batteryless, energy-harvesting IoT nodes.

---

## 🛠️ Architecture & Setup Instructions

### 1. Installation
Ensure Python 3.10+ is installed. Activate your virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Training Models (`src/train.py`)
To train the Proximal Policy Optimization (PPO) and Deep Q-Network (DQN) models from scratch on the energy-conserving physics environment:
```bash
python src/train.py
```
This trains PPO and DQN over 61,440 timesteps with seed initialization (`seed=0`) and saves PyTorch models to `models/ppo_dvfs_model.zip` and `models/dqn_dvfs_model.zip`.

### 3. Evaluation & Benchmarking (`src/evaluate_and_plot.py`)
To evaluate all 5 governors (Always-Max, Powersave, Static Threshold, PPO RL, DQN RL) across 30 held-out test seeds (`seed=100...129`) under deterministic policy inference (`deterministic=True`):
```bash
python src/evaluate_and_plot.py
```
This will:
1. Export raw evaluation trial data to `results/benchmark_raw_results.csv`.
2. Compute the **Wilcoxon Signed-Rank Test** for statistical significance testing.
3. Perform **Multi-Profile Sensitivity Analysis** across 3 solar profiles (`standard_cloudy`, `volatile`, `clear_day`).
4. Save 300 DPI publication plots to `results/benchmark_performance_comparison.png` and `results/benchmark_performance_comparison.pdf`.

---

## ⚡ Energy-Conserving Physics Differential Math

Energy integration across discrete control steps ($\Delta t = 0.1\text{ s}$) enforces strict energy conservation ($E = \frac{1}{2} C_{\text{supercap}} V^2$):

$$\Delta E(t) = \left[ P_{\text{harvested}}(t) - P_{\text{total}}(f, V_{\text{dd}}) \right] \cdot \Delta t$$

$$E(t+1) = \max\left(0.0, \frac{1}{2} C_{\text{supercap}} V_{\text{cap}}^2(t) + \Delta E(t)\right)$$

$$V_{\text{cap}}(t+1) = \min\left(V_{\text{max}}, \sqrt{\frac{2 E(t+1)}{C_{\text{supercap}}}}\right)$$

Accounting for internal series resistance ($R_{\text{esr}}$), terminal voltage under load current ($I_{\text{load}} = P_{\text{total}} / V_{\text{cap}}$) is evaluated after energy integration:
$$V_{\text{terminal}}(t) = \max\left(0.0, V_{\text{cap}}(t) - I_{\text{load}}(t) \cdot R_{\text{esr}}\right)$$
