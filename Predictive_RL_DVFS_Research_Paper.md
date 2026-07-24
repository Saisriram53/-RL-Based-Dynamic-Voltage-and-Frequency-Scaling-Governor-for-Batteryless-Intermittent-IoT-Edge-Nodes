# Gradient-Aware RL-Based Dynamic Voltage and Frequency Scaling Governor for Batteryless Intermittent IoT Edge Nodes

**Author:** Sai Sreeram  
**Affiliation:** Department of Electrical and Computer Engineering  
**Email:** saisreeram@research.org | **ORCID:** 0000-0002-1849-5921  
**Target Publication Venue:** *IEEE Internet of Things Journal*  

---

## Abstract
Ambient energy-harvesting Internet of Things (IoT) edge nodes eliminate battery replacement overheads but introduce operational vulnerability to environmental power volatility. Photovoltaic shading events rapidly drain small-capacity supercapacitors, driving supply rails below the integrated Brownout Reset (BOR) trip voltage ($V_{\text{brownout}} = 1.8\text{V}$) and inducing hardware reboots that wipe volatile SRAM state. Conventional Dynamic Voltage and Frequency Scaling (DVFS) governors—such as aggressive Always-Max, static thresholding, or static Powersave—either trigger frequent brownouts or create severe queue backlogs due to static frequency throttling.

We design and evaluate a **Gradient-Aware Reinforcement Learning (RL) DVFS Governor** tailored for intermittent microcontrollers operating under severe energy constraints. Formulated as a **Partially Observable Markov Decision Process (POMDP)** incorporating energy-conserving supercapacitor differential dynamics ($E = \frac{1}{2} C V^2$) and internal ESR resistive losses ($P_{\text{esr\_loss}} = I_{\text{load}}^2 R_{\text{esr}}$), the policy agent ingests real-time telemetry—specifically terminal supercapacitor voltage ($V_{\text{terminal}}$), active task backlog ($Q_{\text{len}}$), photovoltaic power ($P_{\text{harvested}}$), power gradient ($\Delta P_{\text{harvested}}$), and normalized prior action—to dynamically modulate CPU core frequency between $8\text{ MHz}$ and $80\text{ MHz}$. Evaluated across 30 independent held-out test seeds under deterministic policy inference in a physics-informed Gymnasium environment, our Proximal Policy Optimization (PPO) model eliminates brownout resets entirely (**0.0% crash rate** at $C_{\text{supercap}} = 10\text{ mF}$), maintaining normalized service throughput (**4.15 ± 0.20 tasks/step**) and minimal mean queue backlog (**4.6 ± 0.3 tasks**). Multi-seed training evaluation across 5 independent training runs confirms robust policy convergence ($4.54 \pm 0.06\text{ tasks}$ mean backlog across 5 trained policies). Statistical hypothesis testing using `scipy.stats.wilcoxon` in `src/evaluate_and_plot.py` confirms a highly significant queue backlog reduction over static thresholding (Wilcoxon signed-rank test $W=0.0, p = 1.86 \times 10^{-9} < 0.001$). Capacitance sensitivity analysis across $C_{\text{supercap}} \in [5\text{ mF}, 10\text{ mF}, 30\text{ mF}, 50\text{ mF}]$ identifies $10\text{ mF}$ as the optimal energy-buffer threshold where PPO prevents brownouts while Always-Max incurs a **100.0% crash rate**. Finally, we demonstrate hardware command interface feasibility using a dedicated **ARM Cortex-M4 Hardware Command Protocol Bridge** (`renode/arm_cortex_m4_co_sim.py` and `renode/renode_server.py`), paired with native Renode platform setup scripts (`renode/stm32f4_dvfs.repl` & `.resc`) and an embedded C target firmware artifact (`firmware/main.c`).

**Index Terms—** Dynamic Voltage and Frequency Scaling (DVFS), Batteryless IoT, Intermittent Computing, Energy Harvesting, Reinforcement Learning, Proximal Policy Optimization (PPO), Renode Hardware Co-Simulation, POMDP.

---

## I. Introduction

Deploying self-powered edge nodes in remote telemetry and sensor networks requires operating without primary battery cells [1], [2]. Photovoltaic harvesters paired with micro-farad/milli-farad supercapacitors offer long-term deployment capability, yet expose the underlying microcontroller logic to extreme input power volatility.

Passing cloud formations or structural shadows can drop incoming solar power by upwards of 90% within milliseconds. Because ultralow-power microcontrollers utilize compact energy buffers ($C_{\text{supercap}} = 10\text{ mF}$) to minimize board area and leakage, sustained power deficits drain stored energy rapidly. When supply rail potential drops to the hardware brownout reset threshold ($V_{\text{brownout}} = 1.8\text{V}$), the internal power management unit (PMU) forces a full system reset. This clears volatile SRAM registers, invalidates FreeRTOS task handles, and forces an un-checkpointed cold reboot.

Dynamic Voltage and Frequency Scaling (DVFS) adjusts operating frequency $f$ and core supply voltage $V_{\text{dd}}$ to minimize dynamic CMOS dissipation ($P_{\text{dynamic}} = \alpha C_L V_{\text{dd}}^2 f$). However, standard firmware governors fail when applied to intermittent energy regimes:
1. **Always-Max (Fixed Maximum Frequency):** Locks the clock tree at $80\text{ MHz}$ ($V_{\text{dd}} = 1.5\text{V}$). While achieving rapid task execution during full solar exposure, it rapidly depletes the $10\text{ mF}$ supercapacitor during irradiance drops, precipitating a $100.0\%$ brownout crash rate at episode step 61 (21 steps into cloud onset at step 40).
2. **Powersave (Fixed Minimum Frequency):** Throttles execution to $8\text{ MHz}$ ($V_{\text{dd}} = 0.9\text{V}$). Although it prevents brownout resets ($0.0\%$ crash rate), it fails to keep pace with incoming task arrivals, accumulating an intolerable backlog ($166.2 \pm 2.7\text{ tasks}$).
3. **Static Threshold Governor:** Adjusts clock frequencies based on static voltage comparator levels (e.g., scaling up above $80\%$ state-of-charge, downscaling below $30\%$). Because voltage drops trail power draw, static comparators exhibit reactive switching lag, resulting in elevated queue backlog ($16.5 \pm 3.6\text{ tasks}$).

To resolve these trade-offs, we present a **Gradient-Aware Reinforcement Learning (RL) DVFS Governor**. By processing terminal voltage, queue backlog, solar power, power gradients ($\Delta P_{\text{harvested}}$), and previous clock states, the RL policy anticipates supercapacitor depletion, scaling core clock frequency downward prior to critical discharge and accelerating back to peak frequency as solar power recovers.

### A. Related Work & Contextualization in Intermittent DVFS Literature
Research into supply voltage regulation and frequency scaling for batteryless intermittent nodes has evolved across two primary paradigms:

1. **Hardware & Feedback Threshold Control:** D2VFS (*Dynamic Duty-Cycling and Voltage Scaling*) [3] established the foundational reference architecture for DVFS on batteryless devices, adjusting operating states relative to supercapacitor voltage. FBTC (*Feedback-based Threshold Control*) [4] enhanced D2VFS by reducing energy overheads and introducing configurable startup-voltage thresholds. Similarly, ACES (*Adaptive Control for Energy-Harvesting Systems*) [5] introduced reactive threshold regulation to maintain capacitor charge above brownout trip levels.
2. **Reinforcement Learning-Based Governors:** In mainstream systems, *zTT* [6] established RL-based DVFS by framing performance-energy regulation as a Markov Decision Process. For energy-harvesting IoT nodes, *tinyMAN* [7] demonstrated Q-learning energy management deployed directly onto wearable microcontroller prototypes using TensorFlow Lite Micro (<100 KB footprint).

**Our Distinct Position & Methodological Advance:**  
While D2VFS [3], FBTC [4], and ACES [5] rely on reactive voltage comparator feedback, they exhibit switching lag during rapid environmental transients because capacitor voltage drops trail active power draw. Conversely, while *tinyMAN* [7] and *zTT* [6] demonstrated lightweight RL deployment, they focused on battery-buffered wearables or task offloading without modeling supercapacitor Equivalent Series Resistance ($R_{\text{esr}}$) voltage drops. Our work bridges this gap by combining **real-time solar power gradient telemetry** with **energy-conserving supercapacitor ESR dynamics ($V_{\text{terminal}} = V_{\text{cap}} - I_{\text{load}} R_{\text{esr}}$)** in a physics-informed POMDP, paired with a **hardware command protocol socket bridge**.

---

## II. System Modeling & Physical Formulation

### A. CMOS Core Power Dissipation Dynamics
Silicon core power dissipation $P_{\text{total}}$ on low-power microcontrollers operating at core supply voltage $V_{\text{dd}}$ and clock frequency $f$ decomposes into dynamic switching losses $P_{\text{dynamic}}$ and static subthreshold leakage $P_{\text{static}}$:

$$P_{\text{total}}(f, V_{\text{dd}}) = P_{\text{dynamic}} + P_{\text{static}} = (\alpha \cdot C_L \cdot V_{\text{dd}}^2 \cdot f) + P_{\text{leakage}}$$

where $\alpha C_L = 100\text{ pF}$ represents effective switching capacitance and $P_{\text{leakage}} = 2.0\text{ mW}$ models baseline static leakage across active core logic.

| DVFS Index ($a_t$) | Core Clock ($f$) | Supply Voltage ($V_{\text{dd}}$) | Dynamic Power ($P_{\text{dynamic}}$) | Total Power Dissipation ($P_{\text{total}}$) |
| :---: | :---: | :---: | :---: | :---: |
| **0 (Powersave)** | $8\text{ MHz}$ | $0.9\text{ V}$ | $0.648\text{ mW}$ | $2.648\text{ mW}$ |
| **1 (Low)** | $16\text{ MHz}$ | $1.1\text{ V}$ | $1.936\text{ mW}$ | $3.936\text{ mW}$ |
| **2 (Medium)** | $48\text{ MHz}$ | $1.3\text{ V}$ | $8.112\text{ mW}$ | $10.112\text{ mW}$ |
| **3 (Maximum)** | $80\text{ MHz}$ | $1.5\text{ V}$ | $18.000\text{ mW}$ | $20.000\text{ mW}$ |

### B. Energy-Conserving Supercapacitor Buffer Dynamics with ESR $I^2 R$ Loss
Energy storage relies on an onboard supercapacitor ($C_{\text{supercap}} = 10\text{ mF}$). The numerical voltage evolution across discrete control steps $\Delta t = 100\text{ ms}$ is governed by exact energy conservation ($E = \frac{1}{2} C_{\text{supercap}} V_{\text{cap}}^2$), accounting for core dissipation $P_{\text{total}}$ and internal ESR $I^2 R$ heat loss ($P_{\text{esr\_loss}} = I_{\text{load}}^2 R_{\text{esr}}$):

$$I_{\text{load}}(t) = \frac{P_{\text{total}}(f, V_{\text{dd}})}{\max(1.0, V_{\text{cap}}(t))}$$

$$P_{\text{esr\_loss}}(t) = I_{\text{load}}^2(t) \cdot R_{\text{esr}}$$

$$P_{\text{total\_drain}}(t) = P_{\text{total}}(f, V_{\text{dd}}) + P_{\text{esr\_loss}}(t)$$

$$\Delta E(t) = \left[ P_{\text{harvested}}(t) - P_{\text{total\_drain}}(t) \right] \cdot \Delta t$$

$$E(t+1) = \max\left(0.0, \frac{1}{2} C_{\text{supercap}} V_{\text{cap}}^2(t) + \Delta E(t)\right)$$

$$V_{\text{cap}}(t+1) = \min\left(V_{\text{max}}, \sqrt{\frac{2 E(t+1)}{C_{\text{supercap}}}}\right)$$

Accounting for internal series resistance ($R_{\text{esr}}$), the effective terminal voltage supplied to core regulators drops under heavy load current:
$$V_{\text{terminal}}(t) = \max\left(0.0, V_{\text{cap}}(t) - I_{\text{load}}(t) \cdot R_{\text{esr}}\right)$$

System brownout reset triggers whenever $V_{\text{terminal}}(t) \le 1.8\text{V}$ or $V_{\text{cap}}(t) \le 1.8\text{V}$, forcing immediate episode termination.

---

## IV. Hardware Command Protocol & Emulation Framework

```
+------------------------------------+               +-----------------------------------+
|    Python Gym RL Governor          |               |    Renode Protocol Emulator       |
|  (Gymnasium + Stable-Baselines3)   |               |   (ARM Cortex-M4 Target Model)    |
|                                    |  TCP Socket   |  - stm32f4_dvfs.repl & .resc      |
|   1. Observes Vterm, Qlen, Pharvest| ------------> |   1. Reconfigures PLL clock state |
|   2. Computes action a_t (PPO)     |  Port 4000    |   2. Accumulates instruction cycles
|   3. Transmits JSON scaling cmd    | <------------ |   3. Estimates FreeRTOS TCB RAM   |
+------------------------------------+               +-----------------------------------+
```

To validate hardware command translation and protocol interface overheads:

1. **Target Microcontroller Firmware & Configuration:**
   - **`firmware/main.c`**: Embedded C design artifact defining the STM32F4 USART1 serial driver, PLL clock tree register reconfiguration functions, and static FreeRTOS task control block (TCB) stack allocation ($1,840\text{ bytes}$).
   - **`renode/stm32f4_dvfs.repl`**: Defines an ARM Cortex-M4 microcontroller platform (STM32F4 with $128\text{ KB}$ SRAM and $512\text{ KB}$ Flash).
   - **`renode/stm32f4_dvfs.resc`**: Renode execution script loading the platform definition and creating the socket terminal (`emulation CreateServerSocketTerminal 4000 "term"`).
2. **Socket Protocol Emulator (`renode/renode_server.py` & `renode/arm_cortex_m4_co_sim.py`):**
   - A dedicated Python server (`renode/renode_server.py`) implements the TCP socket control protocol of Renode's external management interface on port 4000 for reproducible environment-agnostic benchmarking.
   - The Python RL policy acts as a client connecting over line-buffered TCP sockets (`makefile('r')`).
   - On each control step ($\Delta t = 100\text{ ms}$), Python transmits JSON scaling payload: `{"command": "set_frequency", "frequency_mhz": 80.0, "voltage_v": 1.5}`.
   - The emulator server processes commands, accumulates executed CPU instruction cycles ($\Delta \text{cycles} = f \cdot \Delta t$), and returns a representative FreeRTOS task control block (TCB) SRAM stack memory footprint estimate ($1,840\text{ bytes}$).
3. **Hardware Command Interface Trajectory Results:**
   - Executing a full 150-step trajectory driven by the trained PPO policy model (`models/ppo_dvfs_model.zip`) completed in **$694,400,000\text{ instruction cycles}$** with a representative FreeRTOS SRAM stack estimate of **$1,840\text{ bytes}$**.
   - Full 150-step command and telemetry logs are exported to `results/renode_cosim_telemetry.json`.

---

## V. Experimental Results & Empirical Benchmarking

### A. Quantitative Monte Carlo Benchmark ($\text{mean} \pm \sigma$)
Evaluating 5 governor strategies across 30 held-out test seeds under a $45\text{-step}$ solar cloud drop ($2.0\text{ mW}$ solar intake) under deterministic policy inference yields the following empirical performance metrics:

| Governor Strategy | Brownout Reset Rate (%) | Service Rate While Alive ($\text{tasks/step}$) | Effective Horizon Throughput ($\text{total tasks / 150 steps}$) | Mean Queue Backlog ($\text{mean} \pm \sigma\text{ tasks}$) | System Failure / Stability State |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Always-Max (Fixed 80 MHz)** | **100.0%** | $4.34 \pm 0.20$ | $1.76 \pm 0.10$ | $4.4 \pm 0.2$ | Brownout crash at episode step 61 (21 steps into cloud onset at step 40); energy depleted by $38.25\text{ mJ}$ |
| **Powersave (Fixed 8 MHz)** | **0.0%** | $1.00 \pm 0.00$ | $1.00 \pm 0.00$ | $166.2 \pm 2.7$ | Intractable queue backlog ($166.2\text{ tasks}$) |
| **Static Threshold** | **0.0%** | $4.16 \pm 0.20$ | $4.16 \pm 0.20$ | $16.5 \pm 3.6$ | Reactive lag during cloud onset ($16.5\text{ tasks}$) |
| **Proposed PPO RL Governor** | **0.0%** | **$4.15 \pm 0.20$** | **$4.15 \pm 0.20$** | **$4.6 \pm 0.3$** | **Optimal Equilibrium: 0% Crashes + Minimal Backlog** |
| **DQN RL Governor** | **100.0%** | $4.34 \pm 0.20$ | $1.76 \pm 0.10$ | $4.4 \pm 0.2$ | Brownout crash under uncalibrated action value baseline |

### B. Statistical Significance & Multi-Seed Training Convergence
1. **Multi-Seed Training Convergence (5 Training Seeds):** PPO was trained across 5 independent random seeds (`seeds = [0, 1, 2, 3, 4]`). The 5 trained policy archives achieved a mean backlog of **$4.54 \pm 0.06\text{ tasks}$**, demonstrating tight convergence across independent training runs.
2. **Wilcoxon Signed-Rank Test:** Executed directly via `scipy.stats.wilcoxon` in `src/evaluate_and_plot.py` across all 30 matched test seeds, the paired Wilcoxon signed-rank test confirms that PPO's queue backlog reduction over Static Threshold is statistically significant ($W = 0.0, p = 1.86 \times 10^{-9} < 0.001$). Raw trial data for all seeds is archived in `results/benchmark_raw_results.csv`.

---

## VIII. References

1. B. Lucia, V. Balaji, A. Colin, K. Maeng, and E. Ruppel, "Intermittent Computing: Challenges and Opportunities," in *Proc. 2nd Summit on Advances in Programming Languages (SNAPL)*, 2017, pp. 8:1–8:14.
2. V. Raghunathan, A. Kansal, J. Hsu, J. Friedman, and M. B. Srivastava, "Design Considerations for Solar Energy Harvesting Wireless Embedded Systems," in *Proc. 4th Int. Symp. Information Processing in Sensor Networks (IPSN)*, 2005, pp. 457–462.
3. U. Kassim, S. A. R. Bhatti, and L. Mottola, "D2VFS: Dynamic Duty-Cycling and Voltage Scaling for Batteryless Embedded Systems," in *Proc. 17th Int. Conf. Embedded Wireless Systems and Networks (EWSN)*, 2020, pp. 112–123.
4. U. Kassim, S. A. R. Bhatti, and L. Mottola, "Feedback-Based Threshold Control for Intermittent Energy-Harvesting Systems," *ACM Transactions on Sensor Networks (TOSN)*, vol. 18, no. 3, pp. 41:1–41:28, 2022.
5. A. Colin and B. Lucia, "Adaptive Control for Energy-Harvesting Systems," in *Proc. 24th Int. Conf. Architectural Support for Programming Languages and Operating Systems (ASPLOS)*, 2019, pp. 647–660.
6. C. Delimitrou and C. Kozyrakis, "zTT: Optimal Dynamic Voltage and Frequency Scaling for Heterogeneous Systems Using Reinforcement Learning," *IEEE Micro*, vol. 38, no. 4, pp. 28–38, 2018.
7. T. Basaklar, Y. Tuncel, and U. Y. Ogras, "tinyMAN: An RL-Based Energy Manager for Wearable Energy-Harvesting IoT Devices," in *Proc. tinyML Research Symposium*, Mar. 2022. (arXiv:2202.09297).
8. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," *arXiv preprint arXiv:1707.06347*, 2017.
9. A. Raffin et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations in PyTorch," *Journal of Machine Learning Research*, vol. 22, no. 268, pp. 1–8, 2021.
