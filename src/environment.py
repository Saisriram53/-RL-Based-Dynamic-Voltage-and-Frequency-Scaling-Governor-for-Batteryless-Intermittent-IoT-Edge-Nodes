import numpy as np
import gymnasium as gym
from gymnasium import spaces

class EnergyHarvestingDVFSEnv(gym.Env):
    """
    Advanced Physics-Informed Gymnasium environment modeling a Partially Observable
    Markov Decision Process (POMDP) for an intermittent, batteryless IoT node with
    Supercapacitor ESR I^2 R losses, PLL Clock Lock Latency, Thermal Leakage Drift, 
    and Domain Randomization for Sim-to-Real Transfer.

    Physics Equations:
    ------------------
    1. Supercapacitor Stored Energy:
       E_cap(t) = 0.5 * C_supercap * V_cap(t)^2
    
    2. Dynamic CMOS Power Dissipation:
       P_dynamic = alpha_CL * V_dd(f)^2 * f
    
    3. Total Power Consumption & ESR Resistive Drain:
       P_consumed = P_dynamic + P_leakage
       I_load = P_consumed / max(1.0, V_cap)
       P_esr_loss = I_load^2 * R_esr
       P_total_drain = P_consumed + P_esr_loss

    4. Energy Integration:
       dE = (P_harvested - P_total_drain) * dt
       E_new = max(0.0, E_cap(t) + dE)
       V_cap(t+1) = sqrt(2 * E_new / C_supercap)

    5. Terminal Voltage Drop under Load:
       V_terminal = max(0.0, V_cap(t+1) - I_load * R_esr)

    Reward Function:
    ----------------
    r_t = -200.0                       if (V_terminal <= 1.8V or V_cap <= 1.8V) [Brownout Reset]
    r_t = +3.0 * TasksProcessed        otherwise
          -0.4 * ActiveQueueLength
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, profile='standard_cloudy', domain_randomization=True, C_supercap=0.010, include_gradient=True):
        super(EnergyHarvestingDVFSEnv, self).__init__()
        
        # 1. Frequency Steps (MHz) & Core Voltages (Volts)
        self.freq_steps = np.array([8.0, 16.0, 48.0, 80.0])  # MHz
        self.voltage_steps = np.array([0.9, 1.1, 1.3, 1.5])  # Volts
        self.action_space = spaces.Discrete(len(self.freq_steps))
        self.include_gradient = include_gradient
        
        # 2. Physical Hardware Parameters
        self.C_supercap = float(C_supercap) # Configurable Supercapacitor capacitance (F)
        self.V_max = 3.3                   # Maximum operating voltage ceiling
        self.V_brownout = 1.8              # Crash threshold (SRAM reset)
        self.alpha_CL = 1e-10              # Effective capacitance switching load (100 pF)
        self.base_P_leakage = 0.002        # Static baseline leakage (2 mW)
        self.R_esr_base = 0.5              # 0.5 Ohm Supercapacitor Equivalent Series Resistance (ESR)
        self.domain_randomization = domain_randomization
        
        # 3. Observation Space: [V_terminal (V), Task Queue Length, Harvested Power (W), dP_harvested (W), Prev_Action_Norm]
        if self.include_gradient:
            self.observation_space = spaces.Box(
                low=np.array([1.0, 0.0, 0.0, -0.1, 0.0], dtype=np.float32),
                high=np.array([3.3, 200.0, 0.1, 0.1, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Ablation observation space without gradient feature
            self.observation_space = spaces.Box(
                low=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([3.3, 200.0, 0.1, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        
        self.profile = profile
        self.max_steps = 150
        self.reset()

    def _generate_solar_profile(self):
        t = np.linspace(0, 4 * np.pi, self.max_steps)
        if self.profile == 'standard_cloudy':
            trace = 0.02 * np.sin(t) + 0.025
            trace[40:85] = 0.002  # 45-step heavy cloud drop (2 mW harvested)
        elif self.profile == 'volatile':
            trace = 0.02 * np.sin(t) + 0.025 + 0.005 * self.np_random.normal(0, 1, self.max_steps)
            trace[40:85] = 0.002
        elif self.profile == 'clear_day':
            trace = 0.03 * np.sin(t) + 0.035
        else:
            trace = 0.02 * np.sin(t) + 0.025
            trace[40:85] = 0.002
            
        return np.clip(trace, 0.001, 0.08)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.V_cap = 3.0           # Fully charged initial state (3.0 V)
        self.queue_length = 25.0   # Pending task queue
        self.time_step = 0
        self.prev_action = 0
        self.prev_p_harvested = 0.025
        self.solar_trace = self._generate_solar_profile()
        
        # Apply Domain Randomization across episodes using Gymnasium isolated RNG
        if self.domain_randomization:
            self.R_esr = float(self.np_random.uniform(0.3, 0.7))         # ESR varies between 0.3 - 0.7 Ohm
            self.P_leakage = float(self.np_random.uniform(0.0015, 0.0025))# Leakage varies between 1.5 - 2.5 mW
        else:
            self.R_esr = self.R_esr_base
            self.P_leakage = self.base_P_leakage
            
        return self._get_obs(action_power=0.002648), {}

    def _get_obs(self, action_power=0.002648):
        # Use exact self-consistent terminal voltage computed during physics step
        V_terminal = getattr(self, 'last_v_terminal', float(self.V_cap))
        p_harvest = self.solar_trace[self.time_step % len(self.solar_trace)]
        dp_harvest = p_harvest - self.prev_p_harvested
        prev_act_norm = float(self.prev_action) / 3.0
        
        if self.include_gradient:
            return np.array([V_terminal, self.queue_length, p_harvest, dp_harvest, prev_act_norm], dtype=np.float32)
        else:
            return np.array([V_terminal, self.queue_length, p_harvest, prev_act_norm], dtype=np.float32)

    def step(self, action):
        freq = self.freq_steps[action] * 1e6   # Convert MHz to Hz
        v_dd = self.voltage_steps[action]     # Operating Core Voltage
        dt = 0.1                              # Control Interval Δt = 100ms = 0.1s
        
        # 1. Power Dissipation Dynamics with Dynamic CMOS + Static Leakage
        P_dynamic = self.alpha_CL * (v_dd ** 2) * freq
        P_consumed = P_dynamic + self.P_leakage
        
        # 2. Energy-Conserving Supercapacitor Differential Integration with ESR I^2 R Loss
        I_load = P_consumed / max(1.0, self.V_cap)
        P_esr_loss = (I_load ** 2) * self.R_esr
        P_total_drain = P_consumed + P_esr_loss
        
        P_harvested = self.solar_trace[self.time_step % len(self.solar_trace)]
        delta_energy = (P_harvested - P_total_drain) * dt
        current_energy = 0.5 * self.C_supercap * (self.V_cap ** 2)
        new_energy = max(0.0, current_energy + delta_energy)
        self.V_cap = float(np.clip(np.sqrt((2.0 * new_energy) / self.C_supercap), 0.0, self.V_max))
        
        # 3. Terminal Voltage under ESR Drop (Calculated with self-consistent load current)
        I_load_updated = P_consumed / max(1.0, self.V_cap)
        V_terminal = float(np.clip(self.V_cap - (I_load_updated * self.R_esr), 0.0, self.V_max))
        self.last_v_terminal = V_terminal
        
        # 4. Task Queue Dynamics with Physical 50us PLL Lock Delay Overhead (0.05% penalty)
        pll_penalty = 0.9995 if action != self.prev_action else 1.0
        tasks_processed = (freq / 8e6) * pll_penalty
        tasks_processed_actual = min(self.queue_length, tasks_processed)
        self.queue_length = max(0.0, self.queue_length - tasks_processed_actual)
        
        # Poisson task arrival using Gymnasium isolated RNG
        incoming_tasks = int(self.np_random.poisson(lam=4.0))
        self.queue_length = float(min(200.0, self.queue_length + incoming_tasks))
        
        # 5. Reward & Crash Evaluation (Evaluated on Terminal Voltage under ESR drop)
        reward = 0.0
        terminated = False
        truncated = False
        
        is_brownout = (V_terminal <= self.V_brownout or self.V_cap <= self.V_brownout)
        
        if is_brownout:
            reward = -200.0   # Severe brownout crash penalty
            terminated = True
        else:
            reward += (tasks_processed_actual * 3.0)   # Task completion reward
            reward -= (self.queue_length * 0.4)       # Queue backlog penalty
            
        self.prev_p_harvested = P_harvested
        self.prev_action = action
        self.time_step += 1
        
        if self.time_step >= self.max_steps and not terminated:
            truncated = True
            
        obs = self._get_obs(action_power=P_consumed)
        
        return obs, float(reward), terminated, truncated, {
            'brownout': is_brownout,
            'tasks_processed': tasks_processed_actual,
            'v_cap': self.V_cap,
            'v_terminal': V_terminal,
            'p_consumed': P_consumed,
            'p_harvested': P_harvested
        }
