import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import EnergyHarvestingDVFSEnv
from baselines import AlwaysMaxGovernor, PowersaveGovernor, StaticThresholdGovernor
from stable_baselines3 import PPO, DQN

def run_episode(env, governor):
    obs, _ = env.reset(seed=42)
    done = False
    
    v_cap_history = []
    freq_history = []
    queue_history = []
    p_harvest_history = []
    p_consumed_history = []
    
    total_tasks = 0
    brownout_occurred = False
    total_reward = 0
    
    freq_map = [8.0, 16.0, 48.0, 80.0]
    
    while not done:
        try:
            act_res = governor.predict(obs, deterministic=True)
        except TypeError:
            act_res = governor.predict(obs)
        action = act_res[0] if isinstance(act_res, tuple) else act_res
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        v_cap_history.append(info['v_cap'])
        freq_history.append(freq_map[action])
        queue_history.append(obs[1])
        p_harvest_history.append(info['p_harvested'])
        p_consumed_history.append(info['p_consumed'])
        
        total_tasks += info['tasks_processed']
        total_reward += reward
        
        if info['brownout']:
            brownout_occurred = True
            
    return {
        'v_cap': v_cap_history,
        'freq': freq_history,
        'queue': queue_history,
        'p_harvest': p_harvest_history,
        'p_consumed': p_consumed_history,
        'total_tasks': total_tasks,
        'brownout': brownout_occurred,
        'total_reward': total_reward
    }

def benchmark_and_plot():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    ppo_path = os.path.join(models_dir, "ppo_dvfs_model.zip")
    dqn_path = os.path.join(models_dir, "dqn_dvfs_model.zip")
    
    env = EnergyHarvestingDVFSEnv(profile='standard_cloudy')
    
    governors = {
        'Always-Max': AlwaysMaxGovernor(),
        'Powersave': PowersaveGovernor(),
        'Static Threshold': StaticThresholdGovernor(),
    }
    
    if os.path.exists(ppo_path):
        governors['Proposed PPO RL'] = PPO.load(ppo_path)
    if os.path.exists(dqn_path):
        governors['DQN RL'] = DQN.load(dqn_path)

    # 1. Run Quantitative Benchmark across 30 seeds with Wilcoxon Signed-Rank Test
    summary_metrics = {}
    episode_traces = {}
    raw_queue_data = {}
    raw_csv_rows = []
    
    for name, gov in governors.items():
        crashes = 0
        tasks_list = []
        norm_throughput_list = []
        queue_means = []
        num_trials = 30
        
        for seed in range(num_trials):
            env_test = EnergyHarvestingDVFSEnv(profile='standard_cloudy')
            obs, _ = env_test.reset(seed=100 + seed)
            done = False
            t_tasks = 0
            steps_active = 0
            q_lens = []
            crashed = False
            
            while not done:
                try:
                    act_res = gov.predict(obs, deterministic=True)
                except TypeError:
                    act_res = gov.predict(obs)
                act = act_res[0] if isinstance(act_res, tuple) else act_res
                act = int(act.item()) if isinstance(act, np.ndarray) else int(act)
                obs, reward, terminated, truncated, info = env_test.step(act)
                done = terminated or truncated
                t_tasks += info['tasks_processed']
                steps_active += 1
                q_lens.append(obs[1])
                if info['brownout']:
                    crashed = True
                    
            if crashed:
                crashes += 1
            tasks_list.append(t_tasks)
            norm_tp = t_tasks / max(1, steps_active)
            norm_throughput_list.append(norm_tp)
            queue_means.append(np.mean(q_lens))
            
            raw_csv_rows.append({
                'governor': name,
                'seed': 100 + seed,
                'crashed': crashed,
                'total_tasks': t_tasks,
                'active_steps': steps_active,
                'norm_throughput': norm_tp,
                'mean_queue_backlog': np.mean(q_lens)
            })
            
        raw_queue_data[name] = queue_means
        summary_metrics[name] = {
            'crash_rate': (crashes / num_trials) * 100,
            'avg_tasks': np.mean(tasks_list),
            'std_tasks': np.std(tasks_list),
            'avg_norm_tp': np.mean(norm_throughput_list),
            'std_norm_tp': np.std(norm_throughput_list),
            'avg_queue': np.mean(queue_means),
            'std_queue': np.std(queue_means)
        }
        
        # Capture single deterministic trace for plotting
        episode_traces[name] = run_episode(EnergyHarvestingDVFSEnv(profile='standard_cloudy'), gov)

    # Export Raw Benchmark Results CSV
    csv_path = os.path.join(results_dir, "benchmark_raw_results.csv")
    import csv
    if raw_csv_rows:
        keys = raw_csv_rows[0].keys()
        with open(csv_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(raw_csv_rows)
    print(f"Saved raw evaluation trial data to: {csv_path}")

    # Print Quantitative Results Table with Std Dev Confidence Intervals
    print("\n================ QUANTITATIVE BENCHMARK RESULTS (mean ± std) ================")
    print(f"{'Governor Strategy':<20} | {'Brownout Rate (%)':<18} | {'Norm Throughput (tasks/step)':<30} | {'Mean Queue Backlog':<20}")
    print("-" * 96)
    for name, m in summary_metrics.items():
        tp_str = f"{m['avg_norm_tp']:.2f} ± {m['std_norm_tp']:.2f}"
        q_str = f"{m['avg_queue']:.1f} ± {m['std_queue']:.1f}"
        print(f"{name:<20} | {m['crash_rate']:<18.1f} | {tp_str:<30} | {q_str:<20}")
    print("=============================================================================\n")

    # Wilcoxon Signed-Rank Test between PPO and Static Threshold Queue Backlog
    if 'Proposed PPO RL' in raw_queue_data and 'Static Threshold' in raw_queue_data:
        from scipy.stats import wilcoxon
        stat, p_val = wilcoxon(raw_queue_data['Proposed PPO RL'], raw_queue_data['Static Threshold'])
        print(f"Wilcoxon Signed-Rank Test (PPO vs Static Threshold Latency): W = {stat:.1f}, p = {p_val:.6e}")
        if p_val < 0.001:
            print("--> Statistically significant latency reduction achieved by PPO (p < 0.001)!\n")

    # Multi-Profile Sensitivity Analysis
    print("================ MULTI-PROFILE SENSITIVITY ANALYSIS ================")
    print(f"{'Profile Scenario':<18} | {'Governor':<18} | {'Crash Rate (%)':<15} | {'Avg Latency':<15}")
    print("-" * 72)
    profiles = ['standard_cloudy', 'volatile', 'clear_day']
    for p in profiles:
        for name in ['Powersave', 'Static Threshold', 'Proposed PPO RL']:
            if name not in governors:
                continue
            gov = governors[name]
            c_cnt = 0
            q_list = []
            for seed in range(30):
                e_p = EnergyHarvestingDVFSEnv(profile=p)
                o_p, _ = e_p.reset(seed=200 + seed)
                d_p = False
                ql = []
                while not d_p:
                    try:
                        a_res = gov.predict(o_p, deterministic=True)
                    except TypeError:
                        a_res = gov.predict(o_p)
                    a_p = a_res[0] if isinstance(a_res, tuple) else a_res
                    a_p = int(a_p.item()) if isinstance(a_p, np.ndarray) else int(a_p)
                    o_p, r_p, term_p, trunc_p, inf_p = e_p.step(a_p)
                    d_p = term_p or trunc_p
                    ql.append(o_p[1])
                    if inf_p['brownout']:
                        c_cnt += 1
                q_list.append(np.mean(ql))
            print(f"{p:<18} | {name:<18} | {(c_cnt/30)*100:<15.1f} | {np.mean(q_list):<15.1f} ± {np.std(q_list):.1f}")
    print("====================================================================\n")

    # 2. Generate High-Resolution Publication-Quality Plots
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), dpi=300)
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
    
    colors = {
        'Always-Max': '#d9534f',
        'Powersave': '#5bc0de',
        'Static Threshold': '#f0ad4e',
        'Proposed PPO RL': '#5cb85c',
        'DQN RL': '#0275d8'
    }
    
    # (a) Supercapacitor Voltage Trajectories
    for name, trace in episode_traces.items():
        steps = np.arange(len(trace['v_cap'])) * 100  # ms
        axs[0, 0].plot(steps, trace['v_cap'], label=name, color=colors.get(name, 'black'), linewidth=2)
    axs[0, 0].axhline(y=1.8, color='red', linestyle='--', label='Brownout Threshold (1.8V)')
    axs[0, 0].set_title('(a) Supercapacitor Voltage ($V_{cap}$)', fontweight='bold')
    axs[0, 0].set_ylabel('Voltage (V)')
    axs[0, 0].set_xlabel('Time (ms)')
    axs[0, 0].set_ylim(1.0, 3.4)
    axs[0, 0].grid(True, linestyle=':', alpha=0.6)
    axs[0, 0].legend(loc='lower left', fontsize=8)

    # (b) CPU Frequency Scaling
    for name, trace in episode_traces.items():
        steps = np.arange(len(trace['freq'])) * 100  # ms
        axs[0, 1].plot(steps, trace['freq'], label=name, color=colors.get(name, 'black'), linewidth=1.8, drawstyle='steps-post')
    axs[0, 1].set_title('(b) CPU Operating Frequency ($f$)', fontweight='bold')
    axs[0, 1].set_ylabel('Frequency (MHz)')
    axs[0, 1].set_xlabel('Time (ms)')
    axs[0, 1].set_ylim(0, 90)
    axs[0, 1].grid(True, linestyle=':', alpha=0.6)

    # (c) Task Queue Length
    for name, trace in episode_traces.items():
        steps = np.arange(len(trace['queue'])) * 100  # ms
        axs[1, 0].plot(steps, trace['queue'], label=name, color=colors.get(name, 'black'), linewidth=2)
    axs[1, 0].set_title('(c) Task Queue Backlog ($Q_{len}$)', fontweight='bold')
    axs[1, 0].set_ylabel('Pending Tasks')
    axs[1, 0].set_xlabel('Time (ms)')
    axs[1, 0].grid(True, linestyle=':', alpha=0.6)

    # (d) Solar Power vs PPO Consumed Power
    if 'Proposed PPO RL' in episode_traces:
        p_trace = episode_traces['Proposed PPO RL']
        steps = np.arange(len(p_trace['p_harvest'])) * 100  # ms
        axs[1, 1].plot(steps, np.array(p_trace['p_harvest']) * 1000, label='Solar Harvested Power (mW)', color='#e67e22', linewidth=2.2)
        axs[1, 1].plot(steps, np.array(p_trace['p_consumed']) * 1000, label='PPO Consumed Power (mW)', color='#2980b9', linewidth=2.0, linestyle='--')
        axs[1, 1].set_title('(d) Power Dynamics (PPO RL Governor)', fontweight='bold')
        axs[1, 1].set_ylabel('Power (mW)')
        axs[1, 1].set_xlabel('Time (ms)')
        axs[1, 1].grid(True, linestyle=':', alpha=0.6)
        axs[1, 1].legend(loc='upper right', fontsize=8)

    # (e) Brownout Crash Rate Comparison
    strat_names = list(summary_metrics.keys())
    crash_rates = [summary_metrics[k]['crash_rate'] for k in strat_names]
    bar_colors = [colors.get(k, 'gray') for k in strat_names]
    
    axs[2, 0].bar(strat_names, crash_rates, color=bar_colors, alpha=0.85)
    axs[2, 0].set_title('(e) Brownout Crash Rate (%)', fontweight='bold')
    axs[2, 0].set_ylabel('Crash Rate (%)')
    axs[2, 0].set_ylim(0, 100)
    axs[2, 0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(crash_rates):
        axs[2, 0].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    # (f) Task Throughput Comparison
    throughputs = [summary_metrics[k]['avg_tasks'] for k in strat_names]
    axs[2, 1].bar(strat_names, throughputs, color=bar_colors, alpha=0.85)
    axs[2, 1].set_title('(f) Total Completed Task Throughput', fontweight='bold')
    axs[2, 1].set_ylabel('Tasks Processed / Episode')
    axs[2, 1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(throughputs):
        axs[2, 1].text(i, v + 5, f"{v:.0f}", ha='center', fontweight='bold')

    plt.tight_layout()
    
    plot_png = os.path.join(results_dir, "benchmark_performance_comparison.png")
    plot_pdf = os.path.join(results_dir, "benchmark_performance_comparison.pdf")
    
    plt.savefig(plot_png, bbox_inches='tight')
    plt.savefig(plot_pdf, bbox_inches='tight')
    plt.close()
    
    print(f"Publication-ready plots saved to:\n  - {plot_png}\n  - {plot_pdf}")

if __name__ == "__main__":
    benchmark_and_plot()
