import os
import sys
import numpy as np

# Add src to sys.path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import EnergyHarvestingDVFSEnv
from stable_baselines3 import PPO, DQN

def train_agents(total_timesteps=61440, seeds=[0, 1, 2, 3, 4]):
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    primary_seed = seeds[0]
    for s in seeds:
        print(f"\n--- Training PPO RL DVFS Governor ({total_timesteps} timesteps, seed={s}) ---")
        env_ppo = EnergyHarvestingDVFSEnv(profile='standard_cloudy')
        env_ppo.reset(seed=s)
        ppo_model = PPO("MlpPolicy", env_ppo, learning_rate=0.001, seed=s, verbose=0)
        ppo_model.learn(total_timesteps=total_timesteps)
        
        # Save primary and seed-specific archives
        seeds_dir = os.path.join(models_dir, "seeds")
        os.makedirs(seeds_dir, exist_ok=True)
        ppo_path = os.path.join(seeds_dir, f"ppo_dvfs_seed_{s}.zip")
        ppo_model.save(ppo_path)
        if s == primary_seed:
            ppo_model.save(os.path.join(models_dir, "ppo_dvfs_model.zip"))
        print(f"PPO Seed {s} Model successfully saved to {ppo_path}")

    print(f"\n--- Training Primary DQN RL DVFS Governor ({total_timesteps} timesteps, seed={primary_seed}) ---")
    env_dqn = EnergyHarvestingDVFSEnv(profile='standard_cloudy')
    env_dqn.reset(seed=primary_seed)
    dqn_model = DQN("MlpPolicy", env_dqn, learning_rate=0.001, learning_starts=1000, seed=primary_seed, verbose=0)
    dqn_model.learn(total_timesteps=total_timesteps)
    
    dqn_path = os.path.join(models_dir, "dqn_dvfs_model.zip")
    dqn_model.save(dqn_path)
    print(f"DQN Model successfully saved to {dqn_path}")

if __name__ == "__main__":
    train_agents()
