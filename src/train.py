import os
import sys
import numpy as np

# Add src to sys.path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import EnergyHarvestingDVFSEnv
from stable_baselines3 import PPO, DQN

def train_agents(total_timesteps=61440, seed=0):
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"--- Training PPO RL DVFS Governor ({total_timesteps} timesteps, seed={seed}) ---")
    env_ppo = EnergyHarvestingDVFSEnv(profile='standard_cloudy')
    env_ppo.reset(seed=seed)
    ppo_model = PPO("MlpPolicy", env_ppo, learning_rate=0.001, seed=seed, verbose=1)
    ppo_model.learn(total_timesteps=total_timesteps)
    
    ppo_path = os.path.join(models_dir, "ppo_dvfs_model.zip")
    ppo_model.save(ppo_path)
    print(f"PPO Model successfully saved to {ppo_path}")

    print(f"\n--- Training DQN RL DVFS Governor ({total_timesteps} timesteps, seed={seed}) ---")
    env_dqn = EnergyHarvestingDVFSEnv(profile='standard_cloudy')
    env_dqn.reset(seed=seed)
    dqn_model = DQN("MlpPolicy", env_dqn, learning_rate=0.001, learning_starts=1000, seed=seed, verbose=1)
    dqn_model.learn(total_timesteps=total_timesteps)
    
    dqn_path = os.path.join(models_dir, "dqn_dvfs_model.zip")
    dqn_model.save(dqn_path)
    print(f"DQN Model successfully saved to {dqn_path}")

if __name__ == "__main__":
    train_agents()
