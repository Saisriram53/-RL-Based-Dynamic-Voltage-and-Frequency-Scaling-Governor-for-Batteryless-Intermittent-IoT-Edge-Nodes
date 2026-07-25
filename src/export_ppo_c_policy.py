import os
import torch
from stable_baselines3 import PPO

def export_ppo_to_c(model_path, output_header_path):
    """
    Extracts PPO policy MLP weights & biases from Stable-Baselines3 PyTorch archive
    and generates standalone C float array constants for ARM Cortex-M4 MCU deployment.
    """
    model = PPO.load(model_path)
    state_dict = model.policy.state_dict()
    
    # 1. Extract Tensors
    w1 = state_dict['mlp_extractor.policy_net.0.weight'].cpu().numpy()  # (64, 5)
    b1 = state_dict['mlp_extractor.policy_net.0.bias'].cpu().numpy()    # (64,)
    
    w2 = state_dict['mlp_extractor.policy_net.2.weight'].cpu().numpy()  # (64, 64)
    b2 = state_dict['mlp_extractor.policy_net.2.bias'].cpu().numpy()    # (64,)
    
    w3 = state_dict['action_net.weight'].cpu().numpy()                   # (4, 64)
    b3 = state_dict['action_net.bias'].cpu().numpy()                     # (4,)
    
    # 2. Format C Arrays
    def format_2d(arr, name):
        rows, cols = arr.shape
        lines = [f"const float {name}[{rows}][{cols}] = {{"]
        for r in range(rows):
            vals = ", ".join([f"{val:.8f}f" for val in arr[r]])
            lines.append(f"    {{ {vals} }},")
        lines.append("};\n")
        return "\n".join(lines)
        
    def format_1d(arr, name):
        vals = ", ".join([f"{val:.8f}f" for val in arr])
        return f"const float {name}[{len(arr)}] = {{ {vals} }};\n"

    header_content = f"""/* Auto-generated PPO Policy Neural Network Weights & Biases */
#ifndef NN_POLICY_H
#define NN_POLICY_H

#ifdef __cplusplus
extern "C" {{
#endif

/* Layer 1: 5 -> 64 (Dense + Tanh) */
{format_2d(w1, "W1_WEIGHTS")}
{format_1d(b1, "B1_BIASES")}

/* Layer 2: 64 -> 64 (Dense + Tanh) */
{format_2d(w2, "W2_WEIGHTS")}
{format_1d(b2, "B2_BIASES")}

/* Action Output Head: 64 -> 4 Logits */
{format_2d(w3, "W3_WEIGHTS")}
{format_1d(b3, "B3_BIASES")}

int predict_action_on_chip(const float obs[5]);

#ifdef __cplusplus
}}
#endif

#endif /* NN_POLICY_H */
"""

    os.makedirs(os.path.dirname(output_header_path), exist_ok=True)
    with open(output_header_path, "w") as f:
        f.write(header_content)

    print(f"Successfully exported PPO Neural Network weights to: {output_header_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_file = os.path.join(base_dir, "models", "ppo_dvfs_model.zip")
    header_file = os.path.join(base_dir, "firmware", "nn_policy.h")
    export_ppo_to_c(model_file, header_file)
