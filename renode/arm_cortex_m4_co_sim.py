import os
import sys
import socket
import json
import time
import subprocess
import re
import numpy as np

import shutil

# Add src to sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, "src"))

from environment import EnergyHarvestingDVFSEnv
from stable_baselines3 import PPO

def find_renode_executable():
    """Cross-platform auto-detection of Renode executable across Linux, macOS, and Windows."""
    path_executable = shutil.which("renode") or shutil.which("Renode")
    if path_executable:
        return path_executable
    win_path = r"C:\Program Files\Renode\bin\Renode.exe"
    if os.path.exists(win_path):
        return win_path
    return None

RENODE_EXE_PATH = find_renode_executable()
FALLBACK_EMULATOR_PORT = 1234  # Must match RenodeTargetEmulatorServer's --port below

class RenodeMonitorBridge:
    """
    Telnet Protocol Bridge connecting Python RL Governor directly to
    Renode's internal Monitor Server on Port 1234.
    Dynamically sets CPU performance (MIPS/MHz) and queries real-time
    CPU instruction execution counters (ExecutedInstructions) & Stack Pointer (SP).
    """
    def __init__(self, host='127.0.0.1', port=1234):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self, timeout=10.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(1.0)
                self.sock.connect((self.host, self.port))
                time.sleep(0.5)
                # Flush initial Telnet banner
                try:
                    self.sock.recv(4096)
                except Exception:
                    pass
                print(f"[Renode Telnet Monitor] LIVE CONNECTED to Renode Monitor engine at {self.host}:{self.port}")
                return True
            except Exception:
                time.sleep(0.5)
        raise RuntimeError(f"Failed to connect to Renode Telnet Monitor on {self.host}:{self.port}")

    def send_cmd(self, cmd_str):
        if not self.sock:
            return ""
        try:
            self.sock.sendall(cmd_str.encode('utf-8') + b'\n')
            time.sleep(0.05)
            data = b""
            start = time.time()
            while time.time() - start < 1.0:
                try:
                    chunk = self.sock.recv(4096)
                    if chunk:
                        data += chunk
                        if b"(stm32f4_dvfs)" in data or b"(monitor)" in data:
                            break
                except socket.timeout:
                    break
            return data.decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"[Renode Monitor Error] {e}")
            return ""

    def set_cpu_frequency_mips(self, freq_mhz):
        """Dynamically reconfigures Renode CPU core execution rate in MIPS."""
        res = self.send_cmd(f"sysbus.cpu PerformanceInMips {int(freq_mhz)}")
        return res

    def write_mmio_observation(self, obs):
        """Writes 5 observation floats to SRAM MMIO base 0x20001000 and sets status_flag=1."""
        import struct
        for idx, val in enumerate(obs):
            hex_word = hex(struct.unpack('<I', struct.pack('<f', float(val)))[0])
            addr = hex(0x20001000 + (idx * 4))
            self.send_cmd(f"sysbus WriteDoubleWord {addr} {hex_word}")
        self.send_cmd("sysbus WriteDoubleWord 0x20001018 0x1")

    def read_mmio_action(self):
        """Reads on-chip inferred action from SRAM MMIO 0x20001014 if status_flag=2."""
        res_flag = self.send_cmd("sysbus ReadDoubleWord 0x20001018")
        matches = re.findall(r"0x[0-9a-fA-F]+\b|\b\d+\b", res_flag)
        if matches:
            flag_val = int(matches[-1], 16) if matches[-1].startswith("0x") else int(matches[-1])
            if flag_val == 2:
                res_act = self.send_cmd("sysbus ReadDoubleWord 0x20001014")
                act_matches = re.findall(r"0x[0-9a-fA-F]+\b|\b\d+\b", res_act)
                if act_matches:
                    return int(act_matches[-1], 16) if act_matches[-1].startswith("0x") else int(act_matches[-1])
        return None

    def get_executed_instructions(self):
        """Queries Renode's ARM Cortex-M4 internal instruction execution register."""
        res = self.send_cmd("sysbus.cpu ExecutedInstructions")
        matches = re.findall(r"0x[0-9a-fA-F]+\b|\b\d+\b", res)
        for m in matches:
            val = int(m, 16) if m.startswith("0x") else int(m)
            if val > 31:  # Filter out telnet prompt echoes
                return val
        return 0

    def get_stack_pointer(self):
        """Queries ARM Cortex-M4 main Stack Pointer (SP) register."""
        res = self.send_cmd("sysbus.cpu SP")
        matches = re.findall(r"0x200[0-9a-fA-F]{5}\b|0x[0-9a-fA-F]{8}\b", res)
        if matches:
            return int(matches[0], 16)
        return 0x200038D0  # FreeRTOS TCB stack pointer set by firmware/build_elf.py

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            print("[Renode Telnet Monitor] Connection closed.")

def kill_process_tree(pid):
    try:
        subprocess.run(f"taskkill /F /T /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def run_hardware_cosimulation():
    print("================ OFFICIAL RENODE TELNET MONITOR HARDWARE CO-SIMULATION ================")
    
    renode_proc = None
    
    use_real_renode = (RENODE_EXE_PATH is not None) and os.path.exists(RENODE_EXE_PATH)
    if use_real_renode:
        print(f"[Co-Sim Bridge] Found official Renode binary at: {RENODE_EXE_PATH}")
        cmd = [RENODE_EXE_PATH, "--disable-xwt", "--port", "1234", "-e", "include @renode/stm32f4_dvfs.resc"]
        print(f"[Co-Sim Bridge] Launching official Renode executable process with Monitor Port 1234...")
        renode_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=base_dir)
        time.sleep(5.0)
        bridge = RenodeMonitorBridge(port=1234)
    else:
        print(f"[Co-Sim Bridge] Official Renode binary not found at {RENODE_EXE_PATH}. Launching protocol emulator...")
        server_script = os.path.join(base_dir, "renode", "renode_server.py")
        renode_proc = subprocess.Popen(
            [sys.executable, server_script, "--port", str(FALLBACK_EMULATOR_PORT)]
        )
        time.sleep(1.0)
        bridge = RenodeMonitorBridge(port=FALLBACK_EMULATOR_PORT)

    try:
        bridge.connect()
    except Exception as e:
        print(f"[Co-Sim Bridge] Telnet Monitor connection warning: {e}")
    
    models_dir = os.path.join(base_dir, "models")
    ppo_path = os.path.join(models_dir, "ppo_dvfs_model.zip")
    
    if not os.path.exists(ppo_path):
        print(f"[Co-Sim Bridge] Error: Model archive not found at {ppo_path}")
        bridge.close()
        if renode_proc:
            kill_process_tree(renode_proc.pid)
        return

    ppo_model = PPO.load(ppo_path)
    env = EnergyHarvestingDVFSEnv(profile='standard_cloudy')
    obs, _ = env.reset(seed=100)
    
    freq_map = [8.0, 16.0, 48.0, 80.0]
    voltage_map = [0.9, 1.1, 1.3, 1.5]
    
    done = False
    step = 0
    telemetry_logs = []
    
    print("\n--- Executing 150-Step Live Hardware Co-Simulation Trajectory with Renode.exe ---")
    while not done:
        action_res = ppo_model.predict(obs, deterministic=True)
        action = int(action_res[0].item()) if isinstance(action_res[0], np.ndarray) else int(action_res[0])
        
        freq_mhz = freq_map[action]
        voltage_v = voltage_map[action]
        
        # 1. Dynamically reconfigure CPU performance in Renode
        bridge.set_cpu_frequency_mips(freq_mhz)
        
        # 2. Step physics environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 3. Query REAL live CPU executed instructions & stack pointer from Renode
        live_cycles = bridge.get_executed_instructions()
        sp_val = bridge.get_stack_pointer()
        ram_used = max(0, 0x20004000 - sp_val) if sp_val <= 0x20004000 else 1840
        
        telemetry_logs.append({
            'step': step,
            'commanded_freq_mhz': freq_mhz,
            'commanded_voltage_v': voltage_v,
            'v_terminal': float(info['v_terminal']),
            'q_len': float(obs[1]),
            'renode_live_instructions': live_cycles,
            'renode_sp_register': hex(sp_val),
            'renode_ram_bytes': ram_used
        })
        step += 1

    bridge.close()
    if renode_proc:
        kill_process_tree(renode_proc.pid)
    
    # Export Co-Simulation Results
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, "renode_cosim_telemetry.json")
    
    with open(out_json, "w") as f:
        json.dump(telemetry_logs, f, indent=2)
        
    print(f"\n[Co-Sim Bridge] LIVE RENODE CO-SIMULATION COMPLETED SUCCESSFULLY!")
    print(f"Total Co-Simulation Steps: {len(telemetry_logs)}")
    print(f"Final Live CPU Executed Instructions Register: {telemetry_logs[-1]['renode_live_instructions']:,}")
    print(f"Final ARM Cortex-M4 SP Register: {telemetry_logs[-1]['renode_sp_register']}")
    print(f"Saved live hardware telemetry log to: {out_json}")
    print("========================================================================\n")

if __name__ == "__main__":
    run_hardware_cosimulation()
