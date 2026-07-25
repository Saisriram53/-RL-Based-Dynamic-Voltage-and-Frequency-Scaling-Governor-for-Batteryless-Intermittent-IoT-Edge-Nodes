import os
import sys
import shutil
import subprocess

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.export_ppo_c_policy import export_ppo_to_c
try:
    from firmware.build_elf import create_arm_cortex_m4_elf
except ImportError:
    from build_elf import create_arm_cortex_m4_elf

def build_closed_loop_firmware():
    """
    Builds closed-loop ARM Cortex-M4 ELF firmware image containing
    the exported C PPO Neural Network inference engine (nn_policy.c / nn_policy.h).
    """
    firmware_dir = os.path.join(base_dir, "firmware")
    models_dir = os.path.join(base_dir, "models")
    
    model_path = os.path.join(models_dir, "ppo_dvfs_model.zip")
    header_path = os.path.join(firmware_dir, "nn_policy.h")
    output_elf = os.path.join(firmware_dir, "firmware.elf")

    # 1. Export PyTorch weights to C header
    print(f"[Firmware Builder] Exporting PPO PyTorch policy weights to {header_path}...")
    export_ppo_to_c(model_path, header_path)

    # 2. Check for ARM GCC Cross-Compiler
    gcc_path = shutil.which("arm-none-eabi-gcc")
    if gcc_path:
        print(f"[Firmware Builder] Found ARM GCC Toolchain at: {gcc_path}")
        linker_script = os.path.join(firmware_dir, "linker.ld")
        startup_file = os.path.join(firmware_dir, "startup_stm32f4.c")
        main_file = os.path.join(firmware_dir, "main_closed_loop.c")
        policy_file = os.path.join(firmware_dir, "nn_policy.c")

        cmd = [
            gcc_path,
            "-mcpu=cortex-m4",
            "-mthumb",
            "-mfpu=fpv4-sp-d16",
            "-mfloat-abi=hard",
            "--specs=nano.specs",
            "--specs=nosys.specs",
            f"-T{linker_script}",
            startup_file,
            main_file,
            policy_file,
            "-o", output_elf,
            "-lm"
        ]
        
        print("[Firmware Builder] Compiling closed-loop C firmware with hardware FPU flags...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[Firmware Builder] Successfully compiled closed-loop C firmware.elf at: {output_elf}")
            return True
        else:
            print(f"[Firmware Builder] GCC Compilation error:\n{res.stderr}")
            print("[Firmware Builder] Falling back to bare-metal assembly generator...")
    else:
        print("[Firmware Builder] arm-none-eabi-gcc cross-compiler not found on system PATH.")
        print("[Firmware Builder] Using bare-metal assembly ELF generator...")

    # 3. Fallback Assembly Builder
    create_arm_cortex_m4_elf(output_elf)
    return False

if __name__ == "__main__":
    build_closed_loop_firmware()
