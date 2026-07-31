"""
Regression test for the paper's hardware-validation claim (Section IV):
the on-chip C policy (firmware/nn_policy.c) must match the PyTorch reference
policy's action on every step of a fixed evaluation trajectory.

This compiles the real nn_policy.c (the same file shipped to the ARM Cortex-M4
target, minus the ARM-specific cross-compile flags) into a native shared library
and calls it directly via ctypes, so a change to the exported weights, the
normalization constants, or the C forward-pass logic itself will break this test --
unlike a Python re-implementation, which could drift from nn_policy.c undetected.

Skips (does not fail) if no native C compiler is available, since this machine
may not have one; it is expected to run for real in CI (see .github/workflows/smoke_test.yml).
"""
import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "src"))

from environment import EnergyHarvestingDVFSEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

MODEL_PATH = os.path.join(REPO_ROOT, "models", "ppo_dvfs_model.zip")
VECNORM_PATH = os.path.join(REPO_ROOT, "models", "ppo_dvfs_model_vecnormalize.pkl")
NN_POLICY_C = os.path.join(REPO_ROOT, "firmware", "nn_policy.c")
NN_POLICY_H_DIR = os.path.join(REPO_ROOT, "firmware")


def _find_native_compiler():
    for candidate in ("cc", "gcc", "clang"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def _compile_nn_policy_shared_lib(compiler, build_dir):
    """Compiles the real firmware/nn_policy.c natively (no ARM cross-compile flags)
    into a shared library exposing predict_action_on_chip for direct ctypes calls."""
    if sys.platform == "win32":
        lib_name = "nn_policy.dll"
    elif sys.platform == "darwin":
        lib_name = "libnn_policy.dylib"
    else:
        lib_name = "libnn_policy.so"

    lib_path = os.path.join(build_dir, lib_name)
    cmd = [
        compiler,
        "-shared",
        "-fPIC",
        "-O2",
        f"-I{NN_POLICY_H_DIR}",
        NN_POLICY_C,
        "-o", lib_path,
        "-lm",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"Failed to compile nn_policy.c natively:\n{result.stderr}")
    return lib_path


@pytest.fixture(scope="module")
def c_policy_predict(tmp_path_factory):
    compiler = _find_native_compiler()
    if compiler is None:
        pytest.skip("No native C compiler (gcc/cc/clang) found on this machine")

    build_dir = str(tmp_path_factory.mktemp("nn_policy_build"))
    lib_path = _compile_nn_policy_shared_lib(compiler, build_dir)

    lib = ctypes.CDLL(lib_path)
    lib.predict_action_on_chip.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.predict_action_on_chip.restype = ctypes.c_int

    def predict(obs_5d):
        arr = (ctypes.c_float * len(obs_5d))(*[float(x) for x in obs_5d])
        return lib.predict_action_on_chip(arr)

    return predict


@pytest.mark.skipif(
    not (os.path.exists(MODEL_PATH) and os.path.exists(VECNORM_PATH)),
    reason="Trained model / VecNormalize stats not present (run src/train.py first)",
)
def test_c_policy_matches_pytorch_reference(c_policy_predict):
    model = PPO.load(MODEL_PATH)
    dummy_env = DummyVecEnv([lambda: EnergyHarvestingDVFSEnv(profile='standard_cloudy')])
    vecnorm = VecNormalize.load(VECNORM_PATH, dummy_env)
    vecnorm.training = False

    env = EnergyHarvestingDVFSEnv(profile='standard_cloudy')
    obs, _ = env.reset(seed=100)

    mismatches = []
    for step in range(150):
        norm_obs = vecnorm.normalize_obs(obs)
        py_action, _ = model.predict(norm_obs, deterministic=True)
        py_action = int(py_action.item()) if isinstance(py_action, np.ndarray) else int(py_action)

        # nn_policy.c takes the RAW physical observation and renormalizes internally using the
        # baked-in OBS_NORM_MEAN/OBS_NORM_VAR constants (mirroring VecNormalize) -- this matches
        # exactly how the real firmware/Renode co-simulation calls it (raw obs over MMIO). Passing
        # an already-normalized observation here would double-normalize and silently corrupt the
        # comparison; caught via manual cross-check against a from-scratch reimplementation before
        # this test was trusted.
        c_action = c_policy_predict(obs)

        if py_action != c_action:
            mismatches.append((step, py_action, c_action))

        obs, _, terminated, truncated, _ = env.step(py_action)
        if terminated or truncated:
            break

    assert not mismatches, (
        f"{len(mismatches)} on-chip/PyTorch action mismatches out of 150 steps: {mismatches[:10]}"
    )
