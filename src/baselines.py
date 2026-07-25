class AlwaysMaxGovernor:
    """Always-Max Governor: Locks CPU at maximum frequency (80 MHz @ 1.5V)."""
    def predict(self, obs, *args, **kwargs):
        return 3, None

class PowersaveGovernor:
    """Powersave Governor: Locks CPU at minimum frequency (8 MHz @ 0.9V)."""
    def predict(self, obs, *args, **kwargs):
        return 0, None

class StaticThresholdGovernor:
    """
    Static Threshold Governor:
    Scales CPU frequency across all 4 discrete levels based on capacitor voltage thresholds.
    - V_terminal > 2.80 V          -> 80 MHz (Action 3)
    - 2.50 V <= V_terminal <= 2.80V -> 48 MHz (Action 2)
    - 2.20 V <= V_terminal < 2.50V  -> 16 MHz (Action 1)
    - V_terminal < 2.20 V          -> 8 MHz (Action 0)
    """
    def predict(self, obs, *args, **kwargs):
        v_terminal = obs[0] if hasattr(obs, '__getitem__') else obs
        if v_terminal > 2.80:
            return 3, None
        elif v_terminal >= 2.50:
            return 2, None
        elif v_terminal >= 2.20:
            return 1, None
        else:
            return 0, None
