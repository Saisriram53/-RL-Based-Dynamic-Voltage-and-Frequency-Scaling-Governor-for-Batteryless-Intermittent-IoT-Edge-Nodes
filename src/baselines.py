class AlwaysMaxGovernor:
    """Always-Max Governor: Locks CPU at maximum frequency (80 MHz @ 1.5V)."""
    def predict(self, obs):
        return 3, None

class PowersaveGovernor:
    """Powersave Governor: Locks CPU at minimum frequency (8 MHz @ 0.9V)."""
    def predict(self, obs):
        return 0, None

class StaticThresholdGovernor:
    """
    Static Threshold Governor:
    Scales CPU frequency based on fixed capacitor voltage thresholds.
    - V_cap > 2.64 V (80% of V_max) -> 80 MHz (Action 3)
    - V_cap < 2.30 V               -> 8 MHz (Action 0)
    - Otherwise                    -> 16 MHz (Action 1)
    """
    def predict(self, obs):
        v_cap = obs[0]
        if v_cap > 2.64:
            return 3, None
        elif v_cap < 2.30:
            return 0, None
        else:
            return 1, None
