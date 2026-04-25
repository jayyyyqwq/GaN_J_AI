"""
Calibrated battery simulator — stateless physics functions.

Returns delta_v values without maintaining internal battery state.
AgentGridEnvironment._batteries is the single source of truth.

Physics layers:
  - SoC→OCV piecewise curve (nonlinear 18650-style discharge)
  - INA219 noise: ADC quantization + thermal jitter + rare spike
  - State-dependent relay efficiency (higher resistance at low SoC)

Calibration constants are module-level stubs. Replace with values
fitted from INA219 discharge logs and relay transfer measurements.
"""
from __future__ import annotations

import random

# ── Calibration constants — replace with fitted values after hardware measurement ──

# SoC (state of charge, 0–1) → OCV (normalized open-circuit voltage, 0–1)
# Piecewise linear breakpoints. Fits 18650-style plateau-then-cliff curve.
# To recalibrate: discharge at constant load, record (timestamp, voltage),
# normalize both axes, fit breakpoints to the resulting curve.
_SOC_CURVE: list[tuple[float, float]] = [
    (1.00, 1.000),
    (0.80, 0.970),
    (0.50, 0.850),
    (0.20, 0.600),
    (0.00, 0.000),
]

# INA219 ADC resolution (12-bit at ~4.2V full scale, normalized to 0–1)
_ADC_RESOLUTION: float = 0.001

# Thermal + ADC noise floor (Gaussian σ)
_VOLTAGE_NOISE_STD: float = 0.003

# I2C glitch / bit-flip spike probability and magnitude
_SPIKE_PROB: float = 0.01
_SPIKE_STD: float = 0.05

# Relay efficiency: base value and how much it drops as sender SoC falls
# efficiency = max(_MIN_EFFICIENCY, _BASE_EFFICIENCY - _SLOPE * (1 - from_soc))
_BASE_EFFICIENCY: float = 0.85
_EFFICIENCY_SLOPE: float = 0.15
_MIN_EFFICIENCY: float = 0.60

# Leakage current per step when relay is nominally off (small constant drain)
LEAKAGE_PER_STEP: float = 0.001


# ── Module-level helpers (importable for observation formatting) ───────────────

def soc_to_voltage(soc: float) -> float:
    """Map SoC (0–1) to normalized OCV via piecewise linear interpolation."""
    soc = max(0.0, min(1.0, soc))
    for i in range(len(_SOC_CURVE) - 1):
        s_hi, v_hi = _SOC_CURVE[i]
        s_lo, v_lo = _SOC_CURVE[i + 1]
        if soc >= s_lo:
            t = (s_hi - soc) / (s_hi - s_lo)
            return round(v_hi + t * (v_lo - v_hi), 4)
    return 0.0


def voltage_to_soc(v: float) -> float:
    """Inverse of soc_to_voltage — used when reading hardware voltage back."""
    v = max(0.0, min(1.0, v))
    for i in range(len(_SOC_CURVE) - 1):
        s_hi, v_hi = _SOC_CURVE[i]
        s_lo, v_lo = _SOC_CURVE[i + 1]
        if v >= v_lo:
            t = (v_hi - v) / (v_hi - v_lo)
            return round(s_hi + t * (s_lo - s_hi), 4)
    return 0.0


class SimBackend:
    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng.seed(seed)

    # ── Public API (same signatures as before) ─────────────────────────────────

    def compute_drain_delta_v(self, current_soc: float, amount: float) -> tuple[float, float]:
        """
        Drain `amount` SoC units from a cell.
        Returns (new_soc, observed_delta_v) — caller updates self._batteries.
        delta_v is in voltage units (INA219-observable), not SoC units.
        """
        actual = min(amount, current_soc)
        new_soc = max(0.0, current_soc - actual)
        true_delta_v = soc_to_voltage(current_soc) - soc_to_voltage(new_soc)
        observed_delta_v = self._ina219_noise(true_delta_v)
        return round(new_soc, 4), round(abs(observed_delta_v), 4)

    def compute_transfer_delta_v(
        self, from_soc: float, to_soc: float, amount: float
    ) -> tuple[float, float, float]:
        """
        Relay-routed energy transfer between two cells.
        Returns (new_from_soc, new_to_soc, observed_delta_v_on_sender).
        Efficiency degrades as sender SoC falls (higher internal resistance).
        """
        actual = min(amount, from_soc)
        efficiency = max(_MIN_EFFICIENCY, _BASE_EFFICIENCY - _EFFICIENCY_SLOPE * (1.0 - from_soc))
        new_from = max(0.0, from_soc - actual)
        new_to = min(1.0, to_soc + actual * efficiency)

        true_delta_v = soc_to_voltage(from_soc) - soc_to_voltage(new_from)
        observed_delta_v = self._ina219_noise(true_delta_v)
        return round(new_from, 4), round(new_to, 4), round(abs(observed_delta_v), 4)

    def get_urgency_from_sensor(self) -> float:
        """Sim version of HC-SR04 distance → urgency mapping."""
        return round(self._rng.uniform(0.1, 1.0), 2)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _ina219_noise(self, true_value: float) -> float:
        """
        Model INA219 ADC output for a known true voltage delta.
          - ADC quantization to nearest _ADC_RESOLUTION
          - Gaussian thermal + noise-floor jitter
          - Rare I2C spike (bit flip / glitch)
        """
        quantized = round(true_value / _ADC_RESOLUTION) * _ADC_RESOLUTION
        jitter = self._rng.gauss(0, _VOLTAGE_NOISE_STD)
        spike = self._rng.gauss(0, _SPIKE_STD) if self._rng.random() < _SPIKE_PROB else 0.0
        return quantized + jitter + spike
