"""
Hardware driver for the Raspberry Pi bridge.

Wraps:
  - INA219 current/voltage sensors (I2C, one per agent cell)
  - 4-channel relay module (GPIO, routes power between cells)
  - HC-SR04 ultrasonic sensor (GPIO, urgency injection)

INA219 I2C addresses (set via A0/A1 pins on the board):
  Agent A: 0x40 (default)
  Agent B: 0x41
  Agent C: 0x44
"""
from __future__ import annotations

import time
from typing import Optional

# ── Hardware import guard ──────────────────────────────────────────────────
# These imports only work on the Pi; the sim_backend handles the Colab/HF case.
try:
    import smbus2
    import RPi.GPIO as GPIO
    _HW_AVAILABLE = True
except ImportError:
    _HW_AVAILABLE = False

# INA219 register map
_INA219_REG_BUS_VOLTAGE = 0x02
_INA219_REG_CONFIG = 0x00

# Relay GPIO pins (BCM numbering)
_RELAY_PINS: dict[tuple[str, str], int] = {
    ("A", "B"): 17,
    ("B", "A"): 17,
    ("A", "C"): 27,
    ("C", "A"): 27,
    ("B", "C"): 22,
    ("C", "B"): 22,
}

# HC-SR04 pins
_TRIG_PIN = 23
_ECHO_PIN = 24

# INA219 I2C addresses per agent
_INA219_ADDR: dict[str, int] = {"A": 0x40, "B": 0x41, "C": 0x44}
_BUS_NUM = 1  # Pi I2C bus 1

# Calibration: LSB = 4 mV for INA219 bus voltage register
_BUS_VOLTAGE_LSB = 0.004


class HardwareDriver:
    def __init__(self) -> None:
        self._prev_voltages: dict[str, float] = {"A": 0.0, "B": 0.0, "C": 0.0}
        if not _HW_AVAILABLE:
            return  # running in dev/CI, no hardware
        self._bus = smbus2.SMBus(_BUS_NUM)
        GPIO.setmode(GPIO.BCM)
        for pin in set(_RELAY_PINS.values()):
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)  # HIGH = relay open
        GPIO.setup(_TRIG_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(_ECHO_PIN, GPIO.IN)

    def read_voltage(self, agent_id: str) -> tuple[float, float]:
        """Returns (current_voltage, delta_v_since_last_read)."""
        if not _HW_AVAILABLE:
            return (1.0, 0.0)
        addr = _INA219_ADDR[agent_id]
        raw = self._bus.read_word_data(addr, _INA219_REG_BUS_VOLTAGE)
        # INA219 returns big-endian 16-bit
        raw = ((raw & 0xFF) << 8) | ((raw >> 8) & 0xFF)
        voltage = ((raw >> 3) * _BUS_VOLTAGE_LSB)
        delta = self._prev_voltages[agent_id] - voltage
        self._prev_voltages[agent_id] = voltage
        return (round(voltage, 4), round(delta, 4))

    def fire_relay(self, from_agent: str, to_agent: str, amount: float) -> float:
        """
        Fire the relay between two cells for long enough to transfer `amount` energy units.
        Returns measured delta_v on sender's cell.
        """
        if not _HW_AVAILABLE:
            return amount * 0.08  # sim-calibrated approximation
        pin = _RELAY_PINS.get((from_agent, to_agent))
        if pin is None:
            return 0.0
        v_before, _ = self.read_voltage(from_agent)
        # Fire relay: LOW = relay closed (power flows)
        duration = amount * 2.5  # seconds per energy unit (calibrate Day 2)
        GPIO.output(pin, GPIO.LOW)
        time.sleep(min(duration, 5.0))  # cap at 5s for safety
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.1)  # let voltage settle
        v_after, _ = self.read_voltage(from_agent)
        return round(v_before - v_after, 4)

    def read_ultrasonic(self) -> float:
        """Returns distance in cm from HC-SR04."""
        if not _HW_AVAILABLE:
            return 50.0  # default: mid-range
        GPIO.output(_TRIG_PIN, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(_TRIG_PIN, GPIO.LOW)
        pulse_start = time.time()
        pulse_end = time.time()
        timeout = time.time() + 0.1
        while GPIO.input(_ECHO_PIN) == 0:
            pulse_start = time.time()
            if pulse_start > timeout:
                return 100.0
        while GPIO.input(_ECHO_PIN) == 1:
            pulse_end = time.time()
            if pulse_end > timeout:
                return 100.0
        duration = pulse_end - pulse_start
        distance = (duration * 34300) / 2  # cm
        return round(distance, 1)

    def reset_all(self) -> None:
        """Open all relays. Charging handled externally via TP4056."""
        if not _HW_AVAILABLE:
            return
        for pin in set(_RELAY_PINS.values()):
            GPIO.output(pin, GPIO.HIGH)

    def cleanup(self) -> None:
        if _HW_AVAILABLE:
            GPIO.cleanup()
