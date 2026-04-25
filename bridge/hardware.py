"""
Hardware driver for the Raspberry Pi bridge.

Wraps:
  - Arduino Uno analog pin voltage sensor (USB serial, one reading per agent cell)
  - 4-channel relay module (GPIO, routes power between cells)
  - HC-SR04 ultrasonic sensor (GPIO, urgency injection)

Uno streams "V <a> <b> <c>\\n" over USB serial every 50ms.
A background thread reads and caches the latest values.
Pi → Uno: no commands needed (LED brightness is autonomous on the Uno).

Serial port default: /dev/ttyACM0 — override with env var AGENTGRID_UNO_PORT.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

# ── Hardware import guard ──────────────────────────────────────────────────
# GPIO only works on the Pi; serial only works when Uno is connected.
try:
    import RPi.GPIO as GPIO  # type: ignore[import-untyped]
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False

try:
    import serial as _serial  # type: ignore[import-untyped]
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False

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

# Uno USB serial defaults
_UNO_PORT = os.environ.get("AGENTGRID_UNO_PORT", "/dev/ttyACM0")
_UNO_BAUD = 115200
_UNO_TIMEOUT = 0.1  # seconds

# Calibration: volts drop per energy unit transferred
_VOLTS_PER_ENERGY_UNIT = 0.08


class _SerialDriver:
    """Background thread that reads the Uno voltage stream and caches values."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._voltages: dict[str, float] = {"A": 3.7, "B": 3.7, "C": 3.7}
        self._running = False
        self._thread: Optional[threading.Thread] = None

        if not _SERIAL_AVAILABLE:
            return
        try:
            self._port = _serial.Serial(_UNO_PORT, _UNO_BAUD, timeout=_UNO_TIMEOUT)
            # Drain any stale buffer from a previous session
            self._port.reset_input_buffer()
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
        except Exception:
            self._running = False

    def _read_loop(self) -> None:
        while self._running:
            try:
                raw = self._port.readline().decode("ascii", errors="ignore").strip()
                if not raw.startswith("V "):
                    continue
                parts = raw.split()
                if len(parts) != 4:
                    continue
                vA, vB, vC = float(parts[1]), float(parts[2]), float(parts[3])
                with self._lock:
                    self._voltages = {"A": vA, "B": vB, "C": vC}
            except Exception:
                time.sleep(0.01)

    def get(self, agent_id: str) -> float:
        with self._lock:
            return self._voltages.get(agent_id, 3.7)

    def stop(self) -> None:
        self._running = False


class HardwareDriver:
    def __init__(self) -> None:
        self._prev_voltages: dict[str, float] = {"A": 0.0, "B": 0.0, "C": 0.0}
        self._serial = _SerialDriver()

        if not _GPIO_AVAILABLE:
            return
        GPIO.setmode(GPIO.BCM)
        for pin in set(_RELAY_PINS.values()):
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)  # HIGH = relay open
        GPIO.setup(_TRIG_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(_ECHO_PIN, GPIO.IN)

    def read_voltage(self, agent_id: str) -> tuple[float, float]:
        """Returns (current_voltage, delta_v_since_last_read)."""
        voltage = self._serial.get(agent_id)
        delta = self._prev_voltages[agent_id] - voltage
        self._prev_voltages[agent_id] = voltage
        return (round(voltage, 4), round(delta, 4))

    def fire_relay(self, from_agent: str, to_agent: str, amount: float) -> float:
        """
        Fire the relay between two cells for long enough to transfer `amount` energy units.
        Returns measured delta_v on sender's cell.
        """
        if not _GPIO_AVAILABLE:
            return amount * _VOLTS_PER_ENERGY_UNIT  # sim-calibrated approximation
        pin = _RELAY_PINS.get((from_agent, to_agent))
        if pin is None:
            return 0.0
        v_before = self._serial.get(from_agent)
        # Fire relay: LOW = relay closed (power flows). HIGH = relay open.
        duration = amount * 2.5  # seconds per energy unit (calibrate Phase 4)
        GPIO.output(pin, GPIO.LOW)
        time.sleep(min(duration, 5.0))  # cap at 5s for safety
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.1)  # let voltage settle before reading
        v_after = self._serial.get(from_agent)
        self._prev_voltages[from_agent] = v_after
        return round(v_before - v_after, 4)

    def read_ultrasonic(self) -> float:
        """Returns distance in cm from HC-SR04."""
        if not _GPIO_AVAILABLE:
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
        distance = ((pulse_end - pulse_start) * 34300) / 2
        return round(distance, 1)

    def reset_all(self) -> None:
        """Open all relays. Charging handled externally via TP4056."""
        if not _GPIO_AVAILABLE:
            return
        for pin in set(_RELAY_PINS.values()):
            GPIO.output(pin, GPIO.HIGH)

    def cleanup(self) -> None:
        self._serial.stop()
        if _GPIO_AVAILABLE:
            GPIO.cleanup()
