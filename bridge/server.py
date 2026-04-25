"""
Raspberry Pi bridge server.

Runs on the Pi. Exposes hardware (INA219, relay matrix, HC-SR04) over HTTP
so the env running on Colab/HF Spaces can close the hardware loop.

Start: uvicorn bridge.server:app --host 0.0.0.0 --port 7000
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from .hardware import HardwareDriver
except ImportError:
    from hardware import HardwareDriver  # type: ignore[no-redef]

app = FastAPI(title="AgentGrid Bridge", version="0.1.0")
hw = HardwareDriver()

CALIBRATION: dict = json.loads(
    (Path(__file__).parent / "calibration.json").read_text()
)


class RelayFireRequest(BaseModel):
    # Field names match what agentgrid_environment._execute_energy_transfer sends
    from_agent: str   # "A" | "B" | "C"
    to_agent: str
    amount: float     # energy units


class VoltageReading(BaseModel):
    agent_id: str
    voltage: float
    delta_v: float


@app.post("/reset")
async def reset_hardware() -> dict:
    """Open all relays and return to known state."""
    hw.reset_all()
    return {"status": "ok", "message": "Hardware reset complete."}


@app.get("/voltage/{agent_id}")
async def get_voltage(agent_id: str) -> VoltageReading:
    """Read INA219 voltage for one agent's cell."""
    if agent_id not in ("A", "B", "C"):
        raise HTTPException(status_code=400, detail=f"Unknown agent: {agent_id}")
    v, delta = hw.read_voltage(agent_id)
    return VoltageReading(agent_id=agent_id, voltage=v, delta_v=delta)


@app.post("/relay/fire")
async def fire_relay(req: RelayFireRequest) -> dict:
    """
    Route power from one cell to another via relay matrix.
    Returns actual delta_v measured by INA219 on sender's cell.
    """
    if req.from_agent not in ("A", "B", "C") or req.to_agent not in ("A", "B", "C"):
        raise HTTPException(status_code=400, detail="Invalid agent IDs.")
    delta_v = hw.fire_relay(req.from_agent, req.to_agent, req.amount)
    return {"status": "ok", "delta_v": delta_v, "amount": req.amount}


@app.post("/heartbeat")
async def heartbeat(payload: dict) -> dict:
    """Receive NodeMCU heartbeat (agent_id + VCC)."""
    return {"status": "ok", "ts": time.time()}


@app.get("/sensor/urgency")
async def get_urgency() -> dict:
    """Read HC-SR04 distance and convert to urgency scalar [0, 1]."""
    distance_cm = hw.read_ultrasonic()
    urgency = max(0.0, min(1.0, 1.0 - (distance_cm / 100.0)))
    return {"distance_cm": round(distance_cm, 1), "urgency": round(urgency, 2)}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "ts": time.time()}
