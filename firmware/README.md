# AgentGrid Firmware

Two sketches — one per MCU type. Flash each with Arduino IDE.

## uno_agent.ino — Arduino Uno (primary sensor + LED driver)

Reads all three 18650 cell voltages via analog pins, drives the three status LEDs with brightness proportional to live voltage, and streams readings to the Pi over USB serial.

### Wiring

| Uno Pin | Connected to |
|---|---|
| A0 | Cell A (+) via 100Ω [+ 5.1V Zener A0→GND] |
| A1 | Cell B (+) via 100Ω [+ 5.1V Zener A1→GND] |
| A2 | Cell C (+) via 100Ω [+ 5.1V Zener A2→GND] |
| GND | Common ground rail (all Cell (−), Pi GND) |
| D9 | 220Ω → LED A → GND |
| D10 | 220Ω → LED B → GND |
| D11 | 220Ω → LED C → GND |
| USB | Raspberry Pi (serial link + power) |

**Common ground is mandatory.** All three cell negative terminals must tie to the same rail as Uno GND and Pi GND. The relay module must switch the **positive** side only (high-side switching).

### LED brightness

LED brightness maps linearly from 3.0V (dim, brightness=30) to 4.2V (full bright, brightness=255). The Uno updates LEDs autonomously — no Pi command needed. During a relay-fire transfer: the sender's LED dims, the receiver's LED brightens. This is the visible demo moment.

### Serial protocol

- **Uno → Pi** (every 50ms): `V 4.123 3.876 4.001` (voltages for A, B, C in order)
- **Pi → Uno**: nothing (LED control is autonomous)
- Baud: 115200

### Flash instructions

1. Install Arduino IDE + select board: **Arduino Uno**
2. Open `firmware/uno_agent.ino`
3. Select correct COM port / `/dev/ttyACM0`
4. Upload — no config needed, all agents handled by one sketch

---

## nodemcu_agent.ino — NodeMCU ESP8266 (agents A and B only)

Sends WiFi heartbeats to the Pi bridge every 5s. No longer drives LEDs (those moved to the Uno). Heartbeat endpoint is cosmetic — the bridge stub accepts and ignores them.

### NodeMCU config

Edit the top of the file for each board:

```cpp
const char* AGENT_ID = "A";  // change to "B" for the second board
```

### NodeMCU wiring

| NodeMCU Pin | Connected to |
|---|---|
| 3V3 | Power |
| GND | GND |
| D4 | Built-in LED (debug blink while connecting) |

Status LEDs are now driven by the Uno — no external LED wiring needed on NodeMCUs.

### NodeMCU flash instructions

1. Install Arduino IDE + ESP8266 board package
2. Board: NodeMCU 1.0 (ESP-12E)
3. Upload speed: 115200
4. Flash board #1 with `AGENT_ID = "A"`, board #2 with `AGENT_ID = "B"`

---

## Arduino Mega

Currently spare. Could be used to offload HC-SR04 ultrasonic sensing from the Pi (cleaner microsecond timing than Linux GPIO). No sketch written yet.
