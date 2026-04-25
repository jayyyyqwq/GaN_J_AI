# NodeMCU Firmware

Flash `nodemcu_agent.ino` onto each of the 3 NodeMCU ESP8266 boards.

## Before flashing

Edit the top of the file for each board:

```cpp
const char* AGENT_ID = "A";  // change to "B" and "C" for the other two
```

All other config (SSID, PASSWORD, BRIDGE_HOST) is shared.

## Wiring

| NodeMCU Pin | Connected to        |
|-------------|---------------------|
| D2          | White LED + 220Ω → GND |
| D4          | Built-in LED (debug) |
| 3V3         | INA219 VCC          |
| GND         | INA219 GND + LED GND |
| D1 (SCL)    | INA219 SCL          |
| D2 (SDA)    | INA219 SDA          |

## Flash instructions

1. Install Arduino IDE + ESP8266 board package
2. Board: NodeMCU 1.0 (ESP-12E)
3. Upload speed: 115200
4. Flash each board with the correct AGENT_ID set
