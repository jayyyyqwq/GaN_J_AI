/*
 * AgentGrid Arduino Uno Firmware
 *
 * Reads 3 cell voltages on A0/A1/A2 (agents A/B/C).
 * Streams "V <a> <b> <c>\n" over USB serial every 50ms.
 * Drives status LEDs on D9/D10/D11 with brightness proportional
 * to cell voltage (3.0V = dim, 4.2V = full bright).
 *
 * No serial input needed — LED control is fully autonomous.
 *
 * Wiring:
 *   A0 ─ 100Ω ─ Cell A (+)     [5.1V Zener A0→GND for spike protection]
 *   A1 ─ 100Ω ─ Cell B (+)
 *   A2 ─ 100Ω ─ Cell C (+)
 *   All Cell (−) → common GND rail → Uno GND → Pi GND
 *   D9  → 220Ω → LED A → GND
 *   D10 → 220Ω → LED B → GND
 *   D11 → 220Ω → LED C → GND
 *   USB → Raspberry Pi (powers Uno + serial link)
 */

// Analog input pins — one per agent cell
const int PIN_A = A0;
const int PIN_B = A1;
const int PIN_C = A2;

// PWM LED output pins (all PWM-capable on Uno)
const int LED_A = 9;
const int LED_B = 10;
const int LED_C = 11;

// ADC oversampling: average this many reads per report to reduce jitter
const int SAMPLES = 8;

// Stream interval in ms
const unsigned long STREAM_INTERVAL = 50;

// Voltage reference (Uno default: 5.0V)
const float VREF = 5.0f;

// ── Helpers ────────────────────────────────────────────────────────────────

float read_voltage(int pin) {
  long sum = 0;
  for (int i = 0; i < SAMPLES; i++) {
    sum += analogRead(pin);
  }
  return (sum / (float)SAMPLES) * (VREF / 1023.0f);
}

// 3.0V (dead) → brightness 30 (dim but visible)
// 4.2V (full) → brightness 255 (maximum)
uint8_t voltage_to_brightness(float v) {
  float pct = (v - 3.0f) / 1.2f;
  if (pct < 0.0f) pct = 0.0f;
  if (pct > 1.0f) pct = 1.0f;
  return (uint8_t)(30 + pct * 225);
}

// ── Setup & loop ───────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  pinMode(LED_A, OUTPUT);
  pinMode(LED_B, OUTPUT);
  pinMode(LED_C, OUTPUT);
  // Start at mid-brightness while settling
  analogWrite(LED_A, 128);
  analogWrite(LED_B, 128);
  analogWrite(LED_C, 128);
}

unsigned long last_stream = 0;

void loop() {
  unsigned long now = millis();
  if (now - last_stream < STREAM_INTERVAL) return;
  last_stream = now;

  float vA = read_voltage(PIN_A);
  float vB = read_voltage(PIN_B);
  float vC = read_voltage(PIN_C);

  // Update LEDs autonomously — no Pi command needed
  analogWrite(LED_A, voltage_to_brightness(vA));
  analogWrite(LED_B, voltage_to_brightness(vB));
  analogWrite(LED_C, voltage_to_brightness(vC));

  // Stream to Pi: "V 4.123 3.876 4.001"
  Serial.print("V ");
  Serial.print(vA, 3);
  Serial.print(" ");
  Serial.print(vB, 3);
  Serial.print(" ");
  Serial.println(vC, 3);
}
