/*
 * AgentGrid NodeMCU Firmware
 * One copy per agent node (A, B, C).
 * Set AGENT_ID before flashing.
 *
 * - Heartbeat POST to Pi bridge every 5s
 * - LED brightness controlled by Pi via /led endpoint
 * - Reports local VCC as a coarse backup voltage signal
 */

#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <ESP8266WebServer.h>

// ── Config — set per board before flashing ─────────────────────────
const char* AGENT_ID       = "A";         // "A", "B", or "C"
const char* SSID           = "AgentGrid";  // Pi hotspot SSID
const char* PASSWORD        = "agentgrid2026";
const char* BRIDGE_HOST    = "192.168.4.1";
const int   BRIDGE_PORT    = 8001;
// ───────────────────────────────────────────────────────────────────

const int LED_PIN    = D4;  // NodeMCU built-in LED (active LOW)
const int STATUS_PIN = D2;  // external white LED + 220Ω resistor

ESP8266WebServer server(80);
WiFiClient wifiClient;

int ledBrightness = 512;  // 0-1023 PWM (0 = off, 1023 = full)

void setup() {
  Serial.begin(115200);
  pinMode(STATUS_PIN, OUTPUT);
  analogWrite(STATUS_PIN, ledBrightness);

  WiFi.begin(SSID, PASSWORD);
  Serial.print("Connecting to ");
  Serial.print(SSID);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    analogWrite(STATUS_PIN, (millis() / 250) % 2 == 0 ? 0 : 128);  // blink while connecting
  }
  Serial.println("\nConnected. IP: " + WiFi.localIP().toString());
  analogWrite(STATUS_PIN, ledBrightness);

  // Endpoint: Pi sets LED brightness
  server.on("/led", HTTP_POST, []() {
    if (server.hasArg("brightness")) {
      ledBrightness = constrain(server.arg("brightness").toInt(), 0, 1023);
      analogWrite(STATUS_PIN, ledBrightness);
      server.send(200, "application/json", "{\"status\":\"ok\"}");
    } else {
      server.send(400, "application/json", "{\"error\":\"missing brightness\"}");
    }
  });

  // Endpoint: health check
  server.on("/health", HTTP_GET, []() {
    String vcc = String(ESP.getVcc() / 1000.0, 3);
    String body = "{\"agent\":\"" + String(AGENT_ID) + "\",\"vcc\":" + vcc + "}";
    server.send(200, "application/json", body);
  });

  server.begin();
}

unsigned long lastHeartbeat = 0;

void loop() {
  server.handleClient();

  // Heartbeat to bridge every 5s
  if (millis() - lastHeartbeat > 5000) {
    lastHeartbeat = millis();
    sendHeartbeat();
  }
}

void sendHeartbeat() {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  String url = "http://" + String(BRIDGE_HOST) + ":" + String(BRIDGE_PORT) + "/heartbeat";
  String body = "{\"agent\":\"" + String(AGENT_ID) + "\",\"vcc\":" + String(ESP.getVcc() / 1000.0, 3) + "}";
  http.begin(wifiClient, url);
  http.addHeader("Content-Type", "application/json");
  http.POST(body);
  http.end();
}
