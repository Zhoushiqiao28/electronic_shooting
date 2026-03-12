const int SHOT_PIN = 7;
const unsigned long DEBOUNCE_MS = 30;

bool lastStableState = HIGH;
bool lastReading = HIGH;
unsigned long lastChangeMs = 0;

void setup() {
  Serial.begin(115200);
  pinMode(SHOT_PIN, INPUT_PULLUP);
}

void loop() {
  bool reading = digitalRead(SHOT_PIN);

  if (reading != lastReading) {
    lastChangeMs = millis();
    lastReading = reading;
  }

  if (millis() - lastChangeMs >= DEBOUNCE_MS && reading != lastStableState) {
    lastStableState = reading;

    if (lastStableState == LOW) {
      Serial.println("SHOT");
    }
  }
}
