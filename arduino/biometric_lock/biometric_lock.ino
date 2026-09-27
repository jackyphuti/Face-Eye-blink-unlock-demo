/*
 * Arduino Biometric Lock Controller
 * 
 * Works with Face-Eye-blink-unlock-demo running on a Raspberry Pi or PC.
 * Receives USB Serial commands to actuate relays, servos, LEDs, and buzzers.
 * 
 * Circuit Connections:
 *   - Pin 7: 5V Relay Module (IN) -> Controls 12V Solenoid Door Strike
 *   - Pin 9: Servo Motor Signal (Yellow/Orange wire)
 *   - Pin 2: Green LED (via 220 ohm resistor) -> Unlocked
 *   - Pin 3: Red LED (via 220 ohm resistor) -> Locked / Denied
 *   - Pin 4: Blue LED (via 220 ohm resistor) -> Sensing / Face Detected
 *   - Pin 5: Piezo Buzzer (+) -> Audible feedback
 *   - Pin 6: Push Button (Between Pin 6 and GND) -> Manual Exit Button
 * 
 * Serial Commands (115200 baud, newline terminated):
 *   - UNLOCK:<name>:<duration_ms>  (e.g., UNLOCK:Alice:3000)
 *   - LOCK
 *   - BLINK:<count>
 *   - DENY
 *   - STATUS:SENSING
 */

#include <Servo.h>

// Pin definitions
const int PIN_RELAY      = 7;   // Active-LOW relay module
const int PIN_SERVO      = 9;   // Servo lock actuator
const int PIN_LED_GREEN  = 2;   // Granted LED
const int PIN_LED_RED    = 3;   // Locked LED
const int PIN_LED_BLUE   = 4;   // Sensing LED
const int PIN_BUZZER     = 5;   // Piezo Buzzer
const int PIN_BUTTON     = 6;   // Manual Exit Pushbutton (INPUT_PULLUP)

Servo lockServo;
const int SERVO_LOCKED_POS   = 0;
const int SERVO_UNLOCKED_POS = 90;

// State management
bool isUnlocked = false;
unsigned long unlockStartTime = 0;
unsigned long unlockDuration = 3000;

// Button debounce
int lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

String inputBuffer = "";

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000); // Wait for serial on USB boards

  pinMode(PIN_RELAY, OUTPUT);
  digitalWrite(PIN_RELAY, HIGH); // Relay OFF (Active-LOW)

  pinMode(PIN_LED_GREEN, OUTPUT);
  pinMode(PIN_LED_RED, OUTPUT);
  pinMode(PIN_LED_BLUE, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_BUTTON, INPUT_PULLUP);

  lockServo.attach(PIN_SERVO);
  setLocked();

  Serial.println("READY: Arduino Biometric Lock Controller Initialized");
}

void loop() {
  // 1. Process serial commands from Raspberry Pi / PC
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += c;
    }
  }

  // 2. Handle automatic non-blocking relocking
  if (isUnlocked && (millis() - unlockStartTime >= unlockDuration)) {
    setLocked();
  }

  // 3. Monitor manual exit button
  int reading = digitalRead(PIN_BUTTON);
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading == LOW && !isUnlocked) {
      Serial.println("EVENT:BUTTON");
      setUnlocked("Manual_Button", 3000);
    }
  }
  lastButtonState = reading;
}

void processCommand(String cmd) {
  cmd.trim();
  if (cmd.startsWith("UNLOCK")) {
    // Expected format: UNLOCK:<name>:<duration_ms>
    int firstColon = cmd.indexOf(':');
    int secondColon = cmd.indexOf(':', firstColon + 1);

    String name = "User";
    unsigned long dur = 3000;

    if (firstColon != -1) {
      if (secondColon != -1) {
        name = cmd.substring(firstColon + 1, secondColon);
        dur = cmd.substring(secondColon + 1).toInt();
      } else {
        name = cmd.substring(firstColon + 1);
      }
    }
    if (dur <= 0) dur = 3000;
    setUnlocked(name, dur);

  } else if (cmd == "LOCK") {
    setLocked();

  } else if (cmd.startsWith("BLINK")) {
    int colon = cmd.indexOf(':');
    int count = 1;
    if (colon != -1) {
      count = cmd.substring(colon + 1).toInt();
    }
    chirpBlink(count);

  } else if (cmd == "DENY") {
    accessDenied();

  } else if (cmd == "STATUS:SENSING") {
    digitalWrite(PIN_LED_BLUE, HIGH);
  }
}

void setUnlocked(String name, unsigned long dur) {
  isUnlocked = true;
  unlockStartTime = millis();
  unlockDuration = dur;

  // Actuate hardware
  digitalWrite(PIN_RELAY, LOW);   // Energize relay (solenoid opens)
  lockServo.write(SERVO_UNLOCKED_POS);

  digitalWrite(PIN_LED_GREEN, HIGH);
  digitalWrite(PIN_LED_RED, LOW);
  digitalWrite(PIN_LED_BLUE, LOW);

  // Victory chime on buzzer (tones: 523Hz (C5), 659Hz (E5), 784Hz (G5))
  tone(PIN_BUZZER, 523, 100);
  delay(120);
  tone(PIN_BUZZER, 659, 100);
  delay(120);
  tone(PIN_BUZZER, 784, 250);

  Serial.print("ACK:UNLOCKED:");
  Serial.print(name);
  Serial.print(":");
  Serial.println(dur);
}

void setLocked() {
  isUnlocked = false;

  digitalWrite(PIN_RELAY, HIGH);  // Release relay (solenoid locked)
  lockServo.write(SERVO_LOCKED_POS);

  digitalWrite(PIN_LED_GREEN, LOW);
  digitalWrite(PIN_LED_RED, HIGH);
  digitalWrite(PIN_LED_BLUE, LOW);

  Serial.println("ACK:LOCKED");
}

void chirpBlink(int count) {
  digitalWrite(PIN_LED_BLUE, HIGH);
  tone(PIN_BUZZER, 1200, 50);
  delay(60);
  digitalWrite(PIN_LED_BLUE, LOW);
  Serial.print("ACK:BLINK:");
  Serial.println(count);
}

void accessDenied() {
  digitalWrite(PIN_LED_BLUE, LOW);
  digitalWrite(PIN_LED_GREEN, LOW);

  for (int i = 0; i < 3; i++) {
    digitalWrite(PIN_LED_RED, HIGH);
    tone(PIN_BUZZER, 220, 150);
    delay(200);
    digitalWrite(PIN_LED_RED, LOW);
    delay(100);
  }
  digitalWrite(PIN_LED_RED, HIGH);
  Serial.println("ACK:DENIED");
}
