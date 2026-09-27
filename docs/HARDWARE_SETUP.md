# Hardware & Electrical Component Setup Guide

This guide explains how to connect electrical components (solenoid door locks, 5V relays, servo actuators, status LEDs, piezo buzzers, and pushbuttons) to either a **Raspberry Pi** directly or an **Arduino / Microcontroller** via USB Serial.

---

## 🔌 System Architectures

You can deploy the biometric unlock system in three hardware configurations:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      OPTION 1: PURE RASPBERRY PI                        │
│                                                                         │
│   ┌──────────────┐   CSI / USB    ┌──────────────┐                      │
│   │ Camera Module│───────────────>│ Raspberry Pi │                      │
│   └──────────────┘                │ (Pi 4, 5, 3) │                      │
│                                   └───────┬──────┘                      │
│                                           │ GPIO Pins                   │
│                                           ▼                             │
│                  ┌─────────────────┬─────────────┬─────────────┐        │
│                  │ 5V Relay Module │ Status LEDs │ Buzzer & PB │        │
│                  └────────┬────────┴─────────────┴─────────────┘        │
│                           ▼                                             │
│                  ┌─────────────────┐                                    │
│                  │12V Solenoid Lock│                                    │
│                  └─────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│              OPTION 2: RASPBERRY PI / PC + ARDUINO MICROCONTROLLER      │
│                                                                         │
│   ┌──────────────┐                ┌──────────────┐                      │
│   │ Camera       │───────────────>│ Host (Pi/PC) │                      │
│   └──────────────┘                └───────┬──────┘                      │
│                                           │ USB Serial (115200 baud)    │
│                                           ▼                             │
│                                   ┌──────────────┐                      │
│                                   │ Arduino /    │                      │
│                                   │ ESP32 / Uno  │                      │
│                                   └───────┬──────┘                      │
│                                           │ Digital Pins                │
│                                           ▼                             │
│                     ┌──────────────┬──────────────┬──────────────┐      │
│                     │ Relay/Lock   │ Servo Motor  │ LEDs/Buzzer  │      │
│                     └──────────────┴──────────────┴──────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Option 1: Direct Raspberry Pi GPIO Wiring

### Pinout Mapping (BCM Numbers)

| Component | Function | Raspberry Pi BCM Pin | Physical Header Pin |
|---|---|---|---|
| **5V Relay (IN)** | Controls 12V lock | **BCM 18** | Pin 12 |
| **Green LED (+)** | Access Granted indicator | **BCM 23** | Pin 16 |
| **Red LED (+)** | Locked / Denied indicator | **BCM 24** | Pin 18 |
| **Blue LED (+)** | Sensing / Challenge indicator | **BCM 22** | Pin 15 |
| **Piezo Buzzer (+)** | Audio beep feedback | **BCM 25** | Pin 22 |
| **Push Button** | Manual exit switch | **BCM 17** | Pin 11 |
| **Relay / LED VCC** | 5V Power | **5V Power** | Pin 2 or 4 |
| **Ground** | Common Ground | **GND** | Pin 6, 9, 14, 20 |

### ⚠️ Crucial Electrical Protection: Solenoid Flyback Diode
Solenoids, maglocks, and inductive motors produce a high-voltage reverse EMF spike when switched off.
- **Always place a 1N4001 or 1N4007 diode** in reverse-bias across the solenoid terminals (Cathode with silver band to `+12V`, Anode to Ground/Relay Common).
- Power the 12V Solenoid from an external 12V DC power adapter; **never power a 12V lock from the Raspberry Pi 5V rail**.

### Launching on Raspberry Pi
```bash
# Basic GPIO unlock
python unlock.py --gpio

# Customize specific pins
python unlock.py --gpio --gpio-relay 18 --gpio-led-green 23 --gpio-buzzer 25 --unlock-duration 3.0
```

---

## 🤖 Option 2: Arduino / ESP32 Microcontroller Integration

Use an Arduino Uno, Nano, Mega, or ESP32 when you want dedicated real-time peripheral control or physical servo-based deadbolts.

### 1. Circuit Connections

| Component | Arduino Pin | Notes |
|---|---|---|
| **5V Relay Module (IN)** | **Pin 7** | Active-LOW relay module |
| **Servo Motor (Signal)** | **Pin 9** | Rotates 90° for mechanical deadbolt |
| **Green LED (+)** | **Pin 2** | With 220Ω resistor |
| **Red LED (+)** | **Pin 3** | With 220Ω resistor |
| **Blue LED (+)** | **Pin 4** | With 220Ω resistor |
| **Piezo Buzzer (+)** | **Pin 5** | Audible tones |
| **Exit Button** | **Pin 6** | Uses internal `INPUT_PULLUP` to GND |

### 2. Flashing Arduino Firmware
1. Open the Arduino IDE.
2. Open [`arduino/biometric_lock/biometric_lock.ino`](file:///arduino/biometric_lock/biometric_lock.ino).
3. Connect your Arduino via USB and select your Board and Port.
4. Click **Upload**.

### 3. Running with Python
Connect the Arduino USB cable to your Raspberry Pi or PC:
```bash
# On Linux / Raspberry Pi:
python unlock.py --serial-port /dev/ttyACM0

# On Windows:
python unlock.py --serial-port COM3
```

### Serial Protocol
The Python host communicates with the Arduino over USB Serial at **115200 baud** with standard ASCII commands:
- `UNLOCK:<name>:<duration_ms>` — Energizes relay, moves servo to 90°, lights Green LED, plays melody.
- `LOCK` — De-energizes relay, restores servo to 0°, lights Red LED.
- `BLINK:<count>` — Chirps buzzer and blinks Blue LED.
- `DENY` — Buzzes alarm tone and blinks Red LED 3 times.
- `EVENT:BUTTON` — Sent by Arduino when manual exit button is pressed.

---

## 🌐 Option 3: Smart Home Webhook Integration

Trigger Home Assistant, ESPHome, or a smart lock hub over HTTP:
```bash
python unlock.py --webhook-url "http://homeassistant.local:8123/api/webhook/biometric_door_unlock"
```

The system sends a JSON payload:
```json
{
  "event": "unlock",
  "user": "Alice",
  "duration": 3.0,
  "timestamp": 1700000000.0
}
```

---

## 🛠️ Auto-Start as a Headless Raspberry Pi Appliance

To run automatically when the Raspberry Pi boots:

```bash
# Copy systemd service file
sudo cp scripts/face-unlock.service /etc/systemd/system/

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable face-unlock.service
sudo systemctl start face-unlock.service

# Check live logs
journalctl -u face-unlock.service -f
```
