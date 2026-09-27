# Face / Eye (Blink) Unlock Demo

[![CI](https://github.com/jackyphuti/Face-Eye-blink-unlock-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/jackyphuti/Face-Eye-blink-unlock-demo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A robust computer vision authentication system demonstrating face recognition paired with active eye-blink and head-pose liveness detection. Powered by OpenCV, dlib / `face_recognition`, and optional Google MediaPipe Face Mesh.

Supports physical embedded biometric installations on **Raspberry Pi** (direct GPIO relays, solenoid locks, LEDs, buzzers) and **Arduino / Microcontrollers** (USB Serial actuators and servos).

---

## 🌟 Key Features

- **Embedded & Hardware Biometrics**:
  - **Raspberry Pi GPIO**: Directly controls 5V relays, 12V solenoid door strikes, status LEDs (Green/Red/Blue), piezo buzzers, and manual exit buttons.
  - **Arduino / Microcontroller Integration**: Transmits serial commands at 115200 baud to drive servos (mechanical deadbolts), relays, and indicators via [`arduino/biometric_lock/biometric_lock.ino`](file:///arduino/biometric_lock/biometric_lock.ino).
  - **Smart Home Webhooks**: Dispatches HTTP POST notifications to Home Assistant, Node-RED, or ESP32 webhooks.
  - **Automated Pi Setup & Systemd Appliance**: One-click install script ([`scripts/install_pi.sh`](file:///scripts/install_pi.sh)) and auto-start service ([`scripts/face-unlock.service`](file:///scripts/face-unlock.service)).
- **Accurate Face Matching**: Employs closest Euclidean distance matching (`np.argmin(face_distance)`) with configurable distance tolerances rather than naive first-match thresholding.
- **Robust Liveness Verification**:
  - **Blink Duration Filter**: Rejects static photo spoof attacks (eyes closed photo) and single-frame sensor noise glitches by validating physiological blink duration (50ms–1500ms).
  - **Debounced Counters & EMA Smoothing**: Eliminates false blinks and jitter with Exponential Moving Average (EMA) filtering.
  - **Head Pose Estimation**: Optional geometric yaw estimation to verify user movement ("FORWARD", "LEFT", "RIGHT").
- **Interactive Challenge-Response Mode**:
  - Requires the user to perform randomized actions (e.g. blink *N* times within a countdown window).
  - Includes isolated per-challenge counters, failure cooldowns, and automatic challenge rotation.
- **Flexible Enrollment Pipeline**:
  - Single-shot or multi-shot enrollment with normalized vector averaging for high stability.
  - Interactive mode (`--interactive`): manually press SPACE to capture each shot when ready.
  - Direct file enrollment (`--image photo.jpg`): enroll without a webcam.
  - Image quality checks (flags blurry or underexposed photos).
  - Safe sanitization against filesystem path traversal.
- **Face Management CLI (`manage_faces.py`)**:
  - `list`, `info`, `remove`, `rename`, `clear`, `export` (zip archive), and `import` commands.
  - `--json` format output for automation and scripting.
- **Real-Time HUD & Audio Feedback**:
  - On-screen EAR gauge meter with live threshold indicator.
  - Real-time FPS counter, styled facial bounding boxes, and status banners.
  - Audio beeps for blink and unlock events.
  - Snapshot capture (`--save-unlock-shots <dir>`) and structured event logging (`--log unlocks.csv` / `unlocks.json`).
  - Headless mode (`--headless`) for server, container, and CI environments.
  - JSON configuration file support (`--config` / `--save-config`).

---

## 📁 Repository Structure

- [`ear.py`](file:///ear.py) — Core Eye Aspect Ratio (EAR) formulas, Euclidean distance, and EMA smoothing.
- [`liveness.py`](file:///liveness.py) — `BlinkDetector`, `ChallengeManager`, and `HeadPoseEstimator`.
- [`hardware.py`](file:///hardware.py) — Raspberry Pi GPIO, Arduino Serial, and Webhook hardware controllers.
- [`enroll.py`](file:///enroll.py) — Enrolls face profiles from webcam or image files.
- [`unlock.py`](file:///unlock.py) — Live authentication loop with recognition, liveness checks, and HUD overlay.
- [`manage_faces.py`](file:///manage_faces.py) — Profile inspection, renaming, removal, and archive import/export.
- [`config.py`](file:///config.py) — Configuration dataclass supporting JSON profiles.
- [`utils.py`](file:///utils.py) — Filesystem sanitization, sound alerts, structured logging, and HUD drawing.
- [`arduino/`](file:///arduino) — Arduino C++ firmware sketch for physical lock, servo, and LED actuation.
- [`scripts/`](file:///scripts) — Automated Raspberry Pi installer and systemd service files.
- [`docs/HARDWARE_SETUP.md`](file:///docs/HARDWARE_SETUP.md) — Electrical component schematics and wiring guide.
- [`tests/`](file:///tests) — Comprehensive unit test suite (runs in CI without requiring webcam/native models).

---

## ⚡ Hardware & Electrical Components Quick Start

For detailed wiring diagrams, schematics, and safety instructions (including flyback diode protection for solenoids), see the [Hardware Setup Guide](file:///docs/HARDWARE_SETUP.md).

### 🍓 Option 1: Raspberry Pi Installation

Run the automated installer on Raspberry Pi OS (Pi 3, 4, 5, or Zero 2W):

```bash
chmod +x scripts/install_pi.sh
./scripts/install_pi.sh
```

Run with direct GPIO relay control (Relay on BCM 18, LEDs on BCM 23/24, Buzzer on BCM 25):
```bash
python unlock.py --gpio
```

To configure as a headless auto-starting smart door appliance:
```bash
sudo cp scripts/face-unlock.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now face-unlock.service
```

---

### 🤖 Option 2: Arduino Microcontroller Setup

1. Open [`arduino/biometric_lock/biometric_lock.ino`](file:///arduino/biometric_lock/biometric_lock.ino) in the Arduino IDE.
2. Upload the sketch to your Arduino Uno, Nano, Mega, or ESP32.
3. Connect the Arduino via USB to your Raspberry Pi or PC.
4. Launch unlock monitoring:
   ```bash
   # On Linux / Raspberry Pi:
   python unlock.py --serial-port /dev/ttyACM0

   # On Windows:
   python unlock.py --serial-port COM3
   ```

---

## 🚀 Standard Quick Start

### 1. Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note on Prerequisites:**
> `face_recognition` depends on `dlib`. On Linux, install prerequisites first:
> `sudo apt update && sudo apt install cmake build-essential libgtk-3-dev libboost-all-dev`

### 2. Enroll a User

**Single-shot (countdown timer):**
```bash
python enroll.py --name Alice
```

**Multi-shot averaging (press SPACE for each shot):**
```bash
python enroll.py --name Alice --shots 3 --interactive
```

**Enroll directly from an image file:**
```bash
python enroll.py --name Bob --image /path/to/bob_photo.jpg
```

### 3. Manage Profiles

```bash
# List all enrolled faces
python manage_faces.py list

# Inspect detailed metadata (shape, date, snapshot)
python manage_faces.py info Alice

# Rename an enrolled face
python manage_faces.py rename Alice Alicia

# Export database to zip archive
python manage_faces.py export backup.zip

# Import profiles from zip
python manage_faces.py import backup.zip

# Remove an enrolled face
python manage_faces.py remove Bob
```

### 4. Run Unlock

**Standard unlock (1 blink required):**
```bash
python unlock.py
```

**Require 2 blinks with MediaPipe Face Mesh:**
```bash
python unlock.py --use-mediapipe --blinks 2 --ear 0.20
```

**Challenge-Response Mode with logging and automatic exit on unlock:**
```bash
python unlock.py --challenge --challenge-min 2 --challenge-max 3 --log unlocks.json --exit-on-unlock
```

**Save timestamped snapshots of authorized entries:**
```bash
python unlock.py --save-unlock-shots ./authorized_snapshots
```

---

## ⚙️ Command-Line Options Reference

### `unlock.py`
| Argument | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera device index |
| `--scale` | `0.5` | Processing downscale factor (0.1 - 1.0) |
| `--ear` | `0.21` | EAR threshold below which eye is considered closed |
| `--consec` | `2` | Consecutive frames below EAR to register a blink |
| `--blinks` | `1` | Total blinks required to unlock |
| `--tolerance` | `0.50` | Face recognition tolerance (lower = stricter) |
| `--challenge` | `False` | Enable randomized challenge-response |
| `--challenge-min` | `1` | Minimum blinks for challenge |
| `--challenge-max` | `3` | Maximum blinks for challenge |
| `--challenge-timeout`| `8.0` | Seconds allowed to satisfy challenge |
| `--head-pose` | `False` | Enable head orientation estimation |
| `--use-mediapipe` | `False` | Use MediaPipe Face Mesh landmarks |
| `--gpio` | `False` | Enable direct Raspberry Pi GPIO hardware control |
| `--gpio-relay` | `18` | BCM pin for electronic door strike / relay |
| `--gpio-led-green` | `23` | BCM pin for Green Granted LED |
| `--gpio-led-red` | `24` | BCM pin for Red Locked/Denied LED |
| `--gpio-buzzer` | `25` | BCM pin for Piezo Buzzer |
| `--gpio-button` | `17` | BCM pin for Manual Exit Button |
| `--unlock-duration` | `3.0` | Seconds to keep relay energized |
| `--serial-port` | `None` | Serial port for Arduino / ESP32 (e.g. `/dev/ttyACM0` or `COM3`) |
| `--webhook-url` | `None` | HTTP URL for smart home webhook dispatch |
| `--log` | `None` | Append events to file (`.csv` or `.json`) |
| `--save-unlock-shots`| `None` | Directory to save snapshots of unlocks |
| `--headless` | `False` | Run without GUI window |
| `--sound` / `--no-sound`| `True` | Enable/disable audio feedback beeps |
| `--exit-on-unlock` | `False` | Terminate script upon successful unlock |
| `--config` | `None` | Load settings from JSON file |
| `--save-config` | `None` | Save settings to JSON file and exit |

---

## 🧪 Running Tests

Run the complete test suite:
```bash
pytest -v tests/
```

All 38 unit tests run in CI environments without requiring physical webcams, Raspberry Pi GPIO, or heavy GPU dependencies.

---

## 🛡️ Security Disclaimer

This software is an educational demonstration. In production environments, face and blink liveness checks should be complemented with multi-modal biometrics (e.g. depth/infrared sensors, thermal cameras, 3D structured light) and hardware-backed credential storage to resist advanced spoofing attacks (e.g., deepfakes, 3D masks, video replay).

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
