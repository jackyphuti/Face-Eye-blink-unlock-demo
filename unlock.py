#!/usr/bin/env python3
"""Live face and eye-blink unlock demo with robust liveness verification.

Features:
- Accurate best-match face recognition (closest Euclidean distance).
- Eye Aspect Ratio (EAR) blink detection with duration filtering (anti-spoofing).
- Interactive randomized challenge-response (blinks and head turns).
- Mediapipe FaceMesh or face_recognition 68-landmarks support.
- Hardware actuators: Raspberry Pi GPIO (Relays, LEDs, Buzzers) & Arduino Serial.
- Modern HUD overlay with EAR meter, FPS counter, and status banners.
- Audio feedback (beeps/alerts) for unlock events.
- Configuration file support (JSON load/save).
- Headless mode and unlock snapshot saving.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from config import UnlockConfig
from ear import eye_aspect_ratio
from hardware import HardwareConfig, HardwareManager
from liveness import BlinkDetector, ChallengeManager, HeadPoseEstimator
from utils import (
    draw_ear_bar,
    draw_hud_box,
    log_unlock_event,
    play_sound,
    sanitize_name,
)

# Optional dependencies
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    import face_recognition
    HAS_FACE_REC = True
except ImportError:
    face_recognition = None
    HAS_FACE_REC = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    mp = None
    HAS_MEDIAPIPE = False

DEFAULT_KNOWN_DIR = Path(__file__).parent / "known_faces"


def load_known_faces(known_dir: Optional[Path] = None) -> Tuple[List[str], List[np.ndarray]]:
    """Load enrolled face encodings and names from known_faces directory."""
    k_dir = known_dir or DEFAULT_KNOWN_DIR
    names = []
    encodings = []

    if not k_dir.exists():
        return names, encodings

    for p in sorted(k_dir.glob("*.npy")):
        try:
            enc = np.load(p)
            encodings.append(enc)
            names.append(p.stem)
        except Exception as e:
            print(f"Warning: Failed to load encoding '{p.name}': {e}", file=sys.stderr)

    return names, encodings


def match_face(
    known_encs: List[np.ndarray],
    names: List[str],
    face_enc: np.ndarray,
    tolerance: float = 0.50,
) -> Tuple[str, float]:
    """Find the best-matching known face using minimum Euclidean distance.

    Returns:
        (name, confidence) where confidence is between 0.0 and 1.0.
    """
    if not known_encs or face_recognition is None:
        return "Unknown", 0.0

    distances = face_recognition.face_distance(known_encs, face_enc)
    if len(distances) == 0:
        return "Unknown", 0.0

    best_idx = int(np.argmin(distances))
    best_dist = float(distances[best_idx])

    if best_dist <= tolerance:
        confidence = max(0.0, min(1.0, 1.0 - (best_dist / tolerance) * 0.5))
        return names[best_idx], confidence

    return "Unknown", 0.0


def mediapipe_eyes_ear(landmarks, image_w: int, image_h: int) -> Optional[float]:
    """Compute EAR using Mediapipe face mesh landmarks."""
    # Indices for 6 points on left and right eyes
    LEFT = [33, 160, 158, 133, 153, 144]
    RIGHT = [362, 385, 387, 263, 373, 380]

    def lm_to_xy(idx):
        lm = landmarks[idx]
        return (lm.x * image_w, lm.y * image_h)

    try:
        left_eye = [lm_to_xy(i) for i in LEFT]
        right_eye = [lm_to_xy(i) for i in RIGHT]
    except Exception:
        return None

    ler = eye_aspect_ratio(left_eye)
    rer = eye_aspect_ratio(right_eye)
    if ler <= 0.0 or rer <= 0.0:
        return None
    return (ler + rer) / 2.0


def find_matching_mesh(
    mesh_landmarks_list: List[Any],
    box: Tuple[int, int, int, int],
    w: int,
    h: int,
) -> Optional[Any]:
    """Associate a face bounding box (top, right, bottom, left) with the nearest Mediapipe FaceMesh."""
    if not mesh_landmarks_list:
        return None

    top, right, bottom, left = box
    box_cx = (left + right) / 2.0
    box_cy = (top + bottom) / 2.0

    best_mesh = None
    min_dist = float("inf")

    for face_lms in mesh_landmarks_list:
        # Landmark 1 is the nose tip in Mediapipe FaceMesh
        nose = face_lms.landmark[1]
        nx = nose.x * w
        ny = nose.y * h

        dist = math.hypot(nx - box_cx, ny - box_cy)
        if dist < min_dist:
            min_dist = dist
            best_mesh = face_lms

    # Only match if nose is reasonably close to the bounding box
    box_diag = math.hypot(right - left, bottom - top)
    if min_dist <= box_diag:
        return best_mesh
    return None


def run_unlock_loop(cfg: UnlockConfig) -> None:
    if not HAS_CV2 or not HAS_FACE_REC:
        raise RuntimeError(
            "opencv-python and face_recognition are required to run the live unlock loop. "
            "Please install requirements.txt."
        )

    names, known_encs = load_known_faces()
    print(f"Loaded {len(names)} enrolled face(s): {names}")
    if not names:
        print("Notice: No known faces enrolled yet. Run `python enroll.py --name <Name>` first.")

    cap = cv2.VideoCapture(cfg.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam device index {cfg.camera}")

    # Initialize Hardware Controller (Raspberry Pi GPIO / Arduino / Relays)
    hw_cfg = HardwareConfig(
        enable_gpio=cfg.gpio,
        relay_pin=cfg.gpio_relay,
        led_green_pin=cfg.gpio_led_green,
        led_red_pin=cfg.gpio_led_red,
        led_blue_pin=cfg.gpio_led_blue,
        buzzer_pin=cfg.gpio_buzzer,
        button_pin=cfg.gpio_button,
        unlock_duration=cfg.unlock_duration,
        serial_port=cfg.serial_port,
        serial_baud=cfg.serial_baud,
        webhook_url=cfg.webhook_url,
    )
    hw = HardwareManager(hw_cfg)

    # Register manual exit button callback
    def _on_manual_button():
        print("\n[HARDWARE] Manual Exit Pushbutton Pressed -> Unlocking Actuators!")
        hw.unlock("Manual_Button", cfg.unlock_duration)

    hw.register_button_callback(_on_manual_button)

    # Set up Mediapipe FaceMesh if requested
    use_mediapipe = cfg.use_mediapipe and HAS_MEDIAPIPE
    if cfg.use_mediapipe and not HAS_MEDIAPIPE:
        print("Warning: --use-mediapipe requested but mediapipe is not installed; falling back to face_recognition landmarks.")

    mp_face = None
    if use_mediapipe:
        mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # State managers
    blink_detectors: Dict[str, BlinkDetector] = {}
    challenge_mgr = ChallengeManager(
        min_blinks=cfg.challenge_min,
        max_blinks=cfg.challenge_max,
        timeout=cfg.challenge_timeout,
    )
    unlocked_users: set[str] = set()

    prev_frame_time = time.time()
    fps = 0.0

    print("Unlock monitor active. Press 'q' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Calculate FPS
            now = time.time()
            dt = now - prev_frame_time
            prev_frame_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Resize frame for performance
            scale = max(0.1, min(1.0, cfg.scale))
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb_small.shape

            display = frame.copy() if not cfg.headless else None

            # Detect faces with face_recognition
            boxes = face_recognition.face_locations(rgb_small)
            encs = face_recognition.face_encodings(rgb_small, boxes)

            # Run mediapipe ONCE per frame if enabled
            mp_results = None
            if use_mediapipe and mp_face is not None:
                mp_results = mp_face.process(rgb_small)

            mesh_landmarks_list = mp_results.multi_face_landmarks if mp_results else None

            for (top, right, bottom, left), enc in zip(boxes, encs):
                name, confidence = match_face(known_encs, names, enc, tolerance=cfg.tolerance)

                # Hardware indicator: face sensing
                if name != "Unknown":
                    hw.indicate_sensing()

                # Scale coordinates back to original frame
                orig_box = (
                    int(top / scale),
                    int(right / scale),
                    int(bottom / scale),
                    int(left / scale),
                )

                # Initialize detector for this identity
                if name not in blink_detectors:
                    blink_detectors[name] = BlinkDetector(
                        ear_threshold=cfg.ear,
                        consecutive_frames=cfg.consec,
                    )
                detector = blink_detectors[name]

                # Extract EAR
                ear: Optional[float] = None

                if use_mediapipe and mesh_landmarks_list:
                    matched_mesh = find_matching_mesh(mesh_landmarks_list, (top, right, bottom, left), w, h)
                    if matched_mesh:
                        ear = mediapipe_eyes_ear(matched_mesh.landmark, w, h)

                if ear is None:
                    landmarks_list = face_recognition.face_landmarks(rgb_small, [(top, right, bottom, left)])
                    if landmarks_list:
                        lm = landmarks_list[0]
                        left_eye = lm.get("left_eye")
                        right_eye = lm.get("right_eye")
                        if left_eye and right_eye:
                            ler = eye_aspect_ratio(left_eye)
                            rer = eye_aspect_ratio(right_eye)
                            if ler > 0 and rer > 0:
                                ear = (ler + rer) / 2.0

                # Process blink update
                blink_occurred = False
                total_blinks = detector.blink_count
                curr_ear = ear or 0.0

                if ear is not None:
                    blink_occurred, total_blinks, curr_ear = detector.update(ear, timestamp=now)
                    if blink_occurred:
                        print(f"Blink registered for '{name}' (total: {total_blinks})")
                        play_sound("blink", cfg.sound)
                        hw.indicate_blink(total_blinks)

                # Head pose estimation (if enabled)
                pose_str = ""
                if cfg.head_pose:
                    lm_list = face_recognition.face_landmarks(rgb_small, [(top, right, bottom, left)])
                    if lm_list:
                        lm = lm_list[0]
                        nose_bridge = lm.get("nose_bridge")
                        left_eye = lm.get("left_eye")
                        right_eye = lm.get("right_eye")
                        if nose_bridge and left_eye and right_eye:
                            pose = HeadPoseEstimator.get_pose_direction(
                                nose_bridge[0], left_eye[0], right_eye[3]
                            )
                            pose_str = f" Pose: {pose}"

                # Handle unlock logic
                is_authenticated = False
                unlock_method = "blink"

                if name != "Unknown" and name not in unlocked_users:
                    if cfg.challenge:
                        status, blinks_done, target, remain = challenge_mgr.update(
                            name, current_blinks=total_blinks, now=now
                        )
                        if status == "passed":
                            is_authenticated = True
                            unlock_method = f"challenge_blinks_{target}"
                        elif status == "expired":
                            print(f"Challenge expired for '{name}'. Resetting...")
                            play_sound("fail", cfg.sound)
                            hw.indicate_deny()
                    else:
                        if total_blinks >= cfg.blinks:
                            is_authenticated = True
                            unlock_method = f"blinks_{cfg.blinks}"

                # Trigger unlock
                if is_authenticated and name not in unlocked_users:
                    unlocked_users.add(name)
                    print(f"\n==========================================")
                    print(f"  >>> ACCESS GRANTED: {name} <<<")
                    print(f"==========================================\n")
                    play_sound("unlock", cfg.sound)

                    # Trigger hardware actuators (Relay / Solenoid / Servo / Webhook)
                    hw.unlock(name, cfg.unlock_duration)

                    # Save snapshot if directory specified
                    if cfg.save_unlock_shots:
                        shot_dir = Path(cfg.save_unlock_shots)
                        shot_dir.mkdir(parents=True, exist_ok=True)
                        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                        shot_path = shot_dir / f"unlock_{sanitize_name(name)}_{ts_str}.jpg"
                        cv2.imwrite(str(shot_path), frame)
                        print(f"Saved unlock snapshot to '{shot_path}'")

                    # Log unlock event
                    if cfg.log:
                        log_unlock_event(
                            cfg.log,
                            name=name,
                            method=unlock_method,
                            details={
                                "confidence": round(confidence, 3),
                                "ear": round(curr_ear, 3),
                                "total_blinks": total_blinks,
                            },
                        )

                    if cfg.exit_on_unlock:
                        return

                # Render HUD graphics
                if not cfg.headless and display is not None:
                    if name in unlocked_users:
                        color = (0, 255, 0)
                    elif name != "Unknown":
                        color = (255, 200, 0)
                    else:
                        color = (0, 0, 255)

                    draw_hud_box(display, orig_box, name, confidence if name != "Unknown" else None, color)

                    l = orig_box[3]
                    b = orig_box[2]

                    if cfg.challenge and name != "Unknown":
                        ch = challenge_mgr.get_challenge(name)
                        if ch and ch.status != "passed":
                            status, done, target, remain = challenge_mgr.update(
                                name, current_blinks=total_blinks, now=now
                            )
                            if status == "cooldown":
                                ch_text = f"Cooldown: {remain:.1f}s"
                            else:
                                ch_text = f"Blink {done}/{target} ({remain:.1f}s)"
                            cv2.putText(display, ch_text, (l, b + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                        elif name in unlocked_users:
                            cv2.putText(display, "UNLOCKED!", (l, b + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        label = f"Blinks: {total_blinks}/{cfg.blinks}{pose_str}"
                        if name in unlocked_users:
                            label += " [UNLOCKED]"
                        cv2.putText(display, label, (l, b + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    draw_ear_bar(display, curr_ear, cfg.ear, pos=(15, 25))
                    cv2.putText(display, f"FPS: {fps:.1f}", (display.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if not cfg.headless and display is not None:
                cv2.imshow("Face & Eye Blink Unlock (Press 'q' to quit)", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    finally:
        cap.release()
        hw.cleanup()
        if mp_face is not None:
            mp_face.close()
        if not cfg.headless and cv2 is not None:
            cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live face recognition and eye blink unlock demo.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON configuration file")
    parser.add_argument("--save-config", type=str, default=None, help="Save current settings to a JSON file and exit")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--scale", type=float, default=0.5, help="Resize scale for faster face processing (0.1-1.0)")
    parser.add_argument("--ear", type=float, default=0.21, help="EAR threshold for blink detection (default: 0.21)")
    parser.add_argument("--consec", type=int, default=2, help="Consecutive frames below EAR to register a blink (default: 2)")
    parser.add_argument("--blinks", type=int, default=1, help="Number of blinks required to unlock (default: 1)")
    parser.add_argument("--tolerance", type=float, default=0.50, help="Face matching tolerance distance (default: 0.50)")
    parser.add_argument("--exit-on-unlock", action="store_true", help="Exit immediately after a successful unlock")
    parser.add_argument("--use-mediapipe", action="store_true", help="Use Mediapipe FaceMesh for landmarks (if installed)")
    parser.add_argument("--log", type=str, default=None, help="Path to append unlock events (CSV or JSON)")
    parser.add_argument("--save-unlock-shots", type=str, default=None, help="Directory to save snapshots of unlock events")
    parser.add_argument("--headless", action="store_true", help="Run without graphical window (for server/daemon use)")
    parser.add_argument("--sound", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable audio feedback beeps")
    parser.add_argument("--head-pose", action="store_true", help="Enable head pose orientation estimation")
    parser.add_argument("--challenge", action="store_true", help="Enable randomized challenge-response liveness check")
    parser.add_argument("--challenge-min", type=int, default=1, help="Minimum required blinks in challenge")
    parser.add_argument("--challenge-max", type=int, default=3, help="Maximum required blinks in challenge")
    parser.add_argument("--challenge-timeout", type=float, default=8.0, help="Seconds allowed to complete the challenge")

    # Hardware & Electrical Component Arguments
    parser.add_argument("--gpio", action="store_true", help="Enable direct Raspberry Pi GPIO control (relays, LEDs, buzzers)")
    parser.add_argument("--gpio-relay", type=int, default=18, help="Raspberry Pi BCM pin for electronic lock relay (default: 18)")
    parser.add_argument("--gpio-led-green", type=int, default=23, help="BCM pin for Green Granted LED (default: 23)")
    parser.add_argument("--gpio-led-red", type=int, default=24, help="BCM pin for Red Locked/Denied LED (default: 24)")
    parser.add_argument("--gpio-led-blue", type=int, default=22, help="BCM pin for Blue Sensing LED (default: 22)")
    parser.add_argument("--gpio-buzzer", type=int, default=25, help="BCM pin for Piezo Buzzer (default: 25)")
    parser.add_argument("--gpio-button", type=int, default=17, help="BCM pin for Manual Exit Pushbutton (default: 17)")
    parser.add_argument("--unlock-duration", type=float, default=3.0, help="Seconds to energize lock relay (default: 3.0)")
    parser.add_argument("--serial-port", type=str, default=None, help="Serial port for Arduino / ESP32 (e.g. /dev/ttyACM0 or COM3)")
    parser.add_argument("--serial-baud", type=int, default=115200, help="Baud rate for Arduino serial (default: 115200)")
    parser.add_argument("--webhook-url", type=str, default=None, help="HTTP URL to trigger webhook on unlock")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load baseline config from file if provided, otherwise default
    if args.config:
        cfg = UnlockConfig.load(args.config)
    else:
        cfg = UnlockConfig()

    # CLI args override file config
    cli_args = set()
    for item in sys.argv[1:]:
        if item.startswith("--"):
            cli_args.add(item.lstrip("-").split("=")[0])

    if "camera" in cli_args:
        cfg.camera = args.camera
    if "scale" in cli_args:
        cfg.scale = args.scale
    if "ear" in cli_args:
        cfg.ear = args.ear
    if "consec" in cli_args:
        cfg.consec = args.consec
    if "blinks" in cli_args:
        cfg.blinks = args.blinks
    if "tolerance" in cli_args:
        cfg.tolerance = args.tolerance
    if "exit_on_unlock" in cli_args or args.exit_on_unlock:
        cfg.exit_on_unlock = args.exit_on_unlock
    if "use_mediapipe" in cli_args or args.use_mediapipe:
        cfg.use_mediapipe = args.use_mediapipe
    if "log" in cli_args:
        cfg.log = args.log
    if "save_unlock_shots" in cli_args:
        cfg.save_unlock_shots = args.save_unlock_shots
    if "headless" in cli_args or args.headless:
        cfg.headless = args.headless
    if "sound" in cli_args or "no-sound" in cli_args or "no_sound" in cli_args:
        cfg.sound = args.sound
    if "head_pose" in cli_args or args.head_pose:
        cfg.head_pose = args.head_pose
    if "challenge" in cli_args or args.challenge:
        cfg.challenge = args.challenge
    if "challenge_min" in cli_args:
        cfg.challenge_min = args.challenge_min
    if "challenge_max" in cli_args:
        cfg.challenge_max = args.challenge_max
    if "challenge_timeout" in cli_args:
        cfg.challenge_timeout = args.challenge_timeout

    # Hardware CLI overrides
    if "gpio" in cli_args or args.gpio:
        cfg.gpio = args.gpio
    if "gpio_relay" in cli_args:
        cfg.gpio_relay = args.gpio_relay
    if "gpio_led_green" in cli_args:
        cfg.gpio_led_green = args.gpio_led_green
    if "gpio_led_red" in cli_args:
        cfg.gpio_led_red = args.gpio_led_red
    if "gpio_led_blue" in cli_args:
        cfg.gpio_led_blue = args.gpio_led_blue
    if "gpio_buzzer" in cli_args:
        cfg.gpio_buzzer = args.gpio_buzzer
    if "gpio_button" in cli_args:
        cfg.gpio_button = args.gpio_button
    if "unlock_duration" in cli_args:
        cfg.unlock_duration = args.unlock_duration
    if "serial_port" in cli_args:
        cfg.serial_port = args.serial_port
    if "serial_baud" in cli_args:
        cfg.serial_baud = args.serial_baud
    if "webhook_url" in cli_args:
        cfg.webhook_url = args.webhook_url

    # Save config and exit if requested
    if args.save_config:
        cfg.save(args.save_config)
        print(f"Saved configuration to '{args.save_config}'")
        return

    run_unlock_loop(cfg)


if __name__ == "__main__":
    main()
