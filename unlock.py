#!/usr/bin/env python3
"""Live unlock demo with command-line options.

Features added:
- CLI args for EAR threshold, required blinks, camera index and resize scale
- Per-person blink counting (require N blinks to unlock)
- On-screen labels showing EAR and blink counts
"""
from pathlib import Path
import argparse
import math
import time
from typing import List, Tuple, Optional

import cv2
import numpy as np
import face_recognition

# Try to import mediapipe for better landmarks (optional)
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    mp = None
    HAS_MEDIAPIPE = False


KNOWN_DIR = Path(__file__).parent / "known_faces"


def load_known():
    names = []
    encodings = []
    if not KNOWN_DIR.exists():
        return names, encodings

    for p in KNOWN_DIR.iterdir():
        if p.suffix.lower() == ".npy":
            enc = np.load(p)
            encodings.append(enc)
            names.append(p.stem)
    return names, encodings


from ear import eye_aspect_ratio


def mediapipe_eyes_ear(landmarks, image_w: int, image_h: int) -> Optional[float]:
    """Compute EAR using Mediapipe face mesh landmarks.

    Uses common landmark indices for left and right eye.
    Returns average EAR or None if required landmarks are missing.
    """
    # Mediapipe Face Mesh indices for eyes (commonly used)
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
    return (ler + rer) / 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--scale", type=float, default=0.5, help="Resize scale for faster processing (0.1-1.0)")
    parser.add_argument("--ear", type=float, default=0.21, help="EAR threshold for blink detection")
    parser.add_argument("--consec", type=int, default=2, help="Consecutive frames below EAR to consider a blink")
    parser.add_argument("--blinks", type=int, default=1, help="Number of blinks required to unlock")
    parser.add_argument("--exit-on-unlock", action="store_true", help="Exit after a successful unlock")
    parser.add_argument("--use-mediapipe", action="store_true", help="Use Mediapipe Face Mesh for more robust landmarks (if installed)")
    parser.add_argument("--log", type=str, default=None, help="Optional path to append unlock events (timestamp,name)")
    parser.add_argument("--challenge", action="store_true", help="Enable challenge-response: random blink count required per unlock_attempt")
    parser.add_argument("--challenge-min", type=int, default=1, help="Minimum blinks for challenge")
    parser.add_argument("--challenge-max", type=int, default=3, help="Maximum blinks for challenge")
    parser.add_argument("--challenge-timeout", type=float, default=8.0, help="Seconds allowed to satisfy the challenge per face")
    args = parser.parse_args()

    names, known_encs = load_known()
    print(f"Loaded {len(names)} known faces: {names}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    use_mediapipe = bool(args.use_mediapipe) and HAS_MEDIAPIPE
    if args.use_mediapipe and not HAS_MEDIAPIPE:
        print("Warning: --use-mediapipe requested but mediapipe is not installed; falling back to face_recognition landmarks")

    mp_face = None
    if use_mediapipe:
        mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)

    # state per detected identity
    consec_counters = {}  # name -> consecutive low-EAR frames
    blink_counts = {}  # name -> total blinks detected
    unlocked = set()

    scale = float(args.scale)
    ear_thresh = float(args.ear)
    consec_needed = int(args.consec)
    blinks_required = int(args.blinks)

    import random

    # per-face challenge state: name -> (target_blinks, start_time, satisfied_flag)
    challenges = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, boxes)

        display = small.copy()

        if encs and known_encs:
            for (top, right, bottom, left), enc in zip(boxes, encs):
                matches = face_recognition.compare_faces(known_encs, enc, tolerance=0.5)
                name = "Unknown"
                if True in matches:
                    idx = matches.index(True)
                    name = names[idx]

                # default state init
                consec_counters.setdefault(name, 0)
                blink_counts.setdefault(name, 0)

                ear = None

                # If mediapipe requested and available, use it for more accurate landmarks
                if use_mediapipe and mp_face is not None:
                    # Mediapipe expects RGB full-size image; we passed scaled RGB
                    results = mp_face.process(rgb)
                    if results and results.multi_face_landmarks:
                        # Use the first face that overlaps with the detected box (best-effort)
                        lm = results.multi_face_landmarks[0].landmark
                        # compute EAR from mediapipe landmarks
                        mp_ear = mediapipe_eyes_ear(lm, rgb.shape[1], rgb.shape[0])
                        if mp_ear is not None:
                            ear = mp_ear
                            cv2.putText(display, f"EAR(MP):{ear:.2f}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
                else:
                    # fallback to face_recognition landmarks
                    landmarks_list = face_recognition.face_landmarks(rgb, [(top, right, bottom, left)])
                    if landmarks_list:
                        lm = landmarks_list[0]
                        left_eye = lm.get("left_eye")
                        right_eye = lm.get("right_eye")
                        if left_eye and right_eye:
                            ler = eye_aspect_ratio(left_eye)
                            rer = eye_aspect_ratio(right_eye)
                            ear = (ler + rer) / 2.0
                            cv2.putText(display, f"EAR:{ear:.2f}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if ear is not None:
                    if ear < ear_thresh:
                        consec_counters[name] += 1
                    else:
                        if consec_counters[name] >= consec_needed:
                            blink_counts[name] += 1
                            print(f"Blink detected for {name} (total={blink_counts[name]})")
                        consec_counters[name] = 0

                    # unlock condition
                    if blink_counts[name] >= blinks_required and name != "Unknown" and name not in unlocked:
                        print(f"UNLOCKED: {name}")
                        unlocked.add(name)
                        # append to log if requested
                        if args.log:
                            try:
                                from datetime import datetime

                                with open(args.log, "a", encoding="utf-8") as fh:
                                    fh.write(f"{datetime.utcnow().isoformat()}Z,{name}\n")
                            except Exception as e:
                                print(f"Warning: failed to write log: {e}")
                        if args.exit_on_unlock:
                            if mp_face is not None:
                                mp_face.close()
                            cap.release()
                            cv2.destroyAllWindows()
                            return

                # draw box and label
                t, r, b, l = int(top), int(right), int(bottom), int(left)
                cv2.rectangle(display, (l, t), (r, b), (0, 255, 0), 2)

                # if challenge mode enabled, possibly create a challenge for this face
                if args.challenge and name != "Unknown":
                    ch = challenges.get(name)
                    now = time.time()
                    if ch is None or ch[2]:
                        # start a new challenge if none or previous satisfied
                        target = random.randint(args.challenge_min, args.challenge_max)
                        challenges[name] = [target, now, False]
                        print(f"Challenge for {name}: blink {target} times")
                    else:
                        target, start_time, done = ch

                    # check timeout
                    target, start_time, done = challenges[name]
                    elapsed = now - start_time
                    remain = max(0.0, args.challenge_timeout - elapsed)
                    label = f"{name} blinks:{blink_counts.get(name,0)} / target:{target} ({remain:.0f}s)"
                else:
                    label = f"{name} blinks:{blink_counts.get(name,0)}"

                cv2.putText(display, label, (l, b + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # If challenge exists, and satisfied now, mark and log
                if args.challenge and name in challenges:
                    target, start_time, done = challenges[name]
                    if not done and blink_counts.get(name, 0) >= target and name != "Unknown":
                        challenges[name][2] = True
                        print(f"Challenge satisfied for {name} (target {target})")
                        # treat this as an unlock
                        if name not in unlocked:
                            print(f"UNLOCKED by challenge: {name}")
                            unlocked.add(name)
                            if args.log:
                                try:
                                    from datetime import datetime

                                    with open(args.log, "a", encoding="utf-8") as fh:
                                        fh.write(f"{datetime.utcnow().isoformat()}Z,{name},challenge={target}\n")
                                except Exception as e:
                                    print(f"Warning: failed to write log: {e}")
                            if args.exit_on_unlock:
                                if mp_face is not None:
                                    mp_face.close()
                                cap.release()
                                cv2.destroyAllWindows()
                                return

        cv2.imshow("Unlock (press q to quit)", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
