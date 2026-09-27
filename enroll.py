#!/usr/bin/env python3
"""Enroll a user by capturing images from webcam or an existing image file,

and saving the face encoding to known_faces/.

Usage examples:
  python enroll.py --name Alice
  python enroll.py --name Alice --shots 3
  python enroll.py --name Alice --interactive
  python enroll.py --name Bob --image path/to/photo.jpg
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

from utils import sanitize_name

DEFAULT_KNOWN_DIR = Path(__file__).parent / "known_faces"


def check_image_quality(frame_bgr: np.ndarray, min_brightness: float = 40.0) -> Tuple[bool, str]:
    """Check basic quality attributes of a captured image."""
    try:
        import cv2
    except ImportError:
        return True, "Quality check skipped (cv2 not available)"

    # Brightness check
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    if mean_val < min_brightness:
        return False, f"Image is too dark (brightness: {mean_val:.1f} < {min_brightness})"

    # Blur check using Laplacian variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 30.0:
        return False, f"Image appears blurry (Laplacian variance: {lap_var:.1f} < 30.0)"

    return True, "Quality check passed"


def capture_webcam_frame(
    camera_idx: int = 0,
    countdown: int = 5,
    interactive: bool = False,
    headless: bool = False,
    shot_index: int = 1,
    total_shots: int = 1,
) -> Optional[np.ndarray]:
    """Capture a single frame from webcam, either with countdown or spacebar press."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("OpenCV (cv2) is required to capture webcam frames.")

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam on index {camera_idx}")

    window_name = f"Enrollment - Shot {shot_index}/{total_shots}"
    captured_frame = None

    try:
        if headless:
            # Headless: discard a few warmup frames then grab
            for _ in range(10):
                cap.read()
            ret, frame = cap.read()
            if ret:
                captured_frame = frame
        elif interactive:
            print(f"[{shot_index}/{total_shots}] Position yourself and press SPACE to capture, or 'q' to cancel.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                disp = frame.copy()
                msg = f"Shot {shot_index}/{total_shots} - Press SPACE to capture, 'q' to cancel"
                cv2.putText(disp, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(window_name, disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    captured_frame = frame
                    break
                elif key == ord("q"):
                    break
        else:
            # Countdown mode
            # For multi-shot, use a shorter countdown after the first shot
            wait_sec = countdown if shot_index == 1 else min(3, countdown)
            print(f"[{shot_index}/{total_shots}] Capturing in {wait_sec} seconds...")
            start_t = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                elapsed = time.time() - start_t
                remaining = int(math.ceil(wait_sec - elapsed))
                disp = frame.copy()
                cv2.putText(
                    disp,
                    f"Shot {shot_index}/{total_shots}: Capturing in {max(1, remaining)}s...",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(window_name, disp)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                if elapsed >= wait_sec:
                    captured_frame = frame
                    break
    finally:
        cap.release()
        if not headless:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

    return captured_frame


def enroll_from_images(
    name: str,
    frames: List[np.ndarray],
    known_dir: Path,
) -> bool:
    """Extract face encodings from images, compute mean encoding, and save to known_dir."""
    try:
        import face_recognition
        import cv2
    except ImportError:
        raise RuntimeError("face_recognition and cv2 are required for face encoding extraction.")

    encs = []
    usable_frames = []

    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)

        if len(boxes) == 0:
            print(f"Shot {i+1}: No face detected — skipping.")
            continue
        if len(boxes) > 1:
            print(f"Shot {i+1}: Multiple faces detected ({len(boxes)}) — skipping.")
            continue

        encoding = face_recognition.face_encodings(rgb, boxes)[0]
        encs.append(encoding)
        usable_frames.append(frame)

    if not encs:
        print("Enrollment failed: No usable face images captured.")
        return False

    # Compute averaged and normalized encoding
    mean_encoding = np.mean(encs, axis=0)
    norm = np.linalg.norm(mean_encoding)
    if norm > 0:
        mean_encoding = mean_encoding / norm

    # Save to known_faces
    clean_name = sanitize_name(name)
    known_dir.mkdir(parents=True, exist_ok=True)

    npy_path = known_dir / f"{clean_name}.npy"
    jpg_path = known_dir / f"{clean_name}.jpg"

    np.save(npy_path, mean_encoding)
    cv2.imwrite(str(jpg_path), usable_frames[-1])

    print(f"Successfully enrolled '{name}' ({clean_name}) with {len(encs)} shots!")
    print(f"  • Encoding: {npy_path.name}")
    print(f"  • Snapshot: {jpg_path.name}")
    return True


def enroll(
    name: str,
    shots: int = 1,
    camera: int = 0,
    image_path: Optional[str] = None,
    interactive: bool = False,
    headless: bool = False,
    timeout: int = 5,
    delay: float = 0.5,
    custom_dir: Optional[str] = None,
) -> bool:
    k_dir = Path(custom_dir) if custom_dir else DEFAULT_KNOWN_DIR

    # Enrollment via existing image file
    if image_path:
        p = Path(image_path)
        if not p.exists():
            print(f"Error: Image file '{image_path}' not found.", file=sys.stderr)
            return False
        try:
            import cv2
            img = cv2.imread(str(p))
            if img is None:
                print(f"Error: Could not read image at '{image_path}'.", file=sys.stderr)
                return False
            return enroll_from_images(name, [img], k_dir)
        except Exception as e:
            print(f"Error loading image: {e}", file=sys.stderr)
            return False

    # Enrollment via webcam
    frames = []
    print(f"Starting enrollment for '{name}' ({shots} shots)...")
    for i in range(shots):
        frame = capture_webcam_frame(
            camera_idx=camera,
            countdown=timeout,
            interactive=interactive,
            headless=headless,
            shot_index=i + 1,
            total_shots=shots,
        )
        if frame is None:
            print(f"Shot {i+1} cancelled or failed.")
            continue

        ok, msg = check_image_quality(frame)
        if not ok:
            print(f"Warning on shot {i+1}: {msg}")

        frames.append(frame)
        if i + 1 < shots:
            time.sleep(delay)

    if not frames:
        print("Enrollment aborted: No frames captured.")
        return False

    return enroll_from_images(name, frames, k_dir)


def main() -> None:
    import math

    parser = argparse.ArgumentParser(description="Enroll a user face for unlock demo.")
    parser.add_argument("--name", required=True, help="Name of the person to enroll")
    parser.add_argument("--shots", type=int, default=1, help="Number of images to capture and average")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--image", type=str, default=None, help="Enroll from an existing image file instead of webcam")
    parser.add_argument("--interactive", action="store_true", help="Press SPACE to capture each shot manually")
    parser.add_argument("--headless", action="store_true", help="Capture without GUI display window")
    parser.add_argument("--timeout", type=int, default=5, help="Countdown timer in seconds before capture")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between multi-shot captures in seconds")
    parser.add_argument("--dir", type=str, default=None, help="Directory to save face encodings")

    args = parser.parse_args()

    success = enroll(
        name=args.name,
        shots=args.shots,
        camera=args.camera,
        image_path=args.image,
        interactive=args.interactive,
        headless=args.headless,
        timeout=args.timeout,
        delay=args.delay,
        custom_dir=args.dir,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
