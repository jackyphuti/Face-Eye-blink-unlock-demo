#!/usr/bin/env python3
"""Enroll a user by capturing an image from webcam and saving face encoding.

Usage: python enroll.py --name NAME
"""
import argparse
import os
import time
from pathlib import Path

import cv2
import face_recognition
import numpy as np


KNOWN_DIR = Path(__file__).parent / "known_faces"
KNOWN_DIR.mkdir(exist_ok=True)


def capture_image(timeout=10):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("Position yourself in front of the camera. Capturing in {} seconds...".format(timeout))
    for i in range(timeout, 0, -1):
        ret, frame = cap.read()
        if not ret:
            continue
        disp = frame.copy()
        cv2.putText(disp, f"Capturing in {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Enroll", disp)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    # grab a final frame
    ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    if not ret:
        raise RuntimeError("Failed to capture image")
    return frame


def enroll(name: str):
    # default to a single shot -- but support multiple-shot averaging for robustness
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--shots", type=int, default=1, help="Number of images to capture and average encodings")
    args, _ = parser.parse_known_args()

    encs = []
    imgs = []
    for i in range(args.shots):
        print(f"Capturing shot {i+1}/{args.shots}")
        frame = capture_image()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        if len(boxes) == 0:
            print("No face found in this shot — skipping. Try better lighting or move closer.")
            continue
        if len(boxes) > 1:
            print("Multiple faces found in this shot — skipping.")
            continue

        encoding = face_recognition.face_encodings(rgb, boxes)[0]
        encs.append(encoding)
        imgs.append(frame)
        time.sleep(0.5)

    if not encs:
        print("No usable shots captured. Enrollment failed.")
        return

    # average encodings (mean) to get a more stable representation
    encoding = np.mean(encs, axis=0)

    # Save encoding and a representative snapshot (the last usable)
    name_sanitized = "_".join(name.strip().split())
    np.save(KNOWN_DIR / f"{name_sanitized}.npy", encoding)
    img_path = KNOWN_DIR / f"{name_sanitized}.jpg"
    cv2.imwrite(str(img_path), imgs[-1])
    print(f"Enrolled '{name}' -> {img_path} ({(KNOWN_DIR / (name_sanitized + '.npy')).name}) with {len(encs)} usable shots")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Name for the person to enroll")
    args = parser.parse_args()
    enroll(args.name)


if __name__ == "__main__":
    main()
