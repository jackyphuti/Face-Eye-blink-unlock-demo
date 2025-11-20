# Face / Eye (blink) unlock demo

This small demo shows how to enroll a user's face and then use face recognition together with an eye-blink liveness check to "unlock" (prints a message). It uses OpenCV for camera I/O and the face_recognition library (dlib under the hood) for face detection/encodings and facial landmarks.

Files:
- `enroll.py` — capture one or more images for a user and save face encoding.
- `unlock.py` — live camera loop that recognizes a stored user and requires a blink to confirm liveness.
- `requirements.txt` — main Python dependencies.

Notes and prerequisites:
- `face_recognition` depends on `dlib`. On Linux you'll likely need to install system packages before pip (cmake, build-essential, libgtk-3-dev, libboost-all-dev, etc.). See https://github.com/ageitgey/face_recognition for platform-specific details.

Quick start (example):

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Enroll a user (replace NAME):

```bash
python enroll.py --name Jacky
```

Follow the on-screen prompts to capture an image.

3. Run unlock (press 'q' to quit):

```bash
python unlock.py
```
4. Run unlock using Mediapipe and log events:

```bash
python unlock.py --use-mediapipe --log unlocks.log --exist-on-unlock
```

5. if --use-mediapipe is given but Mediapipe is not installed, the script warns and falls back to the original landmarks. Please tune the thresholds if needed, like this :

```bash
python unlock.py --use-mediapipe --ear 0.18 --concec 3 --blinks 2 --log unlocks.log
```
6. enrollment improvement (multi-shot averaging)
Enroll (3-shot average):

```bash
python manage_faces.py list
python manage_faces.py remove Alice
```
Unlock with challenge + logging:

```bash
python unlock.py --use-mediapipe --challenge --log unlocks.csv --exit-on-unlock
```
Tune thresholds:

```bash
python unlock.py --ear 0.19 --consec 3 --blinks 1 --challenge --challenge-min 2 --challenge-max 3
```
If you only want non-GUI behavior (no imshow), you can uninstall GUI OpenCV and install the headless build:

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```


When a known face is detected, the script waits for a blink (eye aspect ratio threshold) and then prints "UNLOCKED: <name>".

Security & limitations:
- This is a demo, not a production authentication system. Face recognition and blink checks provide a basic liveness signal but can be spoofed. For production use, combine multiple liveness sensors and secure enrollment.
- Performance and accuracy depend on camera, lighting, and installed versions of dlib/face_recognition.

License: MIT-style demo.
