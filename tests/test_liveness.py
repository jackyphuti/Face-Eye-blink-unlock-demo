import time
import pytest

from liveness import BlinkDetector, ChallengeManager, HeadPoseEstimator


def test_blink_detector_normal_blink():
    detector = BlinkDetector(
        ear_threshold=0.21,
        consecutive_frames=2,
        min_closed_duration=0.05,
        max_closed_duration=1.0,
        cooldown_seconds=0.1,
    )

    t = 1000.0
    # Frame 1: Eyes open
    blink, count, ear = detector.update(0.30, timestamp=t)
    assert not blink
    assert count == 0

    # Frame 2: Eyes closed (frame 1)
    t += 0.05
    blink, count, ear = detector.update(0.15, timestamp=t)
    assert not blink
    assert count == 0

    # Frame 3: Eyes closed (frame 2)
    t += 0.05
    blink, count, ear = detector.update(0.14, timestamp=t)
    assert not blink
    assert count == 0

    # Frame 4: Eyes open again (blink completes)
    t += 0.05
    blink, count, ear = detector.update(0.32, timestamp=t)
    assert blink
    assert count == 1


def test_blink_detector_ignores_too_short():
    detector = BlinkDetector(
        ear_threshold=0.21,
        consecutive_frames=1,
        min_closed_duration=0.10,  # requires 100ms
        max_closed_duration=1.0,
    )
    t = 1000.0
    detector.update(0.30, timestamp=t)

    # Closed for only 20ms (sensor jitter)
    t += 0.02
    detector.update(0.15, timestamp=t)

    # Eyes open immediately
    t += 0.02
    blink, count, _ = detector.update(0.30, timestamp=t)
    assert not blink
    assert count == 0


def test_blink_detector_ignores_too_long():
    detector = BlinkDetector(
        ear_threshold=0.21,
        consecutive_frames=2,
        min_closed_duration=0.05,
        max_closed_duration=0.50,  # Max 500ms
    )
    t = 1000.0
    detector.update(0.30, timestamp=t)

    # Closed for 2.0 seconds (photo spoof or sleeping)
    t += 0.1
    detector.update(0.15, timestamp=t)
    t += 2.0
    detector.update(0.15, timestamp=t)

    # Eyes open
    t += 0.05
    blink, count, _ = detector.update(0.30, timestamp=t)
    assert not blink
    assert count == 0


def test_blink_detector_reset():
    detector = BlinkDetector()
    detector.blink_count = 5
    detector.consecutive_low_frames = 3
    detector.reset()
    assert detector.blink_count == 0
    assert detector.consecutive_low_frames == 0


def test_challenge_manager_isolation_and_success():
    manager = ChallengeManager(min_blinks=2, max_blinks=2, timeout=5.0)

    t = 100.0
    # User already had 10 blinks previously before challenge started
    status, done, target, remaining = manager.update("Alice", current_blinks=10, now=t)
    assert status == "pending"
    assert done == 0
    assert target == 2
    assert remaining == 5.0

    # User does 1 blink (total 11)
    t += 1.0
    status, done, target, remaining = manager.update("Alice", current_blinks=11, now=t)
    assert status == "pending"
    assert done == 1
    assert target == 2

    # User does 2nd blink (total 12)
    t += 1.0
    status, done, target, remaining = manager.update("Alice", current_blinks=12, now=t)
    assert status == "passed"
    assert done == 2


def test_challenge_manager_timeout_and_cooldown():
    manager = ChallengeManager(min_blinks=2, max_blinks=2, timeout=5.0, cooldown_after_fail=2.0)

    t = 100.0
    status, _, _, _ = manager.update("Bob", current_blinks=0, now=t)
    assert status == "pending"

    # Timeout occurs at t=106.0 (> 100.0 + 5.0)
    t = 106.0
    status, _, _, _ = manager.update("Bob", current_blinks=0, now=t)
    assert status == "expired"

    # Subsequent check within cooldown period
    t = 107.0
    status, _, _, cd_remain = manager.update("Bob", current_blinks=0, now=t)
    assert status == "cooldown"
    assert cd_remain > 0


def test_head_pose_estimator():
    left_eye = (100.0, 150.0)
    right_eye = (200.0, 150.0)

    # Centered nose (x=150)
    center_nose = (150.0, 175.0)
    assert HeadPoseEstimator.get_pose_direction(center_nose, left_eye, right_eye) == "FORWARD"

    # Looking left (camera right, nose x=120)
    left_nose = (120.0, 175.0)
    assert HeadPoseEstimator.get_pose_direction(left_nose, left_eye, right_eye) == "LEFT"

    # Looking right (camera left, nose x=180)
    right_nose = (180.0, 175.0)
    assert HeadPoseEstimator.get_pose_direction(right_nose, left_eye, right_eye) == "RIGHT"
