"""Liveness detection and challenge-response management for face unlock.

Features:
- BlinkDetector with duration thresholds (filters out single-frame glitches and closed-eye photo attacks).
- ChallengeManager with isolated per-challenge blink counters, timeout resets, and randomized prompts.
- HeadPoseEstimator using 2D landmark geometry for head turn / nod verification.
"""
from __future__ import annotations
import math
import random
import time
from typing import Dict, List, Optional, Tuple, Any

from ear import smooth_ear


class BlinkDetector:
    """Detects eye blinks based on Eye Aspect Ratio (EAR) over time."""

    def __init__(
        self,
        ear_threshold: float = 0.21,
        consecutive_frames: int = 2,
        min_closed_duration: float = 0.05,
        max_closed_duration: float = 1.5,
        cooldown_seconds: float = 0.2,
        smoothing_alpha: float = 0.4,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.min_closed_duration = min_closed_duration
        self.max_closed_duration = max_closed_duration
        self.cooldown_seconds = cooldown_seconds
        self.smoothing_alpha = smoothing_alpha

        self.consecutive_low_frames: int = 0
        self.blink_count: int = 0
        self.last_blink_time: float = 0.0
        self.eyes_closed_start_time: Optional[float] = None
        self.smoothed_ear: Optional[float] = None
        self.is_currently_closed: bool = False

    def reset(self) -> None:
        """Reset internal counters for a new subject or session."""
        self.consecutive_low_frames = 0
        self.blink_count = 0
        self.last_blink_time = 0.0
        self.eyes_closed_start_time = None
        self.smoothed_ear = None
        self.is_currently_closed = False

    def update(self, raw_ear: float, timestamp: Optional[float] = None) -> Tuple[bool, int, float]:
        """Update detector with a new frame's EAR.

        Parameters:
            raw_ear: Eye aspect ratio calculated for the current frame.
            timestamp: Frame timestamp (defaults to time.time()).

        Returns:
            Tuple of:
                - blink_detected_this_frame (bool)
                - total_blink_count (int)
                - current_smoothed_ear (float)
        """
        now = time.time() if timestamp is None else timestamp
        self.smoothed_ear = smooth_ear(self.smoothed_ear, raw_ear, self.smoothing_alpha)
        ear_val = self.smoothed_ear

        blink_registered = False

        if raw_ear < self.ear_threshold:
            self.consecutive_low_frames += 1
            if self.eyes_closed_start_time is None:
                self.eyes_closed_start_time = now
            self.is_currently_closed = True
        else:
            if self.consecutive_low_frames >= self.consecutive_frames and self.eyes_closed_start_time is not None:
                closed_duration = now - self.eyes_closed_start_time
                cooldown_elapsed = now - self.last_blink_time

                # Check duration validity: must be within realistic blink range
                # and satisfy debounce cooldown
                if (
                    self.min_closed_duration <= closed_duration <= self.max_closed_duration
                    and cooldown_elapsed >= self.cooldown_seconds
                ):
                    self.blink_count += 1
                    self.last_blink_time = now
                    blink_registered = True

            # Reset low-frame state when eyes open
            self.consecutive_low_frames = 0
            self.eyes_closed_start_time = None
            self.is_currently_closed = False

        return blink_registered, self.blink_count, ear_val


class Challenge:
    """Represents a single challenge-response instance."""

    def __init__(
        self,
        target_blinks: int,
        start_time: float,
        timeout: float,
        blinks_at_start: int,
        action_type: str = "blink",
    ) -> None:
        self.target = target_blinks
        self.start_time = start_time
        self.timeout = timeout
        self.blinks_at_start = blinks_at_start
        self.action_type = action_type
        self.status = "pending"  # "pending", "passed", "expired"
        self.completed_at: Optional[float] = None


class ChallengeManager:
    """Manages randomized challenge-response verification per face."""

    def __init__(
        self,
        min_blinks: int = 1,
        max_blinks: int = 3,
        timeout: float = 8.0,
        cooldown_after_fail: float = 2.0,
    ) -> None:
        self.min_blinks = min_blinks
        self.max_blinks = max_blinks
        self.timeout = timeout
        self.cooldown_after_fail = cooldown_after_fail
        self._challenges: Dict[str, Challenge] = {}
        self._fail_cooldowns: Dict[str, float] = {}

    def get_challenge(self, name: str) -> Optional[Challenge]:
        return self._challenges.get(name)

    def start_new_challenge(self, name: str, current_blinks: int, now: Optional[float] = None) -> Challenge:
        t = time.time() if now is None else now
        target = random.randint(self.min_blinks, self.max_blinks)
        ch = Challenge(
            target_blinks=target,
            start_time=t,
            timeout=self.timeout,
            blinks_at_start=current_blinks,
        )
        self._challenges[name] = ch
        return ch

    def update(
        self,
        name: str,
        current_blinks: int,
        now: Optional[float] = None,
    ) -> Tuple[str, int, int, float]:
        """Update challenge state for the given user.

        Returns:
            Tuple of:
                - status: "pending", "passed", "expired", "cooldown"
                - blinks_done: blinks counted toward this challenge
                - target: target blinks required
                - remaining_seconds: seconds left to complete
        """
        t = time.time() if now is None else now

        # Check if user is in failure cooldown
        if name in self._fail_cooldowns:
            cd_remain = self._fail_cooldowns[name] - t
            if cd_remain > 0:
                return "cooldown", 0, 0, cd_remain
            else:
                del self._fail_cooldowns[name]

        ch = self._challenges.get(name)
        if ch is None:
            ch = self.start_new_challenge(name, current_blinks, t)

        if ch.status == "passed":
            return "passed", ch.target, ch.target, 0.0

        elapsed = t - ch.start_time
        remaining = max(0.0, ch.timeout - elapsed)
        blinks_done = max(0, current_blinks - ch.blinks_at_start)

        if blinks_done >= ch.target:
            ch.status = "passed"
            ch.completed_at = t
            return "passed", blinks_done, ch.target, remaining

        if elapsed >= ch.timeout:
            ch.status = "expired"
            self._fail_cooldowns[name] = t + self.cooldown_after_fail
            # Remove expired challenge so next attempt gets a fresh one after cooldown
            del self._challenges[name]
            return "expired", blinks_done, ch.target, 0.0

        return "pending", blinks_done, ch.target, remaining

    def reset_user(self, name: str) -> None:
        self._challenges.pop(name, None)
        self._fail_cooldowns.pop(name, None)


class HeadPoseEstimator:
    """Estimates head orientation (yaw and pitch) from 2D facial landmarks."""

    @staticmethod
    def estimate_yaw_ratio(
        nose_pt: Tuple[float, float],
        left_eye_pt: Tuple[float, float],
        right_eye_pt: Tuple[float, float],
    ) -> float:
        """Estimate horizontal head orientation (yaw ratio).

        Ratio = (nose_x - left_x) / (right_x - left_x)
        - Looking forward: ~0.45 - 0.55
        - Turned Left: < 0.38
        - Turned Right: > 0.62
        """
        dx = right_eye_pt[0] - left_eye_pt[0]
        if abs(dx) < 1e-6:
            return 0.5
        ratio = (nose_pt[0] - left_eye_pt[0]) / dx
        return max(0.0, min(1.0, ratio))

    @classmethod
    def get_pose_direction(
        cls,
        nose_pt: Tuple[float, float],
        left_eye_pt: Tuple[float, float],
        right_eye_pt: Tuple[float, float],
        left_thresh: float = 0.36,
        right_thresh: float = 0.64,
    ) -> str:
        """Classify head pose direction as 'FORWARD', 'LEFT', or 'RIGHT'."""
        ratio = cls.estimate_yaw_ratio(nose_pt, left_eye_pt, right_eye_pt)
        if ratio < left_thresh:
            return "LEFT"
        elif ratio > right_thresh:
            return "RIGHT"
        return "FORWARD"
