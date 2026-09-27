"""Utility helpers for Face-Eye-blink-unlock-demo.

Includes:
- Safe filesystem name sanitization
- Cross-platform audio alerts (Windows winsound / terminal bell)
- Standardized logging (CSV / JSON) with timezone-aware timestamps
- Visual HUD rendering helpers
"""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


def sanitize_name(name: str) -> str:
    """Sanitize a person's name for safe filesystem usage across platforms.

    Strips directory traversal, invalid characters, and normalizes whitespace.
    """
    if not name or not isinstance(name, str):
        return "unnamed"

    # Replace multiple whitespace with a single underscore
    cleaned = re.sub(r"\s+", "_", name.strip())
    # Remove filesystem forbidden characters: \ / : * ? " < > |
    cleaned = re.sub(r'[\\/*?:"<>|]', "", cleaned)
    # Remove directory traversal dots
    cleaned = re.sub(r"\.+", "", cleaned)
    cleaned = cleaned.strip("_")

    if not cleaned:
        return "unnamed"
    return cleaned


def play_sound(sound_type: str, enabled: bool = True) -> None:
    """Play audio feedback without crashing across platforms."""
    if not enabled:
        return

    # Windows native winsound
    if sys.platform == "win32":
        try:
            import winsound

            if sound_type == "unlock":
                winsound.Beep(1760, 150)
            elif sound_type == "blink":
                winsound.Beep(1200, 60)
            elif sound_type == "fail":
                winsound.Beep(440, 250)
            elif sound_type == "alert":
                winsound.Beep(880, 100)
            return
        except Exception:
            pass

    # Fallback to terminal bell
    try:
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass


def log_unlock_event(
    log_path: str | Path,
    name: str,
    method: str = "blink",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Append unlock event to a CSV or JSON log file with UTC ISO timestamps."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    extra_str = json.dumps(details) if details else ""

    if path.suffix.lower() == ".json":
        # Load existing JSON array or start new
        records = []
        if path.exists() and path.stat().st_size > 0:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    records = json.load(f)
                    if not isinstance(records, list):
                        records = [records]
            except Exception:
                records = []
        records.append({
            "timestamp": timestamp,
            "name": name,
            "method": method,
            "details": details or {},
        })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    else:
        # Default to CSV
        file_exists = path.exists() and path.stat().st_size > 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "name", "method", "details"])
            writer.writerow([timestamp, name, method, extra_str])


def draw_hud_box(
    frame: Any,
    box: Tuple[int, int, int, int],
    name: str,
    confidence: Optional[float] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw a styled bounding box with label and confidence."""
    try:
        import cv2
    except ImportError:
        return

    top, right, bottom, left = box
    t, r, b, l = int(top), int(right), int(bottom), int(left)

    # Corner length
    corner_len = min(20, (r - l) // 4, (b - t) // 4)
    thick = 2

    # Draw rectangular frame
    cv2.rectangle(frame, (l, t), (r, b), color, 1)

    # Draw thicker corners
    # Top-left
    cv2.line(frame, (l, t), (l + corner_len, t), color, thick)
    cv2.line(frame, (l, t), (l, t + corner_len), color, thick)
    # Top-right
    cv2.line(frame, (r, t), (r - corner_len, t), color, thick)
    cv2.line(frame, (r, t), (r, t + corner_len), color, thick)
    # Bottom-left
    cv2.line(frame, (l, b), (l + corner_len, b), color, thick)
    cv2.line(frame, (l, b), (l, b - corner_len), color, thick)
    # Bottom-right
    cv2.line(frame, (r, b), (r - corner_len, b), color, thick)
    cv2.line(frame, (r, b), (r, b - corner_len), color, thick)

    # Label text
    label = name
    if confidence is not None:
        label = f"{name} ({confidence*100:.0f}%)"

    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (l, t - h - 6), (l + w + 8, t), color, -1)
    cv2.putText(frame, label, (l + 4, t - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def draw_ear_bar(
    frame: Any,
    ear: float,
    threshold: float,
    pos: Tuple[int, int] = (20, 50),
    bar_width: int = 150,
    bar_height: int = 14,
) -> None:
    """Draw a visual EAR meter with open/closed indicator."""
    try:
        import cv2
    except ImportError:
        return

    x, y = pos
    max_ear = 0.40
    fill_ratio = max(0.0, min(1.0, ear / max_ear))
    fill_w = int(bar_width * fill_ratio)

    is_open = ear >= threshold
    bar_color = (0, 220, 0) if is_open else (0, 100, 255)

    # Background
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (40, 40, 40), -1)
    # Fill
    cv2.rectangle(frame, (x, y), (x + fill_w, y + bar_height), bar_color, -1)
    # Border
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (200, 200, 200), 1)

    # Threshold mark
    thresh_x = x + int(bar_width * (threshold / max_ear))
    cv2.line(frame, (thresh_x, y - 2), (thresh_x, y + bar_height + 2), (0, 255, 255), 2)

    # Label
    status_str = "OPEN" if is_open else "CLOSED"
    text = f"EAR: {ear:.2f} [{status_str}]"
    cv2.putText(frame, text, (x + bar_width + 10, y + bar_height - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
