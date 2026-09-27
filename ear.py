import math
from typing import List, Tuple, Sequence, Any, Optional


def euclid(a: Any, b: Any) -> float:
    """Compute Euclidean distance between two 2D points.

    Supports tuples, lists, and numpy arrays.
    """
    try:
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        dist = math.hypot(dx, dy)
        if math.isnan(dist) or math.isinf(dist):
            return 0.0
        return dist
    except (IndexError, TypeError, ValueError):
        return 0.0


def eye_aspect_ratio(eye: Sequence[Any]) -> float:
    """Compute Eye Aspect Ratio (EAR) for an eye given 6 (x, y) landmark points.

    Formula (Soukupova and Cech, 2016):
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    Parameters:
        eye: A sequence of at least 6 points [(x, y), ...].

    Returns:
        The EAR float value, or 0.0 if invalid or eye width is zero.
    """
    if eye is None or len(eye) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye[:6]
    A = euclid(p2, p6)
    B = euclid(p3, p5)
    C = euclid(p1, p4)

    if C <= 1e-7:
        return 0.0

    ear = (A + B) / (2.0 * C)
    if math.isnan(ear) or math.isinf(ear):
        return 0.0
    return ear


def smooth_ear(prev_ear: Optional[float], current_ear: float, alpha: float = 0.3) -> float:
    """Apply Exponential Moving Average (EMA) to smooth EAR across frames.

    Parameters:
        prev_ear: Previous smoothed EAR, or None if starting fresh.
        current_ear: Current raw EAR measurement.
        alpha: Smoothing factor between 0.0 and 1.0 (higher = more responsive, lower = smoother).

    Returns:
        Smoothed EAR float.
    """
    if prev_ear is None:
        return current_ear
    alpha = max(0.0, min(1.0, alpha))
    return alpha * current_ear + (1.0 - alpha) * prev_ear
