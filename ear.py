import math
from typing import List, Tuple


def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(eye: List[Tuple[float, float]]) -> float:
    """Compute EAR for an eye given 6 (x,y) points.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    if len(eye) < 6:
        return 0.0
    p1, p2, p3, p4, p5, p6 = eye[:6]
    A = euclid(p2, p6)
    B = euclid(p3, p5)
    C = euclid(p1, p4)
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)
