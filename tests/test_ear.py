import math
import numpy as np
import pytest

from ear import eye_aspect_ratio, euclid, smooth_ear


def test_eye_aspect_ratio_basic():
    # Construct a symmetric eye shape where vertical distances are 2 and horizontal is 4
    p1 = (0.0, 0.0)
    p2 = (1.0, -1.0)
    p3 = (2.0, -1.0)
    p4 = (4.0, 0.0)
    p5 = (2.0, 1.0)
    p6 = (1.0, 1.0)
    eye = [p1, p2, p3, p4, p5, p6]
    ear = eye_aspect_ratio(eye)
    # expected EAR = (2 + 2) / (2 * 4) = 0.5
    assert abs(ear - 0.5) < 1e-6


def test_eye_aspect_ratio_with_numpy():
    eye = np.array([
        [0.0, 0.0],
        [1.0, -1.0],
        [2.0, -1.0],
        [4.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0]
    ])
    ear = eye_aspect_ratio(eye)
    assert abs(ear - 0.5) < 1e-6


def test_eye_aspect_ratio_edge_cases():
    # Less than 6 points
    assert eye_aspect_ratio([]) == 0.0
    assert eye_aspect_ratio(None) == 0.0
    assert eye_aspect_ratio([(0, 0), (1, 1)]) == 0.0

    # Zero width eye (p1 == p4)
    degenerate_eye = [(0.0, 0.0), (0.0, 1.0), (0.0, 1.0), (0.0, 0.0), (0.0, -1.0), (0.0, -1.0)]
    assert eye_aspect_ratio(degenerate_eye) == 0.0

    # Closed eye (vertical distance 0)
    closed_eye = [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (4.0, 0.0),
        (2.0, 0.0),
        (1.0, 0.0)
    ]
    assert eye_aspect_ratio(closed_eye) == 0.0


def test_euclid_distance():
    assert abs(euclid((0, 0), (3, 4)) - 5.0) < 1e-6
    assert abs(euclid([1, 1], [4, 5]) - 5.0) < 1e-6
    assert abs(euclid(np.array([0, 0]), np.array([3, 4])) - 5.0) < 1e-6
    # Invalid inputs
    assert euclid((0,), (1, 2)) == 0.0
    assert euclid(None, (1, 2)) == 0.0


def test_smooth_ear():
    # Initial frame has no previous EAR
    assert smooth_ear(None, 0.3) == 0.3

    # Consecutive smoothing with alpha=0.5
    smoothed = smooth_ear(0.3, 0.2, alpha=0.5)
    assert abs(smoothed - 0.25) < 1e-6

    # Clamping alpha
    assert smooth_ear(0.3, 0.1, alpha=-0.5) == 0.3
    assert smooth_ear(0.3, 0.1, alpha=1.5) == 0.1
