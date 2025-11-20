import math

from unlock import eye_aspect_ratio


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
