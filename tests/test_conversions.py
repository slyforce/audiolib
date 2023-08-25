import copy

import numpy as np

from dsp.conversions import DigitalFilter


def test_digital_filter_identity():
    s1 = np.array([0.2, 0.5, 0.7]).reshape((-1, 1))
    converter = DigitalFilter(np.array([1.0]), np.array([0.0, 1.0]), 1)
    s2 = converter(s1)
    assert np.isclose(s1, s2).all()


def test_digital_filter_buffer_carryover():
    s = np.array([0.2, 0.5, 0.7]).reshape((-1, 1))
    c1 = DigitalFilter(np.array([1.0, 3.0]), np.array([5.0, 1.0]), 1)
    c2 = copy.deepcopy(c1)
    s2_p1 = c1(s[:2])
    s2_p2 = c1(s[2:])

    s2_parts = np.concatenate([s2_p1, s2_p2], axis=0)
    s2_whole = c2(s)
    assert s2_whole.shape == s2_parts.shape
    assert np.isclose(s2_parts, s2_whole).all()


