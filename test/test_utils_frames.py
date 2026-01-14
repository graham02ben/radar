# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 09:19:52 2026

@author: bboyg
"""

import numpy as np
from utils_frames import lla_to_ecef, ecef_to_enu_matrix, enu_to_azelr

def test_lla_to_ecef_magnitude_reasonable():
    # Equator, sea level => radius ~ 6378 km
    x, y, z = lla_to_ecef(0.0, 0.0, 0.0)
    r = np.sqrt(x*x + y*y + z*z)
    assert 6.3e6 < r < 6.5e6

def test_ecef_to_enu_matrix_orthonormal():
    R = ecef_to_enu_matrix(52.0, 0.185)
    I = R.T @ R
    assert np.linalg.norm(I - np.eye(3)) < 1e-12

def test_enu_to_azelr_basic_axes():
    # north
    az, el, r = enu_to_azelr(np.array([0.0, 100.0, 0.0]))
    assert abs(az - 0.0) < 1e-12
    assert abs(el - 0.0) < 1e-12
    assert abs(r - 100.0) < 1e-12

    # east
    az, el, r = enu_to_azelr(np.array([100.0, 0.0, 0.0]))
    assert abs(az - np.pi/2) < 1e-12
    assert abs(el - 0.0) < 1e-12

    # up
    az, el, r = enu_to_azelr(np.array([0.0, 0.0, 100.0]))
    assert abs(el - np.pi/2) < 1e-12
