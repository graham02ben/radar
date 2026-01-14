# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 10:08:02 2026

@author: bboyg
"""

import numpy as np
from rcs_model import RCSGrid, RCSModel

def test_rcs_grid_shape_check():
    phi = np.arange(0, 360, 10)
    theta = np.arange(0, 181, 1)
    bad = np.zeros((len(theta), len(phi)))  # wrong order
    try:
        RCSGrid(phi, theta, bad)
        assert False, "Expected shape error"
    except ValueError:
        assert True

def test_rcs_constant_field_returns_constant():
    phi = np.arange(0, 360, 10)
    theta = np.arange(0, 181, 1)
    sigma = np.ones((len(phi), len(theta))) * 5.0  # 5 m^2 everywhere
    grid = RCSGrid(phi, theta, sigma)
    model = RCSModel(grid)

    val = model.sigma(az_b_rad=1.0, el_b_rad=0.2)
    assert abs(val - 5.0) < 1e-6
