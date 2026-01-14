# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 10:08:44 2026

@author: bboyg
"""

import numpy as np
from trajectory import Trajectory

def test_interp_endpoints():
    t = np.array([0.0, 1.0, 2.0])
    pos = np.array([[0,0,0],[1,1,1],[2,2,2]], dtype=float)
    roll = np.zeros((3,3))
    tr = Trajectory(t, pos, roll)

    p0, r0 = tr.interp(-1.0)
    assert np.allclose(p0, pos[0])

    p2, r2 = tr.interp(99.0)
    assert np.allclose(p2, pos[-1])

def test_orientation_orthonormal():
    t = np.linspace(0, 1.0, 101)
    pos = np.zeros((len(t), 3))
    roll = np.zeros((len(t), 3))
    roll[:, 2] = 1.0  # r = 1 rad/s yaw spin
    tr = Trajectory(t, pos, roll)

    Rs = tr.integrate_orientation()
    for R in Rs[::10]:
        assert np.linalg.norm(R.T @ R - np.eye(3)) < 1e-9
