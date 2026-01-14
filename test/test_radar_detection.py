# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 10:10:52 2026

@author: bboyg
"""

import numpy as np
from radar_params import RadarParams
from scan_patterns import FixedPointingScan
from radar import Radar
from rcs_model import RCSGrid, RCSModel

def test_radar_detects_when_snr_high():
    rp = RadarParams(
        radar_lla=np.array([52.0, 0.185, 0.1]),
        f_hz=4e9,
        p_tx_w=1e9,                 # huge power
        g_lin=10**(55/10),          # huge gain
        l_sys_lin=1.0,
        nf_dB=0.0,
        bw_hz=1e3,
        t0_k=290.0,
        bw3dB_az_deg=30.0,
        bw3dB_el_deg=30.0,
        snr_min_dB=10.0
    )

    # Point beam north, target north
    scan = FixedPointingScan(az_rad=0.0, el_rad=0.0, dwell_s=1.0)
    radar = Radar(rp, scan)

    # Very large RCS
    phi = np.arange(0, 360, 10)
    theta = np.arange(0, 181, 1)
    sigma = np.ones((len(phi), len(theta))) * 1e6  # 1,000,000 m^2
    rcs = RCSModel(RCSGrid(phi, theta, sigma))

    tgt = np.array([52.009, 0.185, 0.1])

    meas = radar.dwell(0.0, tgt, rcs, np.eye(3))
    assert meas["snr_dB"] > 10.0
    assert meas["detected"] is True
