# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 10:10:20 2026

@author: bboyg
"""

import numpy as np
from radar_params import RadarParams
from scan_patterns import FixedPointingScan
from radar import Radar
from rcs_model import RCSGrid, RCSModel

def test_radar_geometry_returns_finite_angles():
    rp = RadarParams(
        radar_lla=np.array([52.0, 0.185, 0.1]),
        f_hz=4e9,
        p_tx_w=1e6,
        g_lin=10**(45/10),
        l_sys_lin=10**(2/10),
        nf_dB=3.0,
        bw_hz=1e4,
        t0_k=290.0,
        bw3dB_az_deg=20.0,
        bw3dB_el_deg=10.0,
        snr_min_dB=-999.0
    )

    scan = FixedPointingScan(az_rad=0.0, el_rad=0.0, dwell_s=0.1)
    radar = Radar(rp, scan)

    # Constant RCS 10 m^2
    phi = np.arange(0, 360, 10)
    theta = np.arange(0, 181, 1)
    sigma = np.ones((len(phi), len(theta))) * 10.0
    rcs = RCSModel(RCSGrid(phi, theta, sigma))

    # Target ~1km north (roughly +0.009 deg latitude)
    tgt = np.array([52.009, 0.185, 0.1])

    meas = radar.dwell(t=0.0, tgt_pos_lla=tgt, rcs_model=rcs, tgt_body_to_enu=np.eye(3))
    assert meas is not None
    assert meas["range_m"] > 100.0
    assert np.isfinite(meas["az_t_rad"])
    assert np.isfinite(meas["el_t_rad"])
