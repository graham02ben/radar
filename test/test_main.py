# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 10:31:15 2026

@author: bboyg
"""

import numpy as np
from utils_frames import lla_to_ecef
from radar_params import RadarParams
from radar import Radar
from radar_face import RadarFace
from scan_patterns import FixedPointingScan, deg2rad
from rcs_model import RCSGrid, RCSModel

def haversine_range_m(lat1, lon1, lat2, lon2):
    # rough surface distance (good enough for sanity)
    import math
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def test_radar_and_target_not_thousands_km_apart():
    radar_lat, radar_lon = 52.0, 0.185
    tgt_lat, tgt_lon = 33.30182723, -37.422133969

    surface = haversine_range_m(radar_lat, radar_lon, tgt_lat, tgt_lon)
    # This will FAIL with your current numbers (that’s the point)
    assert surface < 2_500_000, f"Target is {surface/1000:.1f} km away (outside 2500 km max range)"

def test_face_fov_accepts_known_direction():
    rp = RadarParams(
        radar_lla=np.array([52.0, 0.185, 0.1]),
        f_hz=4e9, p_tx_w=1e6, g_lin=10**(45/10),
        l_sys_lin=1.0, nf_dB=3.0, bw_hz=1e4, t0_k=290.0,
        bw3dB_az_deg=20.0, bw3dB_el_deg=10.0, snr_min_dB=-999.0
    )

    scan = FixedPointingScan(az_rad=deg2rad(240), el_rad=deg2rad(10), dwell_s=0.1)
    radar = Radar(rp, scan)

    face = RadarFace(
        radar=radar, scan_pattern=scan,
        fov_az_min_rad=deg2rad(210), fov_az_max_rad=deg2rad(270),
        fov_el_min_rad=deg2rad(0),   fov_el_max_rad=deg2rad(90),
        max_range_m=2500e3
    )

    # constant RCS
    phi = np.arange(0, 360, 10)
    theta = np.arange(0, 181, 1)
    sigma = np.ones((len(phi), len(theta))) * 10.0
    rcs = RCSModel(RCSGrid(phi, theta, sigma))

    # Put target ~1 km in front of face direction (roughly north-ish from radar)
    tgt = np.array([52.009, 0.185, 0.1])

    meas = face.dwell(0.0, tgt, rcs, tgt_body_to_enu=np.eye(3))
    assert meas is not None
    # If target is inside FOV, meas should not be forcibly set to detected=False by face
    assert np.isfinite(meas["az_t_rad"])
    assert np.isfinite(meas["el_t_rad"])

def test_detection_happens_in_easy_case():
    rp = RadarParams(
        radar_lla=np.array([52.0, 0.185, 0.1]),
        f_hz=4e9, p_tx_w=1e9, g_lin=10**(55/10),
        l_sys_lin=1.0, nf_dB=0.0, bw_hz=1e3, t0_k=290.0,
        bw3dB_az_deg=60.0, bw3dB_el_deg=60.0, snr_min_dB=0.0
    )

    scan = FixedPointingScan(az_rad=0.0, el_rad=0.0, dwell_s=1.0)
    radar = Radar(rp, scan)

    face = RadarFace(
        radar=radar, scan_pattern=scan,
        fov_az_min_rad=deg2rad(-180), fov_az_max_rad=deg2rad(180),
        fov_el_min_rad=deg2rad(-10),  fov_el_max_rad=deg2rad(90),
        max_range_m=50e3
    )

    # huge RCS
    phi = np.arange(0, 360, 10)
    theta = np.arange(0, 181, 1)
    sigma = np.ones((len(phi), len(theta))) * 1e6
    rcs = RCSModel(RCSGrid(phi, theta, sigma))

    tgt = np.array([52.009, 0.185, 0.1])  # ~1 km north
    meas = face.dwell(0.0, tgt, rcs, tgt_body_to_enu=np.eye(3))

    assert meas["detected"] is True

