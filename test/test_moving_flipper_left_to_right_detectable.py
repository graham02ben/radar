# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 17:52:19 2026

@author: bboyg
"""

import numpy as np

from trajectory import Trajectory
from radar_params import RadarParams
from scan_patterns import FixedPointingScan
from radar import Radar
from radar_face import RadarFace
from utils_frames import deg2rad
from rcs_simple import SimpleFlipRCS


def test_moving_flipper_left_to_right_detectable():
    # --- Radar settings: stare, 1 dwell per second ---
    rp = RadarParams(
        radar_lla=np.array([52.0, 0.185, 0.1]),
        f_hz=4e9,
        p_tx_w=2e6,
        g_lin=10 ** (45 / 10),
        l_sys_lin=10 ** (2 / 10),
        nf_dB=3.0,
        bw_hz=1e4,
        t0_k=290.0,
        bw3dB_az_deg=60.0,   # make beam wider so a moving target stays illuminated
        bw3dB_el_deg=60.0,
        snr_min_dB=0.0
    )

    # Stare north (target is generally north-ish)
    scan = FixedPointingScan(az_rad=0.0, el_rad=0.0, dwell_s=1.0)
    radar = Radar(params=rp, scan_pattern=scan)

    # Wide face: allow az sweep across a chunk of sky
    face = RadarFace(
        radar=radar,
        scan_pattern=scan,
        fov_az_min_rad=deg2rad(300),   # -60 deg
        fov_az_max_rad=deg2rad(60),    # +60 deg (wrap window)
        fov_el_min_rad=deg2rad(-20),
        fov_el_max_rad=deg2rad(90),
        max_range_m=80e3
    )

    # --- Time setup ---
    t = np.arange(0.0, 20.0 + 1e-9, 1.0)  # 0..20 inclusive at 1 Hz

    # --- Motion: move east ("left to right") at constant velocity ---
    # Start about 1 km north of radar, slightly west of radar's longitude
    lat = 52.009
    alt_km = 0.1

    # Move ~10 km east over 20 s
    # 10 km at lat 52 => delta_lon ≈ 10km / (111km*cos(lat)) ≈ 0.145 degrees
    total_east_km = 10.0
    delta_lon_deg = total_east_km / (111.0 * np.cos(np.deg2rad(lat)))

    lon0 = 0.185 - delta_lon_deg / 2  # start west of center
    lon1 = 0.185 + delta_lon_deg / 2  # end east of center

    lon = np.linspace(lon0, lon1, len(t))

    pos = np.column_stack([
        np.full_like(t, lat, dtype=float),
        lon.astype(float),
        np.full_like(t, alt_km, dtype=float)
    ])

    # --- Flip/rotation: 90 deg per second around BODY +X ---
    omega = np.pi / 2
    roll = np.zeros((len(t), 3), dtype=float)
    roll[:, 0] = omega

    traj = Trajectory(t=t, pos=pos, roll=roll)
    Rs = traj.integrate_orientation(R0_body_to_enu=np.eye(3))

    # --- RCS: front high, back medium, edge low but detectable ---
    rcs_model = SimpleFlipRCS(
        sigma_front_m2=100.0,
        sigma_back_m2=30.0,
        sigma_edge_m2=8.0
    )

    # Expected flip cycle
    expected = []
    for k in range(len(t)):
        phase = k % 4
        expected.append("front" if phase == 0 else "back" if phase == 2 else "edge")

    sigmas = []
    detects = []
    azs = []
    els = []
    ranges = []
    snrs = []

    for k, tk in enumerate(t):
        meas = face.dwell(
            t=float(tk),
            tgt_pos_lla=pos[k],
            rcs_model=rcs_model,
            tgt_body_to_enu=Rs[k]
        )

        assert meas is not None
        sigmas.append(meas["sigma_true_m2"])
        detects.append(meas["detected"])
        azs.append(meas["az_t_rad"])
        els.append(meas["el_t_rad"])
        ranges.append(meas["range_m"])
        snrs.append(meas["snr_dB"])

    # --- Assertions ---
    # We want it to be detectable MOST of the time (since it moves through the beam)
    det_rate = np.mean(detects)
    assert det_rate >= 0.7, f"Detection rate too low: {det_rate*100:.1f}%"

    # Still ensure the RCS ordering shows up when detected
    sigmas = np.array(sigmas, dtype=float)
    expected = np.array(expected)

    front_vals = sigmas[(expected == "front")]
    back_vals  = sigmas[(expected == "back")]
    edge_vals  = sigmas[(expected == "edge")]

    assert np.median(front_vals) > np.median(back_vals) > np.median(edge_vals), \
        "Expected sigma_front > sigma_back > sigma_edge ordering."

    # Sanity: az should change over time due to left-to-right motion
    azs = np.unwrap(np.array(azs, dtype=float))  # unwrap for monotonic check
    assert abs(azs[-1] - azs[0]) > np.deg2rad(5.0), "Azimuth did not change enough; target may not be moving across LOS."
