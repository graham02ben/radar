# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 16:17:18 2026

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


def test_stationary_flipping_object_detects_and_cycles_rcs():
    # --- Radar settings: stare straight at target, 1 dwell per second ---
    rp = RadarParams(
        radar_lla=np.array([52.0, 0.185, 0.1]),  # UK-ish radar site
        f_hz=4e9,
        p_tx_w=2e6,
        g_lin=10 ** (45 / 10),
        l_sys_lin=10 ** (2 / 10),
        nf_dB=3.0,
        bw_hz=1e4,
        t0_k=290.0,
        bw3dB_az_deg=30.0,
        bw3dB_el_deg=30.0,
        snr_min_dB=0.0  # keep easy for this test
    )

    # Stare north on the horizon (target will be north)
    scan = FixedPointingScan(az_rad=0.0, el_rad=0.0, dwell_s=1.0)
    radar = Radar(params=rp, scan_pattern=scan)

    # Face with "wide open" FOV so gating can't break the test
    face = RadarFace(
        radar=radar,
        scan_pattern=scan,
        fov_az_min_rad=deg2rad(0),
        fov_az_max_rad=deg2rad(360),
        fov_el_min_rad=deg2rad(-20),
        fov_el_max_rad=deg2rad(90),
        max_range_m=50e3
    )

    # --- Target: ~1 km north of radar ---
    tgt_lla = np.array([52.009, 0.185, 0.1])  # about 1 km north

    # --- Trajectory: stationary position, rotating 90 deg/s around BODY +X ---
    # 20 seconds inclusive: t = 0..20 at 1 Hz -> 21 frames
    t = np.arange(0.0, 20.0 + 1e-9, 1.0)

    pos = np.tile(tgt_lla, (len(t), 1))  # constant position

    omega = np.pi / 2  # rad/s -> 90 deg/s
    roll = np.zeros((len(t), 3), dtype=float)
    roll[:, 0] = omega  # p = omega, q=r=0

    traj = Trajectory(t=t, pos=pos, roll=roll)

    # Make R0 so BODY +Y points toward LOS at t=0
    # If radar LOS is approximately +North in ENU, we want body +Y aligned with +North (ENU y-axis).
    # That can be done with an identity matrix if you interpret body axes as ENU at start.
    Rs = traj.integrate_orientation(R0_body_to_enu=np.eye(3))

    # --- RCS: front high, back medium, edge small but detectable ---
    rcs_model = SimpleFlipRCS(
        sigma_front_m2=100.0,
        sigma_back_m2=30.0,
        sigma_edge_m2=8.0
    )

    # Expected pattern at integer seconds:
    # t=0 front, t=1 edge, t=2 back, t=3 edge, t=4 front, ...
    expected = []
    for k in range(len(t)):
        phase = k % 4
        if phase == 0:
            expected.append("front")
        elif phase == 1:
            expected.append("edge")
        elif phase == 2:
            expected.append("back")
        else:
            expected.append("edge")

    # Run dwell and check:
    sigmas = []
    detects = []

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

    # All should be detectable (edge is small but still above threshold in this tuned setup)
    assert all(detects), "Expected detection every second, but some seconds were missed."

    # Check relative levels follow the intended cycle
    # (Allow some tolerance because beam gain & geometry can slightly modulate)
    front_vals = [s for s, tag in zip(sigmas, expected) if tag == "front"]
    back_vals  = [s for s, tag in zip(sigmas, expected) if tag == "back"]
    edge_vals  = [s for s, tag in zip(sigmas, expected) if tag == "edge"]

    assert np.median(front_vals) > np.median(back_vals) > np.median(edge_vals), \
        "Expected sigma_front > sigma_back > sigma_edge ordering."
