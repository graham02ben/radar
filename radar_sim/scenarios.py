# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 17:52:42 2026

@author: bboyg
"""

import numpy as np
import pandas as pd

from trajectory import Trajectory
from radar_params import RadarParams
from scan_patterns import FixedPointingScan
from radar import Radar
from radar_face import RadarFace
from utils_frames import deg2rad
from rcs_simple import SimpleFlipRCS


def run_moving_flipper_scenario(t_end_s=20.0, dt_s=1.0) -> pd.DataFrame:
    # Radar
    rp = RadarParams(
        radar_lla=np.array([52.0, 0.185, 0.1]),
        f_hz=4e9,
        p_tx_w=2e6,
        g_lin=10 ** (45 / 10),
        l_sys_lin=10 ** (2 / 10),
        nf_dB=3.0,
        bw_hz=1e4,
        t0_k=290.0,
        bw3dB_az_deg=5.0,
        bw3dB_el_deg=3.0,
        snr_min_dB=0.0
    )

    scan = FixedPointingScan(az_rad=deg2rad(125.0), el_rad=deg2rad(5), dwell_s=dt_s)
    radar = Radar(params=rp, scan_pattern=scan)

    face = RadarFace(
        radar=radar,
        scan_pattern=scan,
        fov_az_min_rad=deg2rad(100),  # wrap window: -60 to +60
        fov_az_max_rad=deg2rad(150),
        fov_el_min_rad=deg2rad(-10),
        fov_el_max_rad=deg2rad(10),
        max_range_m=80e3
    )

    # Time
    t = np.arange(0.0, t_end_s + 1e-9, dt_s)

    # Motion: fixed range, azimuth sweeps 100° → 150°
    radar_lat, radar_lon, radar_alt = rp.radar_lla
    range_km = 20.0
    az_start_deg = 100.0
    az_end_deg = 150.0
    el_deg = 5.0

    az_deg = np.linspace(az_start_deg, az_end_deg, len(t))
    el_rad = np.deg2rad(el_deg)
    az_rad = np.deg2rad(az_deg)

    # Convert spherical (range, az, el) in ENU to delta ENU
    r = range_km * 1000.0
    east = r * np.sin(az_rad) * np.cos(el_rad)
    north = r * np.cos(az_rad) * np.cos(el_rad)
    up = np.full_like(az_rad, r * np.sin(el_rad), dtype=float)  # array length len(t)

    # Convert ENU offsets to lat/lon deltas (small-angle approx)
    Re = 6371000.0
    dlat = north / Re
    dlon = east / (Re * np.cos(np.deg2rad(radar_lat)))

    lat = radar_lat + np.rad2deg(dlat)
    lon = radar_lon + np.rad2deg(dlon)
    alt_km = radar_alt + up / 1000.0  # now also an array

    pos = np.column_stack([lat, lon, alt_km])

    # Flip: 90 deg/s around +X
    omega = np.pi / 2
    roll = np.zeros((len(t), 3), dtype=float)
    roll[:, 0] = omega

    traj = Trajectory(t=t, pos=pos, roll=roll)
    Rs = traj.integrate_orientation(R0_body_to_enu=np.eye(3))

    rcs_model = SimpleFlipRCS(100.0, 30.0, 8.0)

    rows = []
    for k, tk in enumerate(t):
        meas = face.dwell(
            t=float(tk),
            tgt_pos_lla=pos[k],
            rcs_model=rcs_model,
            tgt_body_to_enu=Rs[k]
        )

        R = Rs[k]
        rows.append({
            "t": float(tk),
            "lat_deg": float(pos[k, 0]),
            "lon_deg": float(pos[k, 1]),
            "alt_km": float(pos[k, 2]),

            "detected": bool(meas.get("detected", False)) if meas else False,
            "snr_dB": float(meas.get("snr_dB", np.nan)) if meas else np.nan,
            "range_m": float(meas.get("range_m", np.nan)) if meas else np.nan,
            "az_cmd_rad": float(meas.get("az_cmd_rad", np.nan)) if meas else np.nan,
            "el_cmd_rad": float(meas.get("el_cmd_rad", np.nan)) if meas else np.nan,
            "az_t_rad": float(meas.get("az_t_rad", np.nan)) if meas else np.nan,
            "el_t_rad": float(meas.get("el_t_rad", np.nan)) if meas else np.nan,
            "sigma_true_m2": float(meas.get("sigma_true_m2", np.nan)) if meas else np.nan,
            "sigma_app_m2": float(meas.get("sigma_app_m2", np.nan)) if meas else np.nan,

            # Save orientation for optional animation later
            "R00": float(R[0, 0]), "R01": float(R[0, 1]), "R02": float(R[0, 2]),
            "R10": float(R[1, 0]), "R11": float(R[1, 1]), "R12": float(R[1, 2]),
            "R20": float(R[2, 0]), "R21": float(R[2, 1]), "R22": float(R[2, 2]),
        })

    return pd.DataFrame(rows)
