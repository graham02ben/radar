# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 16:19:32 2026

@author: bboyg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trajectory import Trajectory
from radar_params import RadarParams
from scan_patterns import FixedPointingScan
from radar import Radar
from radar_face import RadarFace
from utils_frames import deg2rad
from rcs_simple import SimpleFlipRCS


def main():
    # =========================
    # Radar parameters
    # =========================
    rp = RadarParams(
        radar_lla=np.array([52.0, 0.185, 0.1]),
        f_hz=4e9,
        p_tx_w=2e6,
        g_lin=10 ** (45 / 10),
        l_sys_lin=10 ** (2 / 10),
        nf_dB=3.0,
        bw_hz=1e4,
        t0_k=290.0,
        bw3dB_az_deg=30.0,
        bw3dB_el_deg=30.0,
        snr_min_dB=0.0
    )

    # Stare at target direction (north, horizon)
    scan = FixedPointingScan(az_rad=0.0, el_rad=0.0, dwell_s=1.0)
    radar = Radar(params=rp, scan_pattern=scan)

    # Wide face so nothing is gated
    face = RadarFace(
        radar=radar,
        scan_pattern=scan,
        fov_az_min_rad=deg2rad(0),
        fov_az_max_rad=deg2rad(360),
        fov_el_min_rad=deg2rad(-20),
        fov_el_max_rad=deg2rad(90),
        max_range_m=50e3
    )

    # =========================
    # Target setup
    # =========================
    tgt_lla = np.array([52.009, 0.185, 0.1])  # ~1 km north

    # Time: 0..20 seconds at 1 Hz
    t = np.arange(0.0, 20.0 + 1e-9, 1.0)

    # Position constant
    pos = np.tile(tgt_lla, (len(t), 1))

    # Rotate 90 deg/s about BODY +X axis
    omega = np.pi / 2  # rad/s
    roll = np.zeros((len(t), 3), dtype=float)
    roll[:, 0] = omega  # p-rate

    traj = Trajectory(t=t, pos=pos, roll=roll)

    # Initial orientation: BODY aligned with ENU
    Rs = traj.integrate_orientation(R0_body_to_enu=np.eye(3))

    # =========================
    # RCS model
    # =========================
    rcs_model = SimpleFlipRCS(
        sigma_front_m2=100.0,
        sigma_back_m2=30.0,
        sigma_edge_m2=8.0
    )

    # =========================
    # Simulation loop
    # =========================
    rows = []

    for k, tk in enumerate(t):
    
        
        meas = face.dwell(
            t=float(tk),
            tgt_pos_lla=pos[k],
            rcs_model=rcs_model,
            tgt_body_to_enu=Rs[k]
        )

        R = Rs[k]

        row = {
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

            "R00": float(R[0, 0]), "R01": float(R[0, 1]), "R02": float(R[0, 2]),
            "R10": float(R[1, 0]), "R11": float(R[1, 1]), "R12": float(R[1, 2]),
            "R20": float(R[2, 0]), "R21": float(R[2, 1]), "R22": float(R[2, 2]),
        }

        rows.append(row)

        print(
            f"t={tk:4.0f}s | "
            f"az={np.degrees(row['az_t_rad']):6.1f}° "
            f"el={np.degrees(row['el_t_rad']):5.1f}° "
            f"sigma={row['sigma_true_m2']:6.1f} m² "
            f"snr={row['snr_dB']:6.1f} dB "
            f"det={row['detected']}"
        )

    df = pd.DataFrame(rows)
    df.to_csv("sim_output_flip.csv", index=False)
    print("\nSaved sim_output_flip.csv")


if __name__ == "__main__":
    main()
    df = pd.read_csv("sim_output_flip.csv")

    t = df["t"].to_numpy()
    sigma = df["sigma_true_m2"].to_numpy()
    snr = df["snr_dB"].to_numpy()
    
    plt.figure()
    plt.plot(t, 10*np.log10(sigma), marker="o")
    plt.xlabel("Time (s)")
    plt.ylabel("RCS (dBsm)")
    plt.title("RCS flip pattern (front/edge/back/edge)")
    plt.grid(True)

    plt.figure()
    plt.plot(t, snr, marker="o")
    plt.xlabel("Time (s)")
    plt.ylabel("SNR (dB)")
    plt.title("SNR vs time")
    plt.grid(True)

    plt.show()