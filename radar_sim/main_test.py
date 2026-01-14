# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 13:32:04 2026

@author: bboyg
"""

# main.py
import numpy as np
import pandas as pd

from trajectory import Trajectory
from radar_params import RadarParams
from scan_patterns import RasterScan, deg2rad
from radar import Radar
from radar_face import RadarFace
from rcs_model import RCSGrid, RCSModel
from tracking import TrackManager
from utils_frames import lla_to_ecef, ecef_to_enu_matrix



def trajectory_from_start_end_time(
    start_lat_deg: float,
    start_lon_deg: float,
    end_lat_deg: float,
    end_lon_deg: float,
    time_in_air_s: float,
    time_step_s: float = 0.1,
    start_alt_m: float = 0.0,
    end_alt_m: float = 0.0,
    apex_alt_m: float = 2000.0,
    roll_p_rad_s: float = 2.0,
    roll_q_cycle=None,
    roll_r_cycle=None,
) -> pd.DataFrame:
    """
    Generates a "ballistic-like" trajectory that exactly goes from start -> end in time_in_air_s.

    Horizontal motion: constant velocity in local ENU (computed from start->end).
    Vertical motion: parabola that starts at start_alt_m, ends at end_alt_m, peaks at apex_alt_m.

    Output columns:
        time_s, lat_deg, lon_deg, alt_km, p_rad_s, q_rad_s, r_rad_s
    """

    if roll_q_cycle is None:
        roll_q_cycle = [0.2, -0.05, -0.2, 0.05]
    if roll_r_cycle is None:
        roll_r_cycle = [0.05, -0.2, -0.05, 0.2]

    # --- Convert start/end into ECEF ---
    xs, ys, zs = lla_to_ecef(start_lat_deg, start_lon_deg, start_alt_m)
    xe, ye, ze = lla_to_ecef(end_lat_deg, end_lon_deg, end_alt_m)

    start_ecef = np.array([xs, ys, zs], dtype=float)
    end_ecef = np.array([xe, ye, ze], dtype=float)

    # --- Local ENU frame at start ---
    R_ecef_to_enu = ecef_to_enu_matrix(start_lat_deg, start_lon_deg)

    # ENU displacement from start to end
    d_ecef = end_ecef - start_ecef
    d_enu = R_ecef_to_enu @ d_ecef  # [east, north, up]

    # We will treat "up" separately with our vertical parabola
    east_total = d_enu[0]
    north_total = d_enu[1]

    T = float(time_in_air_s)
    if T <= 0:
        raise ValueError("time_in_air_s must be > 0")

    # Constant horizontal velocities in ENU
    v_e = east_total / T
    v_n = north_total / T

    # Time grid
    t = np.arange(0.0, T + 1e-9, time_step_s)

    # Horizontal positions in ENU
    east = v_e * t
    north = v_n * t

    # --- Vertical arc (parabola) ---
    # We want z(0)=start_alt_m, z(T)=end_alt_m, and z(T/2)=apex_alt_m (by default).
    # Use quadratic: z(t) = a t^2 + b t + c
    c = start_alt_m

    # Solve for a, b from constraints at t=T and t=T/2
    # a*T^2 + b*T + c = end_alt_m
    # a*(T/2)^2 + b*(T/2) + c = apex_alt_m
    A = np.array([[T**2, T],
                  [(T/2)**2, (T/2)]], dtype=float)
    y = np.array([end_alt_m - c,
                  apex_alt_m - c], dtype=float)
    a, b = np.linalg.solve(A, y)

    alt_m = a * t**2 + b * t + c
    alt_m = np.maximum(alt_m, 0.0)  # avoid negative due to numeric issues

    # --- Convert ENU back to ECEF, then to LLA (approx) ---
    # NOTE: ECEF->LLA is non-trivial; rather than implement a full inverse here,
    # we do a small-angle approximation in lat/lon for short ranges OR you can add ecef_to_lla later.
    #
    # For radar sims with local ranges (tens to hundreds of km), a local ENU->LLA approx is OK.
    # We'll compute delta lat/lon using Earth radius at start.
    Re = 6371000.0
    lat0_rad = np.deg2rad(start_lat_deg)

    d_lat = north / Re
    d_lon = east / (Re * np.cos(lat0_rad))

    lat_deg = start_lat_deg + np.rad2deg(d_lat)
    lon_deg = start_lon_deg + np.rad2deg(d_lon)

    # --- Roll rates ---
    p = np.full_like(t, roll_p_rad_s, dtype=float)
    q = np.array([roll_q_cycle[i % len(roll_q_cycle)] for i in range(len(t))], dtype=float)
    r = np.array([roll_r_cycle[i % len(roll_r_cycle)] for i in range(len(t))], dtype=float)

    df = pd.DataFrame({
        "time_s": t,
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "alt_km": alt_m / 1000.0,
        "p_rad_s": p,
        "q_rad_s": q,
        "r_rad_s": r,
    })

    return df

def main():
    # =========================
    # 1) Radar parameters (tuned for easy detections)
    # =========================
    rp = RadarParams(
        radar_lla=np.array([52.0, 0.185, 0.1]),  # [deg, deg, km]
        f_hz=4e9,
        p_tx_w=5e6,
        g_lin=10 ** (45 / 10),
        l_sys_lin=10 ** (2 / 10),
        nf_dB=3.0,
        bw_hz=1e4,
        t0_k=290.0,
        bw3dB_az_deg=20.0,
        bw3dB_el_deg=10.0,
        snr_min_dB=5.0
    )

    # =========================
    # 2) Scan pattern (covers the face)
    # =========================
    scan = RasterScan.from_degrees(
        az_min_deg=30, az_max_deg=90,
        el_min_deg=0, el_max_deg=60,
        az_rate_dps=40,
        el_step_deg=5,
        dwell_s=0.2
    )

    radar = Radar(params=rp, scan_pattern=scan)

    # =========================
    # 3) Radar face definition (FOV gate)
    # =========================
    face = RadarFace(
        radar=radar,
        scan_pattern=scan,
        fov_az_min_rad=deg2rad(30),
        fov_az_max_rad=deg2rad(90),
        fov_el_min_rad=deg2rad(0),
        fov_el_max_rad=deg2rad(60),
        max_range_m=100e3
    )

    # =========================
    # 4) RCS Model (constant, moderately large)
    # =========================
    phi_deg = np.arange(0, 360, 5)
    theta_deg = np.arange(0, 181, 2)
    sigma_m2 = np.ones((len(phi_deg), len(theta_deg))) * 50.0  # 50 m^2 everywhere

    grid = RCSGrid(phi_deg, theta_deg, sigma_m2)
    rcs_model = RCSModel(grid)

    # =========================
    # 5) Build trajectory near radar (guaranteed in-range)
    # =========================
    df_traj = trajectory_from_start_end_time(
        start_lat_deg=52.05, start_lon_deg=0.25,
        end_lat_deg=52.10,   end_lon_deg=0.40,
        time_in_air_s=60.0,
        time_step_s=0.2,
        start_alt_m=0.0,
        end_alt_m=0.0,
        apex_alt_m=5000.0,      # 5 km arc
        roll_p_rad_s=2.0
    )

    t = df_traj["time_s"].to_numpy()
    pos = df_traj[["lat_deg", "lon_deg", "alt_km"]].to_numpy()
    roll = df_traj[["p_rad_s", "q_rad_s", "r_rad_s"]].to_numpy()

    traj = Trajectory(t=t, pos=pos, roll=roll)
    Rs = traj.integrate_orientation()  # list of BODY->ENU matrices

    # =========================
    # 6) Tracker
    # =========================
    tm = TrackManager(
        gate_az_deg=5.0,
        gate_el_deg=5.0,
        gate_range_m=2000.0,
        promote_hits=3,
        drop_misses=10
    )

    # =========================
    # 7) Simulation loop
    # =========================
    out_rows = []
    last_t = float(t[0])

    for k in range(len(t)):
        tk = float(t[k])
        dt = max(tk - last_t, 1e-6)
        last_t = tk

        tm.predict_all(dt)

        pos_k = pos[k]
        R_body_to_enu = Rs[k]

        meas = face.dwell(
            t=tk,
            tgt_pos_lla=pos_k,
            rcs_model=rcs_model,
            tgt_body_to_enu=R_body_to_enu
        )

        # Feed tracker detections
        dets = []
        if meas and meas.get("detected", False):
            dets.append((meas["range_m"], meas["az_t_rad"], meas["el_t_rad"]))
        tm.associate_and_update(tk, radar_pos_enu=np.zeros(3), dets=dets)

        n_tracks = len(tm.tracks)
        n_confirmed = sum(1 for tr in tm.tracks.values() if tr.confirmed)

        # Save orientation matrix into CSV (Option B)
        R = R_body_to_enu
        row = {
            "t": tk,
            "lat_deg": float(pos_k[0]),
            "lon_deg": float(pos_k[1]),
            "alt_km": float(pos_k[2]),

            "detected": bool(meas.get("detected", False)) if meas else False,
            "snr_dB": float(meas.get("snr_dB", -999.0)) if meas else -999.0,
            "range_m": float(meas.get("range_m", np.nan)) if meas else np.nan,
            "az_cmd_rad": float(meas.get("az_cmd_rad", np.nan)) if meas else np.nan,
            "el_cmd_rad": float(meas.get("el_cmd_rad", np.nan)) if meas else np.nan,
            "az_t_rad": float(meas.get("az_t_rad", np.nan)) if meas else np.nan,
            "el_t_rad": float(meas.get("el_t_rad", np.nan)) if meas else np.nan,

            "sigma_true_m2": float(meas.get("sigma_true_m2", np.nan)) if meas else np.nan,
            "sigma_app_m2": float(meas.get("sigma_app_m2", np.nan)) if meas else np.nan,

            # BODY->ENU rotation matrix elements
            "R00": float(R[0, 0]), "R01": float(R[0, 1]), "R02": float(R[0, 2]),
            "R10": float(R[1, 0]), "R11": float(R[1, 1]), "R12": float(R[1, 2]),
            "R20": float(R[2, 0]), "R21": float(R[2, 1]), "R22": float(R[2, 2]),

            "tracks_total": int(n_tracks),
            "tracks_confirmed": int(n_confirmed),
        }

        out_rows.append(row)

        # Quick debug prints for the first 10 frames
        if k < 10 and meas:
            print(
                f"k={k:02d}",
                f"t={tk:5.1f}s",
                f"R={meas['range_m']/1000:6.2f}km",
                f"az={np.degrees(meas['az_t_rad']):6.1f}°",
                f"el={np.degrees(meas['el_t_rad']):6.1f}°",
                f"snr={meas['snr_dB']:6.1f}dB",
                f"det={meas['detected']}"
            )

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv("sim_output.csv", index=False)

    print("Saved sim_output.csv")
    print("Detections:", int(out_df["detected"].sum()))
    print("Final tracks:", len(tm.tracks), "confirmed:", sum(1 for tr in tm.tracks.values() if tr.confirmed))


if __name__ == "__main__":
    main()