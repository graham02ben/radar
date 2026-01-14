# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 08:46:31 2026

@author: bboyg
"""

import numpy as np
import pandas as pd

from trajectory import Trajectory
from radar_params import RadarParams
from scan_patterns import RasterScan, deg2rad
from radar import Radar
from radar_face import RadarFace
from rcs_model import RCSGrid, RCSModel
from tracking import TrackManager


# ------------------------------------------------------------
# Optional: simple vacuum ballistic trajectory generator
# Put this into its own file later if you want (physics.py)
# ------------------------------------------------------------
def ballistic_trajectory(
    start_lat, start_lon,
    launch_angle_deg,
    azimuth_deg,
    initial_velocity,
    time_step=0.1,
    max_time=1000.0
):
    import math

    g = 9.80665
    earth_radius = 6371000.0

    lat0 = math.radians(start_lat)
    lon0 = math.radians(start_lon)

    launch_angle = math.radians(launch_angle_deg)
    azimuth = math.radians(azimuth_deg)

    v_horizontal = initial_velocity * math.cos(launch_angle)
    v_vertical = initial_velocity * math.sin(launch_angle)

    v_north = v_horizontal * math.cos(azimuth)
    v_east = v_horizontal * math.sin(azimuth)

    # Example roll rates (rad/s)
    p_value = 2.0
    q_cycle = [0.2, -0.05, -0.2, 0.05]
    r_cycle = [0.05, -0.2, -0.05, 0.2]

    rows = []
    t = 0.0
    step_index = 0

    while t <= max_time:
        alt_m = v_vertical * t - 0.5 * g * t**2
        if alt_m < 0:
            break

        d_north = v_north * t
        d_east = v_east * t

        d_lat = d_north / earth_radius
        lat_now = lat0 + d_lat

        # IMPORTANT: use updated latitude for longitude conversion
        d_lon = d_east / (earth_radius * math.cos(lat_now))

        lat_deg = math.degrees(lat_now)
        lon_deg = math.degrees(lon0 + d_lon)

        p = p_value
        q = q_cycle[step_index % len(q_cycle)]
        r = r_cycle[step_index % len(r_cycle)]

        rows.append([t, lat_deg, lon_deg, alt_m / 1000.0, p, q, r])

        step_index += 1
        t += time_step

    return pd.DataFrame(
        rows,
        columns=["time_s", "lat_deg", "lon_deg", "alt_km", "p_rad_s", "q_rad_s", "r_rad_s"]
    )


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


# ------------------------------------------------------------
# Optional: mock RCS generator (dBsm)
# Put this into its own file later if you want (rcs_generators.py)
# ------------------------------------------------------------
def generate_mock_rcs(num_phi=180, num_theta=181,
                      front_dB=20.0, edge_dB=0.0, back_dB=10.0,
                      decay_rate=0.03):
    """
    Returns sigma_dBsm with shape (num_phi, num_theta).
    phi dimension is "samples", theta is polar angle 0..180.
    """
    theta = np.linspace(0, 180, num_theta)

    peaks = np.array([0, 90, 180])
    peak_vals = np.array([front_dB, edge_dB, back_dB])

    base_lin = np.zeros_like(theta, dtype=float)
    for peak_angle, peak_db in zip(peaks, peak_vals):
        base_lin += 10 ** (peak_db / 10.0) * np.exp(-decay_rate * np.abs(theta - peak_angle))

    base_db = 10.0 * np.log10(base_lin + 1e-30)

    sigma_dBsm = []
    for _ in range(num_phi):
        noise = np.random.normal(0, 0.5, size=num_theta)
        sigma_dBsm.append(base_db + noise)

    return np.array(sigma_dBsm, dtype=float)


def main():
    # =========================
    # 1) Radar parameters
    # =========================
    rp = RadarParams(
        radar_lla=np.array([52.0, 0.185, 0.1]),  # [deg,deg,km]
        f_hz=4e9,
        p_tx_w=1e6,
        g_lin=10 ** (45 / 10),
        l_sys_lin=10 ** (2 / 10),
        nf_dB=3.0,
        bw_hz=1e4,
        t0_k=290.0,
        bw3dB_az_deg=20.0,
        bw3dB_el_deg=10.0,
        snr_min_dB=0.0
    )

    # =========================
    # 2) Scan pattern (radians internally)
    # =========================
    scan = RasterScan.from_degrees(
        az_min_deg=250, az_max_deg=300,
        el_min_deg=0, el_max_deg=90,
        az_rate_dps=70,
        el_step_deg=5,
        dwell_s=0.1
    )

    radar = Radar(params=rp, scan_pattern=scan)

    # =========================
    # 3) Radar face (FOV gate)
    # =========================
    face = RadarFace(
        radar=radar,
        scan_pattern=scan,
        fov_az_min_rad=deg2rad(250),
        fov_az_max_rad=deg2rad(300),
        fov_el_min_rad=deg2rad(0),
        fov_el_max_rad=deg2rad(90),
        max_range_m=2500e3
    )

    # =========================
    # 4) RCS Model
    # =========================
    phi_deg = np.arange(0, 360, 2)         # 0..358
    theta_deg = np.arange(0, 181, 1)       # 0..180

    sigma_dBsm = generate_mock_rcs(num_phi=len(phi_deg), num_theta=len(theta_deg))
    # Ensure shape is (len(phi_deg), len(theta_deg))
    assert sigma_dBsm.shape == (len(phi_deg), len(theta_deg))

    grid = RCSGrid.from_sigma_dBsm(phi_deg, theta_deg, sigma_dBsm)
    rcs_model = RCSModel(grid)

    # =========================
    # 5) Trajectory + orientation
    # =========================
    # df_traj = ballistic_trajectory(
    #     start_lat=33.30182723,
    #     start_lon=-37.422133969,
    #     launch_angle_deg=45,
    #     azimuth_deg=45,
    #     initial_velocity=6000,
    #     time_step=0.1,
    #     max_time=1000.0
    # )
    
    df_traj = trajectory_from_start_end_time(
        start_lat_deg=33.30182723, 
        start_lon_deg=-37.422133969,
        end_lat_deg=51.363817,
        end_lon_deg=-1.149837,
        time_in_air_s=900.0,
        time_step_s=0.1,
        apex_alt_m=500000.0
    )

    t = df_traj["time_s"].to_numpy()
    pos = df_traj[["lat_deg", "lon_deg", "alt_km"]].to_numpy()
    roll = df_traj[["p_rad_s", "q_rad_s", "r_rad_s"]].to_numpy()

    traj = Trajectory(t=t, pos=pos, roll=roll)
    Rs = traj.integrate_orientation()  # list of BODY->ENU

    # =========================
    # 6) Tracker
    # =========================
    tm = TrackManager(
        gate_az_deg=5.0,
        gate_el_deg=5.0,
        gate_range_m=1000.0,
        promote_hits=3,
        drop_misses=20
    )

    # =========================
    # 7) Simulation loop
    # =========================
    out_rows = []

    last_t = t[0]
    for k in range(len(t)):
        tk = t[k]
        dt = max(tk - last_t, 1e-6)
        last_t = tk

        tm.predict_all(dt)

        pos_k = pos[k]
        R_body_to_enu = Rs[k]
        R = R_body_to_enu
        R00, R01, R02 = R[0, 0], R[0, 1], R[0, 2]
        R10, R11, R12 = R[1, 0], R[1, 1], R[1, 2]
        R20, R21, R22 = R[2, 0], R[2, 1], R[2, 2]

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

        # Track bookkeeping (optional)
        n_tracks = len(tm.tracks)
        n_confirmed = sum(1 for tr in tm.tracks.values() if tr.confirmed)

        out_rows.append({
            "t": tk,
            "detected": bool(meas.get("detected", False)) if meas else False,
            "snr_dB": meas.get("snr_dB", -999.0) if meas else -999.0,
            "range_m": meas.get("range_m", np.nan) if meas else np.nan,
            "az_cmd_rad": meas.get("az_cmd_rad", np.nan) if meas else np.nan,
            "el_cmd_rad": meas.get("el_cmd_rad", np.nan) if meas else np.nan,
            "az_t_rad": meas.get("az_t_rad", np.nan) if meas else np.nan,
            "el_t_rad": meas.get("el_t_rad", np.nan) if meas else np.nan,
            "sigma_true_m2": meas.get("sigma_true_m2", np.nan) if meas else np.nan,
            
            # --- Orientation matrix BODY->ENU (saved for animation) ---
            "R00": R00, "R01": R01, "R02": R02,
            "R10": R10, "R11": R11, "R12": R12,
            "R20": R20, "R21": R21, "R22": R22,
            
            "tracks_total": n_tracks,
            "tracks_confirmed": n_confirmed
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv("sim_output.csv", index=False)

    print("Saved sim_output.csv")
    print("Detections:", int(out_df["detected"].sum()))
    print("Final tracks:", len(tm.tracks), "confirmed:", sum(1 for tr in tm.tracks.values() if tr.confirmed))


if __name__ == "__main__":
    main()

