# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 12:52:22 2026

@author: bboyg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Ellipse

# If you want the 3D disc animation:
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_output(csv_path="sim_output.csv"):
    df = pd.read_csv(csv_path)
    return df

def load_Rs_from_csv(df):
    Rs = []
    for _, row in df.iterrows():
        R = np.array([
            [row["R00"], row["R01"], row["R02"]],
            [row["R10"], row["R11"], row["R12"]],
            [row["R20"], row["R21"], row["R22"]],
        ], dtype=float)
        Rs.append(R)
    return Rs


def plot_az_el(df):
    """Plot target azimuth vs elevation (degrees)."""
    az = np.degrees(df["az_t_rad"].to_numpy())
    el = np.degrees(df["el_t_rad"].to_numpy())

    m = np.isfinite(az) & np.isfinite(el)

    plt.figure()
    plt.plot(az[m], el[m])
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Elevation (deg)")
    plt.title("Target Azimuth vs Elevation")
    plt.grid(True)
    plt.show()


def plot_rcs_vs_time(df):
    """Plot True vs Apparent RCS in dBsm (only where finite/positive)."""
    t = df["t"].to_numpy()

    sig_true = df.get("sigma_true_m2", pd.Series(np.nan, index=df.index)).to_numpy()
    sig_app = df.get("sigma_app_m2", pd.Series(np.nan, index=df.index)).to_numpy()
    det = df["detected"].to_numpy().astype(bool)

    # Convert to dBsm carefully
    def to_dBsm(x):
        x = np.asarray(x)
        out = np.full_like(x, np.nan, dtype=float)
        m = np.isfinite(x) & (x > 0)
        out[m] = 10.0 * np.log10(x[m])
        return out

    true_db = to_dBsm(sig_true)
    app_db = to_dBsm(sig_app)

    plt.figure(figsize=(10, 4))
    plt.plot(t, app_db, ".", label="Apparent RCS (dBsm)")
    plt.plot(t[det], true_db[det], ".", label="True RCS (dBsm) (detected frames)")
    plt.xlabel("Time (s)")
    plt.ylabel("RCS (dBsm)")
    plt.title("Apparent vs True RCS along flight path")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_beam_vs_target(df, beamwidth_az_deg=20.0, beamwidth_el_deg=10.0):
    """
    Recreate your 2D 'beam circle/ellipse' plot with slider + pause.
    Uses:
      az_cmd_rad, el_cmd_rad, az_t_rad, el_t_rad, detected
    """
    t = df["t"].to_numpy()

    az_target = np.degrees(df["az_t_rad"].to_numpy())
    el_target = np.degrees(df["el_t_rad"].to_numpy())
    az_beam = np.degrees(df["az_cmd_rad"].to_numpy())
    el_beam = np.degrees(df["el_cmd_rad"].to_numpy())
    detected = df["detected"].to_numpy().astype(bool)
    rng_km = df["range_m"].to_numpy() / 1000.0

    # Mask frames with valid target angles
    valid = np.isfinite(az_target) & np.isfinite(el_target) & np.isfinite(az_beam) & np.isfinite(el_beam)
    frames = np.where(valid)[0]
    if len(frames) == 0:
        print("No valid frames to plot (az/el are NaN).")
        return

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title("Radar Beam vs Target Position")
    ax.grid(True)

    # Keep bounds sensible based on data
    ax.set_xlim(np.nanmin(az_target[frames]) - 5, np.nanmax(az_target[frames]) + 5)
    ax.set_ylim(np.nanmin(el_target[frames]) - 5, np.nanmax(el_target[frames]) + 5)

    target_dot, = ax.plot([], [], "ro", label="Target")

    # Use Ellipse instead of Circle (Circle ignores width/height)
    beam_ellipse = Ellipse((0, 0), width=beamwidth_az_deg, height=beamwidth_el_deg,
                           alpha=0.3)
    ax.add_patch(beam_ellipse)

    detection_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                             va="top", bbox=dict(facecolor="white", alpha=0.6))
    time_text = ax.text(0.02, 0.85, "", transform=ax.transAxes,
                        va="top", bbox=dict(facecolor="white", alpha=0.6))
    dist_text = ax.text(0.02, 0.75, "", transform=ax.transAxes,
                        va="top", bbox=dict(facecolor="white", alpha=0.6))

    ax.legend()
    plt.subplots_adjust(bottom=0.25)

    ax_slider = plt.axes([0.2, 0.12, 0.6, 0.03])
    slider = Slider(ax_slider, "Frame", valmin=int(frames[0]), valmax=int(frames[-1]),
                    valinit=int(frames[0]), valstep=1)

    ax_button = plt.axes([0.42, 0.03, 0.16, 0.06])
    btn = Button(ax_button, "Pause")

    state = {"paused": False}
    ani_holder = {"ani": None}

    def update(i):
        i = int(i)
        target_dot.set_data(az_target[i], el_target[i])
        beam_ellipse.center = (az_beam[i], el_beam[i])

        if detected[i]:
            beam_ellipse.set_facecolor("red")
        else:
            beam_ellipse.set_facecolor("yellow")

        detection_text.set_text(f"Detections: {int(np.sum(detected[:i+1]))}")
        time_text.set_text(f"Time: {t[i]:.2f} s")
        dist_text.set_text(f"Range: {rng_km[i]:.2f} km")
        return target_dot, beam_ellipse, detection_text, time_text, dist_text

    ani = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)
    ani_holder["ani"] = ani

    def on_slider(val):
        update(int(slider.val))
        fig.canvas.draw_idle()

    slider.on_changed(on_slider)

    def toggle(event):
        if state["paused"]:
            ani.event_source.start()
            btn.label.set_text("Pause")
        else:
            ani.event_source.stop()
            btn.label.set_text("Play")
        state["paused"] = not state["paused"]

    btn.on_clicked(toggle)
    plt.show()


# ------------------------------------------------------------
# Optional: 3D disc orientation animation (needs Rs saved/available)
# ------------------------------------------------------------
def animate_disc_orientations(Rs, df):
    """
    Recreates the rotating disc visual.
    Rs: list of 3x3 rotations BODY->ENU (from Trajectory.integrate_orientation()).
    df: sim output df (for slider frames).
    """
    az_target = np.degrees(df["az_t_rad"].to_numpy())
    el_target = np.degrees(df["el_t_rad"].to_numpy())
    valid = np.isfinite(az_target) & np.isfinite(el_target)
    frames = np.where(valid)[0]
    if len(frames) == 0:
        print("No valid frames for disc animation.")
        return

    # Disc mesh in BODY frame: XY plane
    radius = 1.0
    n = 80
    th = np.linspace(0, 2*np.pi, n)
    rr = np.linspace(0, radius, n)
    th, rr = np.meshgrid(th, rr)
    x = rr * np.cos(th)
    y = rr * np.sin(th)
    z = np.zeros_like(x)

    # feature line along +X axis
    feat_x = rr[:, 0]
    feat_y = np.zeros_like(feat_x)
    feat_z = np.zeros_like(feat_x)

    fig = plt.figure(figsize=(10, 5))
    ax2d = fig.add_subplot(121)
    ax3d = fig.add_subplot(122, projection="3d")

    ax2d.set_xlabel("Azimuth (deg)")
    ax2d.set_ylabel("Elevation (deg)")
    ax2d.set_title("Target Angles")
    ax2d.grid(True)

    ax2d.set_xlim(np.nanmin(az_target[frames]) - 5, np.nanmax(az_target[frames]) + 5)
    ax2d.set_ylim(np.nanmin(el_target[frames]) - 5, np.nanmax(el_target[frames]) + 5)
    target_dot, = ax2d.plot([], [], "ro")

    # 3D settings
    ax3d.set_xlim(-1.5, 1.5)
    ax3d.set_ylim(-1.5, 1.5)
    ax3d.set_zlim(-1.5, 1.5)
    ax3d.set_title("Disc Orientation")

    surf = [None]
    feature_line, = ax3d.plot([], [], [], "b-", linewidth=2)

    plt.subplots_adjust(bottom=0.25)
    ax_slider = plt.axes([0.2, 0.12, 0.6, 0.03])
    slider = Slider(ax_slider, "Frame", valmin=int(frames[0]), valmax=int(frames[-1]),
                    valinit=int(frames[0]), valstep=1)

    ax_button = plt.axes([0.42, 0.03, 0.16, 0.06])
    btn = Button(ax_button, "Pause")
    state = {"paused": False}

    def update(i):
        i = int(i)
        target_dot.set_data(az_target[i], el_target[i])

        R = Rs[i]  # BODY->ENU

        pts = np.vstack((x.flatten(), y.flatten(), z.flatten()))
        pts_rot = R @ pts
        xr = pts_rot[0].reshape(x.shape)
        yr = pts_rot[1].reshape(y.shape)
        zr = pts_rot[2].reshape(z.shape)

        ax3d.collections.clear()
        ax3d.plot_surface(xr, yr, zr, alpha=0.8)

        feat_pts = np.vstack((feat_x, feat_y, feat_z))
        feat_rot = R @ feat_pts
        feature_line.set_data(feat_rot[0], feat_rot[1])
        feature_line.set_3d_properties(feat_rot[2])

        return target_dot, feature_line

    ani = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)

    def on_slider(val):
        update(int(slider.val))
        fig.canvas.draw_idle()

    slider.on_changed(on_slider)

    def toggle(event):
        if state["paused"]:
            ani.event_source.start()
            btn.label.set_text("Pause")
        else:
            ani.event_source.stop()
            btn.label.set_text("Play")
        state["paused"] = not state["paused"]

    btn.on_clicked(toggle)
    plt.show()


if __name__ == "__main__":
    df = load_output("sim_output.csv")
    Rs = load_Rs_from_csv(df)
    plot_az_el(df)

    # Only works if you added sigma_app_m2 to radar.py output
    if "sigma_true_m2" in df.columns:
        plot_rcs_vs_time(df)

    # Use your actual beamwidths here (from RadarParams)
    plot_beam_vs_target(df, beamwidth_az_deg=20.0, beamwidth_el_deg=10.0)

    animate_disc_orientations(Rs, df)
