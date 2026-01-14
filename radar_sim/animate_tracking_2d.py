# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 17:59:27 2026

@author: bboyg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Ellipse


def animate(csv_path: str, az_min_deg=None, az_max_deg=None, el_min_deg=None, el_max_deg=None,
            beamwidth_az_deg: float = 20.0, beamwidth_el_deg: float = 10.0,
            interval_ms: int = 200):
    """
    2D animation:
      - red dot: target (az_t, el_t)
      - yellow ellipse: beam centered at (az_cmd, el_cmd)
      - beam turns red when detected=True
      - slider to scrub frames
      - pause/play button

    If az_min_deg/az_max_deg/el_min_deg/el_max_deg are provided, axis is fixed to those limits.
    Otherwise uses data bounds.
    """

    df = pd.read_csv(csv_path)

    # Pull required columns
    t = df["t"].to_numpy(dtype=float)
    az_target = np.degrees(df["az_t_rad"].to_numpy(dtype=float))
    el_target = np.degrees(df["el_t_rad"].to_numpy(dtype=float))
    az_beam = np.degrees(df["az_cmd_rad"].to_numpy(dtype=float))
    el_beam = np.degrees(df["el_cmd_rad"].to_numpy(dtype=float))

    det_col = df["detected"].to_numpy()
    if det_col.dtype == object:
        detected = np.array([str(x).lower() == "true" for x in det_col], dtype=bool)
    else:
        detected = det_col.astype(bool)

    range_m = df["range_m"].to_numpy(dtype=float) if "range_m" in df.columns else np.full_like(t, np.nan)

    # Valid frames: finite target + beam angles
    valid_mask = np.isfinite(az_target) & np.isfinite(el_target) & np.isfinite(az_beam) & np.isfinite(el_beam)
    valid_frames = np.where(valid_mask)[0]
    if len(valid_frames) == 0:
        raise ValueError("No valid frames found (angles are NaN).")

    # ---- Figure ----
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title("Radar Beam vs Target Position")
    ax.grid(True)

    # Axis limits
    if az_min_deg is not None and az_max_deg is not None:
        ax.set_xlim(az_min_deg, az_max_deg)
    else:
        ax.set_xlim(np.nanmin(az_target[valid_frames]) - 5, np.nanmax(az_target[valid_frames]) + 5)

    if el_min_deg is not None and el_max_deg is not None:
        ax.set_ylim(el_min_deg, el_max_deg)
    else:
        ax.set_ylim(np.nanmin(el_target[valid_frames]) - 5, np.nanmax(el_target[valid_frames]) + 5)

    # Target dot
    target_dot, = ax.plot([], [], "ro", label="Target")

    # Beam ellipse (instead of Circle, since az/el beamwidth differ)
    beam_patch = Ellipse((0, 0), width=beamwidth_az_deg, height=beamwidth_el_deg,
                         facecolor="yellow", alpha=0.3, edgecolor="none", label="Radar Beam")
    ax.add_patch(beam_patch)
    ax.legend()

    # Text boxes (match your old style)
    detection_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=dict(facecolor="white", alpha=0.6)
    )
    time_text = ax.text(
        0.02, 0.85, "", transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=dict(facecolor="white", alpha=0.6)
    )
    distance_text = ax.text(
        0.02, 0.75, "", transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=dict(facecolor="white", alpha=0.6)
    )

    plt.subplots_adjust(bottom=0.28)

    # Slider
    ax_slider = plt.axes([0.2, 0.12, 0.6, 0.03])
    frame_slider = Slider(
        ax_slider, "Frame",
        valmin=int(valid_frames[0]),
        valmax=int(valid_frames[-1]),
        valinit=int(valid_frames[0]),
        valstep=1
    )

    # Pause button
    ax_button = plt.axes([0.4, 0.03, 0.2, 0.06])
    pause_button = Button(ax_button, "Pause", color="lightgray", hovercolor="gray")

    def update(frame):
        frame = int(frame)

        az_t = az_target[frame]
        el_t = el_target[frame]
        az_b = az_beam[frame]
        el_b = el_beam[frame]

        # Update target
        target_dot.set_data([az_t], [el_t])

        # Update beam position + size
        beam_patch.center = (az_b, el_b)
        beam_patch.width = beamwidth_az_deg
        beam_patch.height = beamwidth_el_deg

        # Color on detection
        if detected[frame]:
            beam_patch.set_facecolor("red")
        else:
            beam_patch.set_facecolor("yellow")

        # Detections count up to this frame
        num_det = int(np.sum(detected[:frame + 1]))
        detection_text.set_text(f"Detections: {num_det}")

        # Time
        time_text.set_text(f"Current Time: {t[frame]:.2f} s")

        # Range
        if np.isfinite(range_m[frame]):
            distance_text.set_text(f"Distance: {range_m[frame] / 1000.0:.2f} km")
        else:
            distance_text.set_text("Distance: N/A")

        return target_dot, beam_patch, detection_text, time_text, distance_text

    ani = FuncAnimation(fig, update, frames=valid_frames, interval=interval_ms, blit=False)

    def slider_update(_val):
        frame = int(frame_slider.val)
        update(frame)
        fig.canvas.draw_idle()

    frame_slider.on_changed(slider_update)

    state = {"paused": False}

    def toggle_pause(_event):
        if state["paused"]:
            ani.event_source.start()
            pause_button.label.set_text("Pause")
        else:
            ani.event_source.stop()
            pause_button.label.set_text("Play")
        state["paused"] = not state["paused"]

    pause_button.on_clicked(toggle_pause)

    # Initialize
    update(valid_frames[0])
    plt.show()


if __name__ == "__main__":
    # For your main sim:
    #   beamwidth_az_deg / beamwidth_el_deg should match RadarParams
    animate(
        csv_path="sim_output.csv",
        beamwidth_az_deg=20.0,
        beamwidth_el_deg=10.0,
        interval_ms=200
    )

    # If you want the flip file instead, comment above and use:
    # animate(csv_path="sim_output_flip.csv", beamwidth_az_deg=30.0, beamwidth_el_deg=30.0, interval_ms=200)
