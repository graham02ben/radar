# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 08:45:32 2026

@author: bboyg
"""

from dataclasses import dataclass
import numpy as np

from scan_patterns import FixedPointingScan
from utils_frames import angle_wrap


@dataclass
class RadarFace:
    """
    A radar face defines a field-of-view (FOV) and a scan pattern for that face.

    All angles are radians.
      - az in [-pi, +pi) convention (wrapped)
      - el typically in [-pi/2, +pi/2], but you can define any band

    max_range_m in meters.
    """
    radar: object                  # expects Radar-like interface with .dwell(...)
    scan_pattern: object           # expects ScanPattern interface

    fov_az_min_rad: float
    fov_az_max_rad: float
    fov_el_min_rad: float
    fov_el_max_rad: float
    max_range_m: float

    def is_within_fov(self, az_t_rad: float, el_t_rad: float, range_m: float) -> bool:
        if range_m > self.max_range_m:
            return False

        # elevation gate
        if not (self.fov_el_min_rad <= el_t_rad <= self.fov_el_max_rad):
            return False

        # --- azimuth gate with wrap-safe full-circle handling ---
        az = angle_wrap(az_t_rad)
        amin = angle_wrap(self.fov_az_min_rad)
        amax = angle_wrap(self.fov_az_max_rad)

        # If the window spans ~360 degrees, accept all az
        span = (amax - amin) % (2.0 * np.pi)
        if span > (2.0 * np.pi - 1e-6):
            return True

        # Normal wrapped-interval check
        if amin <= amax:
            return amin <= az <= amax
        else:
            # wraps through -pi/pi boundary
            return (az >= amin) or (az <= amax)


    def dwell(self, t: float, tgt_pos_lla, rcs_model, tgt_body_to_enu=None, azel_override=None):
        """
        Performs a dwell through this face.

        If azel_override is provided, it must be (az_rad, el_rad),
        and the dwell time is taken from the face scan pattern at time t.

        Returns a measurement dict with at least:
            - detected (bool)
            - range_m (float)
            - az_t_rad, el_t_rad (float)
            - az_cmd_rad, el_cmd_rad (float)
            - snr_dB (float)
        """

        # Choose commanded pointing
        if azel_override is None:
            az_cmd, el_cmd, dwell_s = self.scan_pattern.next_pointing(t)
        else:
            az_cmd, el_cmd = azel_override
            # keep dwell from scan pattern (so face governs dwell behavior)
            _, _, dwell_s = self.scan_pattern.next_pointing(t)

        # Temporarily override radar scan for this dwell only
        prev_scan = self.radar.scan
        self.radar.scan = FixedPointingScan(az_cmd, el_cmd, dwell_s)

        meas = self.radar.dwell(
            t=t,
            tgt_pos_lla=tgt_pos_lla,
            rcs_model=rcs_model,
            tgt_body_to_enu=tgt_body_to_enu
        )

        # Restore original scan no matter what
        self.radar.scan = prev_scan

        if meas is None:
            return {
                "t": t,
                "detected": False,
                "snr_dB": -999.0,
                "range_m": np.nan,
                "az_cmd_rad": az_cmd,
                "el_cmd_rad": el_cmd,
                "az_t_rad": np.nan,
                "el_t_rad": np.nan,
            }

        # Apply FOV gate
        if not self.is_within_fov(meas["az_t_rad"], meas["el_t_rad"], meas["range_m"]):
            meas = meas.copy()
            meas["detected"] = False
            meas["sigma_app_m2"] = np.nan  # optional for compatibility
            meas["snr_dB"] = meas.get("snr_dB", -999.0)
            return meas

        return meas
