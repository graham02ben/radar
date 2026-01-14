# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 08:45:33 2026

@author: bboyg
"""

from dataclasses import dataclass
import numpy as np
from utils_frames import deg2rad, angle_wrap


class ScanPattern:
    """
    Base class for scan patterns.

    next_pointing(t) -> (az_rad, el_rad, dwell_s)
    """
    def next_pointing(self, t: float):
        raise NotImplementedError


# ------------------------------------------------------------
# Fixed pointing (stare)
# ------------------------------------------------------------
@dataclass
class FixedPointingScan(ScanPattern):
    az_rad: float
    el_rad: float
    dwell_s: float

    @staticmethod
    def from_degrees(az_deg: float, el_deg: float, dwell_s: float):
        return FixedPointingScan(deg2rad(az_deg), deg2rad(el_deg), dwell_s)

    def next_pointing(self, t: float):
        return self.az_rad, self.el_rad, self.dwell_s


# ------------------------------------------------------------
# Circular scan: azimuth spins at constant rate, fixed elevation
# ------------------------------------------------------------
@dataclass
class CircularScan(ScanPattern):
    """
    spin_rate_rpm : rotations per minute
    el_rad        : elevation (rad)
    dwell_s       : dwell per beam position (s)
    az0_rad       : starting azimuth (rad)
    """
    spin_rate_rpm: float
    el_rad: float
    dwell_s: float
    az0_rad: float = 0.0

    @staticmethod
    def from_degrees(spin_rate_rpm: float, el_deg: float, dwell_s: float, az0_deg: float = 0.0):
        return CircularScan(spin_rate_rpm, deg2rad(el_deg), dwell_s, deg2rad(az0_deg))

    def next_pointing(self, t: float):
        # rpm -> rad/s: 2π rad per rev, rpm/60 rev/s
        omega = 2.0 * np.pi * (self.spin_rate_rpm / 60.0)
        az = angle_wrap(self.az0_rad + omega * t)
        return az, self.el_rad, self.dwell_s


# ------------------------------------------------------------
# Sector scan: triangle-wave back and forth in az, fixed elevation
# ------------------------------------------------------------
@dataclass
class SectorScan(ScanPattern):
    """
    az_center_rad : sector center azimuth (rad)
    az_width_rad  : total sector width (rad)
    el_rad        : elevation (rad)
    sweep_rate_rads : sweep speed (rad/s) (speed along az axis)
    dwell_s       : dwell (s)
    """
    az_center_rad: float
    az_width_rad: float
    el_rad: float
    sweep_rate_rads: float
    dwell_s: float

    @staticmethod
    def from_degrees(az_center_deg: float, az_width_deg: float, el_deg: float,
                     sweep_rate_dps: float, dwell_s: float):
        return SectorScan(
            deg2rad(az_center_deg),
            deg2rad(az_width_deg),
            deg2rad(el_deg),
            deg2rad(sweep_rate_dps),
            dwell_s
        )

    def next_pointing(self, t: float):
        half = self.az_width_rad / 2.0
        rate = max(self.sweep_rate_rads, 1e-12)  # avoid divide-by-zero

        # time to go from -half to +half is (2*half)/rate
        # full back-and-forth period is twice that
        one_way = (2.0 * half) / rate
        period = 2.0 * one_way

        tau = t % period

        if tau <= one_way:
            # forward sweep: -half -> +half
            az = (self.az_center_rad - half) + rate * tau
        else:
            # backward sweep: +half -> -half
            az = (self.az_center_rad + half) - rate * (tau - one_way)

        az = angle_wrap(az)
        return az, self.el_rad, self.dwell_s


# ------------------------------------------------------------
# Raster scan: sweep azimuth linearly, step elevation each line (boustrophedon)
# ------------------------------------------------------------
@dataclass
class RasterScan(ScanPattern):
    """
    Raster scan in az/el.

    az_min_rad, az_max_rad : sweep bounds (rad)
    el_min_rad, el_max_rad : elevation bounds (rad)
    az_rate_rads           : azimuth sweep rate (rad/s)
    el_step_rad            : elevation step per line (rad)
    dwell_s                : dwell (s)

    Behavior:
      - Each line sweeps az from min->max or max->min (alternating)
      - Then elevation increases by el_step
      - After reaching el_max, repeats from el_min
    """
    az_min_rad: float
    az_max_rad: float
    el_min_rad: float
    el_max_rad: float
    az_rate_rads: float
    el_step_rad: float
    dwell_s: float

    @staticmethod
    def from_degrees(az_min_deg: float, az_max_deg: float,
                     el_min_deg: float, el_max_deg: float,
                     az_rate_dps: float, el_step_deg: float,
                     dwell_s: float):
        return RasterScan(
            deg2rad(az_min_deg),
            deg2rad(az_max_deg),
            deg2rad(el_min_deg),
            deg2rad(el_max_deg),
            deg2rad(az_rate_dps),
            deg2rad(el_step_deg),
            dwell_s
        )

    def next_pointing(self, t: float):
        az_span = self.az_max_rad - self.az_min_rad
        rate = max(abs(self.az_rate_rads), 1e-12)

        line_time = abs(az_span) / rate

        # number of elevation lines
        el_span = self.el_max_rad - self.el_min_rad
        n_lines = int(np.floor(abs(el_span) / max(self.el_step_rad, 1e-12))) + 1
        n_lines = max(n_lines, 1)

        frame_time = n_lines * line_time
        tau = t % frame_time

        line_idx = int(tau // line_time)
        line_tau = tau - line_idx * line_time

        # elevation for this line
        el = self.el_min_rad + line_idx * self.el_step_rad

        # clamp to bounds in case of rounding
        el = np.clip(el, min(self.el_min_rad, self.el_max_rad), max(self.el_min_rad, self.el_max_rad))

        # alternate direction each line (boustrophedon)
        if line_idx % 2 == 0:
            az = self.az_min_rad + np.sign(az_span) * rate * line_tau
        else:
            az = self.az_max_rad - np.sign(az_span) * rate * line_tau

        az = angle_wrap(az)
        return az, el, self.dwell_s
