# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 08:46:31 2026

@author: bboyg
"""

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from utils_frames import rad2deg, angle_wrap_2pi


@dataclass
class RCSGrid:
    """
    RCS lookup table on a spherical grid.

    phi_deg   : azimuth grid in degrees, expected increasing, typically [0..360)
    theta_deg : polar angle grid in degrees, expected increasing, [0..180]
                where theta=0 is +Z axis (north pole), theta=90 is in XY plane.

    sigma_m2  : 2D array shape (len(phi_deg), len(theta_deg)) in m² (linear)

    Notes:
      - We store sigma in linear m² to avoid log issues during interpolation.
      - Use `from_sigma_dBsm(...)` helper if your data is in dBsm.
    """
    phi_deg: np.ndarray
    theta_deg: np.ndarray
    sigma_m2: np.ndarray

    def __post_init__(self):
        self.phi_deg = np.asarray(self.phi_deg, dtype=float)
        self.theta_deg = np.asarray(self.theta_deg, dtype=float)
        self.sigma_m2 = np.asarray(self.sigma_m2, dtype=float)

        if self.sigma_m2.ndim != 2:
            raise ValueError("sigma_m2 must be a 2D array.")

        expected = (len(self.phi_deg), len(self.theta_deg))
        if self.sigma_m2.shape != expected:
            raise ValueError(
                f"sigma_m2 shape {self.sigma_m2.shape} does not match "
                f"(len(phi_deg), len(theta_deg)) = {expected}."
            )

        # Interpolator: expects points as (phi, theta)
        self._interp = RegularGridInterpolator(
            (self.phi_deg, self.theta_deg),
            self.sigma_m2,
            bounds_error=False,
            fill_value=None  # allow extrapolation if needed; we clamp anyway
        )

    @staticmethod
    def from_sigma_dBsm(phi_deg, theta_deg, sigma_dBsm):
        """
        Create an RCSGrid from dBsm values.
        """
        sigma_dBsm = np.asarray(sigma_dBsm, dtype=float)
        sigma_m2 = 10.0 ** (sigma_dBsm / 10.0)
        return RCSGrid(np.asarray(phi_deg, float), np.asarray(theta_deg, float), sigma_m2)


class RCSModel:
    """
    RCS model that maps body-frame LOS angles -> sigma (m²).

    Input angles (az_b, el_b) are in radians:
        az_b: azimuth in body frame (x east/right, y forward, z up convention depends on you)
        el_b: elevation in body frame

    Internal conversion:
        - phi (deg)   = azimuth mapped into [0,360)
        - theta (deg) = polar angle in [0,180] where theta = 90 - elevation_deg
    """

    def __init__(self, grid: RCSGrid, clamp_theta=True):
        self.grid = grid
        self.clamp_theta = clamp_theta

    def sigma(self, az_b_rad: float, el_b_rad: float, t: float = 0.0) -> float:
        """
        Returns sigma in m² (linear).
        t is unused here but kept for interface compatibility (you may add time-varying RCS later).
        """

        az = angle_wrap_2pi(az_b_rad)
        el = el_b_rad

        phi = (rad2deg(az))  # [0..360)
        theta = 90.0 - rad2deg(el)  # elevation -> polar

        if self.clamp_theta:
            theta = float(np.clip(theta, 0.0, 180.0))

        val = self.grid._interp((phi, theta))

        # RegularGridInterpolator returns ndarray scalar sometimes
        val = float(val) if np.isscalar(val) or np.size(val) == 1 else float(val.item())

        # Safety: sigma cannot be negative
        return max(val, 0.0)
