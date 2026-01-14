# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 16:16:40 2026

@author: bboyg
"""

import numpy as np

class SimpleFlipRCS:
    """
    Analytic RCS for a flipping/rotating object.

    We assume the object's "front normal" is BODY +Y.
    The radar provides az/el in BODY frame (rad), which we convert to a LOS unit vector in BODY.

    RCS logic:
      - front: LOS aligned with +Y  -> sigma_front
      - back:  LOS aligned with -Y  -> sigma_back
      - edge:  LOS perpendicular to Y (dot ~ 0) -> sigma_edge
    Smoothly interpolates between these based on dot product with +Y.

    Returns sigma in m^2 (linear).
    """

    def __init__(self, sigma_front_m2=100.0, sigma_back_m2=30.0, sigma_edge_m2=8.0):
        self.sigma_front = float(sigma_front_m2)
        self.sigma_back = float(sigma_back_m2)
        self.sigma_edge = float(sigma_edge_m2)

    @staticmethod
    def _u_from_az_el(az: float, el: float) -> np.ndarray:
        """
        az/el convention matches the one used in radar.py when computing az_b/el_b:
          az = atan2(x, y)
          el = atan2(z, sqrt(x^2+y^2))

        Reconstruct unit vector in that same coordinate system:
          x = cos(el)*sin(az)
          y = cos(el)*cos(az)
          z = sin(el)
        """
        ce = np.cos(el)
        return np.array([ce * np.sin(az), ce * np.cos(az), np.sin(el)], dtype=float)

    def sigma(self, az_b_rad: float, el_b_rad: float, t: float = 0.0) -> float:
        u = self._u_from_az_el(az_b_rad, el_b_rad)

        # front normal = +Y
        dot = float(u[1])  # u · +Y  (since +Y = [0,1,0])

        # At dot = +1 => front
        # At dot =  0 => edge
        # At dot = -1 => back
        if dot >= 0:
            # interpolate edge -> front
            sigma = self.sigma_edge + dot * (self.sigma_front - self.sigma_edge)
        else:
            # interpolate edge -> back (dot negative)
            sigma = self.sigma_edge + (-dot) * (self.sigma_back - self.sigma_edge)

        return max(sigma, 0.0)
