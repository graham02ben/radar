# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 08:32:50 2026

@author: bboyg
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class Trajectory:
    """
    Trajectory container for a rigid body target.

    t    : time array [s] shape (N,)
    pos  : LLA position [deg, deg, km] shape (N,3)
    roll : body angular rates [rad/s] (p,q,r) shape (N,3)

    Orientation convention:
        R[i] maps BODY vectors -> ENU frame at time t[i]
    """
    t: np.ndarray
    pos: np.ndarray
    roll: np.ndarray

    # ---------------------------------------------------------
    # Linear interpolation of trajectory
    # ---------------------------------------------------------
    def interp(self, tq: float):
        """
        Interpolate trajectory at time tq.

        Returns:
            pos_interp  : LLA [deg,deg,km]
            roll_interp : body rates [rad/s]
        """
        t = self.t

        if tq <= t[0]:
            return self.pos[0], self.roll[0]

        if tq >= t[-1]:
            return self.pos[-1], self.roll[-1]

        i = np.searchsorted(t, tq) - 1
        i = np.clip(i, 0, len(t)-2)

        w = (tq - t[i]) / (t[i+1] - t[i])

        pos_i = self.pos[i] + w * (self.pos[i+1] - self.pos[i])
        roll_i = self.roll[i] + w * (self.roll[i+1] - self.roll[i])

        return pos_i, roll_i

    # ---------------------------------------------------------
    # Orientation integration
    # ---------------------------------------------------------
    def integrate_orientation(self, R0_body_to_enu=None):
        """
        Integrates body angular rates into orientation matrices.

        Returns:
            Rs : list of rotation matrices, BODY -> ENU
        """

        N = len(self.t)

        if R0_body_to_enu is None:
            R = np.eye(3)
        else:
            R = R0_body_to_enu.copy()

        Rs = [R.copy()]

        for i in range(N-1):

            dt = self.t[i+1] - self.t[i]
            p, q, r = self.roll[i]

            omega = np.array([p, q, r])
            omega_mag = np.linalg.norm(omega)

            if omega_mag < 1e-12:
                dR = np.eye(3)
            else:
                u = omega / omega_mag
                theta = omega_mag * dt

                ux = np.array([
                    [ 0,     -u[2],  u[1]],
                    [ u[2],   0,    -u[0]],
                    [-u[1],  u[0],   0   ]
                ])

                dR = (
                    np.eye(3)
                    + np.sin(theta) * ux
                    + (1 - np.cos(theta)) * (ux @ ux)
                )

            # IMPORTANT: pre-multiply (body frame update)
            R = dR @ R

            Rs.append(R.copy())

        return Rs