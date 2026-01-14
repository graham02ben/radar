# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 08:44:22 2026

@author: bboyg
"""

import numpy as np
from radar_params import RadarParams
from utils_frames import (
    lla_to_ecef,
    ecef_to_enu_matrix,
    enu_to_azelr,
    angle_wrap,
    deg2rad,
    gaussian_beam_gain
)

class Radar:
    """
    Monostatic radar model.

    Frame conventions:
        - Radar position stored in ECEF (meters)
        - LOS vectors computed in ENU (meters)
        - Azimuth measured clockwise from North
        - Elevation positive upward

    Orientation:
        Target orientation provided as BODY -> ENU rotation matrix.
    """

    def __init__(self, params: RadarParams, scan_pattern):
        self.p = params
        self.scan = scan_pattern

        lat, lon, alt_km = self.p.radar_lla

        # Radar position in ECEF meters
        self.radar_ecef_m = np.array(
            lla_to_ecef(lat, lon, alt_km * 1000.0)
        )

        # ECEF -> ENU rotation at radar
        self.ecef_to_enu = ecef_to_enu_matrix(lat, lon)

        # Radar origin in ENU frame
        self.radar_enu_m = np.zeros(3)

        # Wavelength
        self.lam = 3e8 / self.p.f_hz

        # Boltzmann constant
        self.kB = 1.380649e-23

    # ---------------------------------------------------------
    # Beam pattern
    # ---------------------------------------------------------
    def beam_gain_offaxis(self, az_err_rad, el_err_rad):
        gaz = gaussian_beam_gain(
            abs(az_err_rad),
            deg2rad(self.p.bw3dB_az_deg)
        )
        gel = gaussian_beam_gain(
            abs(el_err_rad),
            deg2rad(self.p.bw3dB_el_deg)
        )
        return gaz * gel * self.p.g_lin

    # ---------------------------------------------------------
    # Radar equation (linear SNR)
    # ---------------------------------------------------------
    def snr_linear(self, R, sigma_m2, g_eff, dwell_s):
        num = (
            self.p.p_tx_w *
            (g_eff**2) *
            (self.lam**2) *
            sigma_m2 *
            dwell_s
        )

        den = (
            (4*np.pi)**3 *
            R**4 *
            self.kB *
            self.p.t0_k *
            self.p.bw_hz *
            self.p.l_sys_lin *
            (10**(self.p.nf_dB/10))
        )

        return num / den

    # ---------------------------------------------------------
    # Single dwell measurement
    # ---------------------------------------------------------
    def dwell(self, t, tgt_pos_lla, rcs_model,
              tgt_body_to_enu=None):
        """
        Performs one radar dwell.

        Inputs:
            t               : time [s]
            tgt_pos_lla     : [deg,deg,km]
            rcs_model       : RCSModel instance
            tgt_body_to_enu : 3x3 rotation matrix (optional)

        Returns:
            dict with measurement fields
        """

        # Beam pointing
        az_cmd, el_cmd, dwell_s = self.scan.next_pointing(t)

        # Target ECEF
        xt, yt, zt = lla_to_ecef(
            tgt_pos_lla[0],
            tgt_pos_lla[1],
            tgt_pos_lla[2] * 1000.0
        )
        tgt_ecef = np.array([xt, yt, zt])

        # LOS in ECEF
        los_ecef = tgt_ecef - self.radar_ecef_m

        # LOS in ENU
        los_enu = self.ecef_to_enu @ los_ecef

        R = np.linalg.norm(los_enu)
        if R < 1e-6:
            return None

        u_los_enu = los_enu / R

        # Target angles
        az_t, el_t, _ = enu_to_azelr(los_enu)

        # Beam error
        az_err = angle_wrap(az_t - az_cmd)
        el_err = el_t - el_cmd
        

        # Beam gain
        g_eff = self.beam_gain_offaxis(az_err, el_err)

        # Aspect angles in body frame
        if tgt_body_to_enu is not None:
            u_body = tgt_body_to_enu.T @ u_los_enu

            az_b = np.arctan2(u_body[0], u_body[1])
            el_b = np.arctan2(
                u_body[2],
                np.sqrt(u_body[0]**2 + u_body[1]**2)
            )
        else:
            az_b, el_b = az_t, el_t

        # RCS lookup
        sigma = rcs_model.sigma(az_b, el_b, t)
        sigma_app = sigma * (g_eff / self.p.g_lin)

        # SNR
        snr = self.snr_linear(R, sigma, g_eff, dwell_s)
        snr_dB = 10*np.log10(snr + 1e-30)

        detected = snr_dB >= self.p.snr_min_dB

        return {
            "t": t,
            "range_m": R,
            "az_cmd_rad": az_cmd,
            "el_cmd_rad": el_cmd,
            "az_t_rad": az_t,
            "el_t_rad": el_t,
            "az_err_rad": az_err,
            "el_err_rad": el_err,
            "g_eff_lin": g_eff,
            "sigma_true_m2": sigma,
            "sigma_app_m2": sigma_app,
            "snr_dB": snr_dB,
            "detected": bool(detected)
        }
