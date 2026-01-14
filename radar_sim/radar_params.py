# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 08:38:11 2026

@author: bboyg
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class RadarParams:
    """
    Radar parameter container.

    radar_lla : [deg, deg, km] radar position
    f_hz      : carrier frequency [Hz]
    p_tx_w    : transmit power [W]
    g_lin     : antenna boresight gain (linear)
    l_sys_lin : system losses (linear)
    nf_dB     : noise figure [dB]
    bw_hz     : receiver noise bandwidth [Hz]
    t0_k      : system temperature [K]

    bw3dB_az_deg : azimuth 3 dB beamwidth [deg]
    bw3dB_el_deg : elevation 3 dB beamwidth [deg]
    snr_min_dB   : detection threshold [dB]
    """

    radar_lla: np.ndarray

    f_hz: float
    p_tx_w: float
    g_lin: float
    l_sys_lin: float
    nf_dB: float
    bw_hz: float
    t0_k: float

    bw3dB_az_deg: float
    bw3dB_el_deg: float
    snr_min_dB: float
