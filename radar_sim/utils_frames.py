# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 07:48:22 2026

@author: bboyg
"""

import numpy as np


# ============================================================
# Angle utilities
# ============================================================

def deg2rad(deg: float) -> float:
    """ 
    Converts angle from degrees to radians.

    """
    return deg * np.pi / 180.0


def rad2deg(rad: float) -> float:
    """
    Converts angle from radians to degrees.

    """
    return rad * 180.0 / np.pi


def angle_wrap(ang: float) -> float:
    """
    Wrap angle to [-pi, +pi].
    """
    return (ang + np.pi) % (2.0 * np.pi) - np.pi


def angle_wrap_2pi(ang: float) -> float:
    """
    Wrap angle to [0, 2pi].
    """
    return (ang + 2.0 * np.pi) % (2.0 * np.pi)


# ============================================================
# Radar beam / antenna pattern helper
# ============================================================

def gaussian_beam_gain(off_boresight_rad: float, bw3dB_rad: float) -> float:
    """
    Gaussian approximation of mainlobe gain roll-off.

    bw3dB_rad is FULL 3dB beamwidth (FWHM) in radians.
    """
    if bw3dB_rad <= 0:
        return 0.0
    return float(np.exp(-4.0 * np.log(2.0) * (off_boresight_rad / bw3dB_rad) ** 2))


# ============================================================
# WGS84: LLA <-> ECEF
# ============================================================

def lla_to_ecef(lat_deg: float, lon_deg: float, alt_m: float):
    """
    Convert geodetic latitude/longitude/altitude to ECEF.

    Inputs:
        lat_deg, lon_deg : degrees
        alt_m            : meters

    Returns:
        (x, y, z) in meters (ECEF)
    """
    # WGS84 constants
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)

    sLat = np.sin(lat)
    cLat = np.cos(lat)
    sLon = np.sin(lon)
    cLon = np.cos(lon)

    N = a / np.sqrt(1.0 - e2 * (sLat ** 2))

    x = (N + alt_m) * cLat * cLon
    y = (N + alt_m) * cLat * sLon
    z = (N * (1.0 - e2) + alt_m) * sLat

    return float(x), float(y), float(z)


def ecef_to_enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    """
    Rotation matrix that maps ECEF vectors to local ENU at (lat, lon).
    ENU axes: x=east, y=north, z=up.

    Returns:
        R (3x3) such that: v_enu = R @ v_ecef
    """
    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)

    sLat = np.sin(lat)
    cLat = np.cos(lat)
    sLon = np.sin(lon)
    cLon = np.cos(lon)

    R = np.array([
        [-sLon,          cLon,         0.0],
        [-sLat * cLon,  -sLat * sLon,  cLat],
        [ cLat * cLon,   cLat * sLon,  sLat]
    ], dtype=float)

    return R


def lla_to_enu_vector_m(radar_lla_deg_km, target_lla_deg_km) -> np.ndarray:
    """
    Convenience: returns ENU vector (meters) from radar -> target.

    Inputs:
        radar_lla_deg_km  : (lat_deg, lon_deg, alt_km)
        target_lla_deg_km : (lat_deg, lon_deg, alt_km)
    """
    lat_r, lon_r, alt_r_km = radar_lla_deg_km
    lat_t, lon_t, alt_t_km = target_lla_deg_km

    xr, yr, zr = lla_to_ecef(lat_r, lon_r, alt_r_km * 1000.0)
    xt, yt, zt = lla_to_ecef(lat_t, lon_t, alt_t_km * 1000.0)

    d_ecef = np.array([xt - xr, yt - yr, zt - zr], dtype=float)

    R = ecef_to_enu_matrix(lat_r, lon_r)
    return R @ d_ecef


# ============================================================
# ENU vector -> (az, el, range)
# ============================================================

def enu_to_azelr(v_enu: np.ndarray):
    """
    Convert ENU vector to azimuth, elevation, range.

    Inputs:
        v_enu : [east, north, up] in meters

    Returns:
        az_rad : [0, 2pi)
        el_rad : [-pi/2, +pi/2]
        r_m    : range in meters
    """
    east, north, up = float(v_enu[0]), float(v_enu[1]), float(v_enu[2])
    r = float(np.linalg.norm(v_enu))

    if r < 1e-12:
        return 0.0, 0.0, 0.0

    az = np.arctan2(east, north)
    if az < 0.0:
        az += 2.0 * np.pi

    el = np.arctan2(up, np.sqrt(east * east + north * north))

    return float(az), float(el), r
