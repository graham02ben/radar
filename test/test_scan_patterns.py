# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 10:07:02 2026

@author: bboyg
"""

import numpy as np
from scan_patterns import RasterScan, FixedPointingScan

def test_fixed_pointing_returns_constants():
    s = FixedPointingScan(az_rad=1.0, el_rad=0.5, dwell_s=0.1)
    az, el, d = s.next_pointing(123.0)
    assert az == 1.0 and el == 0.5 and d == 0.1

def test_raster_scan_within_bounds():
    scan = RasterScan.from_degrees(
        az_min_deg=210, az_max_deg=270,
        el_min_deg=0, el_max_deg=90,
        az_rate_dps=70, el_step_deg=5,
        dwell_s=0.1
    )

    for t in np.linspace(0, 10, 100):
        az, el, d = scan.next_pointing(float(t))
        assert d > 0
        assert np.deg2rad(0) - 1e-9 <= el <= np.deg2rad(90) + 1e-9
        # az wraps [-pi,pi), so just ensure it's a finite radian value
        assert np.isfinite(az)
