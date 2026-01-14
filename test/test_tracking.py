# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 10:11:08 2026

@author: bboyg
"""

import numpy as np
from tracking import TrackManager

def test_tracking_spawns_and_confirms():
    tm = TrackManager(promote_hits=3, drop_misses=10,
                      gate_az_deg=10, gate_el_deg=10, gate_range_m=5000)

    radar_pos = np.zeros(3)

    # Same detection 3 times
    for k in range(3):
        tm.predict_all(0.1)
        tm.associate_and_update(
            t=0.1*k,
            radar_pos_enu=radar_pos,
            dets=[(1000.0, 0.1, 0.05)]
        )

    assert len(tm.tracks) >= 1
    assert any(tr.confirmed for tr in tm.tracks.values())
