# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 13:02:09 2026

@author: bboyg
"""

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

def test_saved_rotation_is_orthonormal():
    # Create a fake CSV with valid rotation matrices
    rows = []
    for _ in range(5):
        R = np.eye(3)
        rows.append({
            "R00": R[0,0], "R01": R[0,1], "R02": R[0,2],
            "R10": R[1,0], "R11": R[1,1], "R12": R[1,2],
            "R20": R[2,0], "R21": R[2,1], "R22": R[2,2],
        })

    df = pd.DataFrame(rows)

    # Save to a temporary file
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sim_output.csv"
        df.to_csv(path, index=False)

        # Reload and test
        df2 = pd.read_csv(path)
        for _, row in df2.iterrows():
            R = np.array([
                [row["R00"], row["R01"], row["R02"]],
                [row["R10"], row["R11"], row["R12"]],
                [row["R20"], row["R21"], row["R22"]],
            ], float)

            err = np.linalg.norm(R.T @ R - np.eye(3))
            assert err < 1e-12