# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 10:20:14 2026

@author: bboyg
"""

import sys
from pathlib import Path

# Add the project root folder (the parent of /test) to the import path
ROOT = Path(__file__).resolve().parents[1] / "radar_sim"
sys.path.insert(0, str(ROOT))