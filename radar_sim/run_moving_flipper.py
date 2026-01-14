# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:06:55 2026

@author: bboyg
"""

from scenarios import run_moving_flipper_scenario
from animate_tracking_2d import animate

def main():
    df = run_moving_flipper_scenario(t_end_s=20.0, dt_s=1.0)
    csv_path = "sim_output_moving_flip.csv"
    
    df.to_csv(csv_path, index=False)
    print("Saved sim_output_moving_flip.csv")
    print("Detections:", int(df["detected"].sum()))

    # Beamwidths should match RadarParams in the scenario
    beamwidth_az_deg = 5.0
    beamwidth_el_deg = 3.0

    print("Launching animation...")
    animate(
        csv_path=csv_path,
        beamwidth_az_deg=beamwidth_az_deg,
        beamwidth_el_deg=beamwidth_el_deg,
        interval_ms=200
    )
    
if __name__ == "__main__":
    main()
