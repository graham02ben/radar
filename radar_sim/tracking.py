# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 08:56:48 2026

@author: bboyg
"""

from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np

from utils_frames import angle_wrap


# ============================================================
# Constant-Velocity Kalman Filter in ENU
# State: [x, y, z, vx, vy, vz] in meters / m/s
# Measurement: [range, az, el] (m, rad, rad)
# ============================================================

class CVKalman:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, q_acc: float):
        """
        q_acc: acceleration PSD-ish tuning (m^2/s^3 scale).
        """
        self.x = x0.astype(float).copy()
        self.P = P0.astype(float).copy()
        self.q = float(q_acc)

    def F(self, dt: float) -> np.ndarray:
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def Q(self, dt: float) -> np.ndarray:
        # White-acceleration model
        q = self.q
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        Q = np.zeros((6, 6))
        # position-position
        Q[0, 0] = Q[1, 1] = Q[2, 2] = q * dt4 / 4.0
        # position-velocity
        Q[0, 3] = Q[1, 4] = Q[2, 5] = q * dt3 / 2.0
        Q[3, 0] = Q[4, 1] = Q[5, 2] = q * dt3 / 2.0
        # velocity-velocity
        Q[3, 3] = Q[4, 4] = Q[5, 5] = q * dt2
        return Q

    # ---- measurement model ----
    @staticmethod
    def h(x: np.ndarray, radar_pos_enu: np.ndarray) -> np.ndarray:
        los = x[:3] - radar_pos_enu
        R = np.linalg.norm(los)
        if R < 1e-9:
            return np.array([0.0, 0.0, 0.0])
        east, north, up = los[0], los[1], los[2]
        az = np.arctan2(east, north)
        if az < 0:
            az += 2.0 * np.pi
        el = np.arctan2(up, np.sqrt(east**2 + north**2))
        return np.array([R, az, el])

    @staticmethod
    def H_numeric(x: np.ndarray, radar_pos_enu: np.ndarray) -> np.ndarray:
        """
        Numeric Jacobian with scale-aware epsilon.
        """
        z0 = CVKalman.h(x, radar_pos_enu)
        H = np.zeros((3, 6))

        # scale-aware eps per state component
        eps_vec = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])  # meters and m/s perturbations

        for i in range(6):
            dx = np.zeros(6)
            dx[i] = eps_vec[i]
            z1 = CVKalman.h(x + dx, radar_pos_enu)
            dz = z1 - z0
            dz[1] = angle_wrap(dz[1])  # az wrap
            H[:, i] = dz / eps_vec[i]

        return H

    def predict(self, dt: float):
        F = self.F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q(dt)

    def update(self, z: np.ndarray, Rmeas: np.ndarray, radar_pos_enu: np.ndarray):
        zhat = self.h(self.x, radar_pos_enu)
        H = self.H_numeric(self.x, radar_pos_enu)

        y = z - zhat
        y[1] = angle_wrap(y[1])  # wrap az residual
        # el residual typically small enough; keep raw

        S = H @ self.P @ H.T + Rmeas
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P


# ============================================================
# Track object
# ============================================================

@dataclass
class Track:
    id: int
    kf: CVKalman
    hits: int = 0
    misses: int = 0
    confirmed: bool = False
    last_update_t: float = 0.0


# ============================================================
# Track manager
# ============================================================

class TrackManager:
    def __init__(
        self,
        gate_az_deg: float = 5.0,
        gate_el_deg: float = 5.0,
        gate_range_m: float = 500.0,
        promote_hits: int = 3,
        drop_misses: int = 20,
    ):
        self.gate_az = np.deg2rad(gate_az_deg)
        self.gate_el = np.deg2rad(gate_el_deg)
        self.gate_R = float(gate_range_m)

        self.promote_hits = int(promote_hits)
        self.drop_misses = int(drop_misses)

        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._rr_queue = deque()

    def predict_all(self, dt: float):
        for tr in self.tracks.values():
            tr.kf.predict(dt)

    @staticmethod
    def _init_state_from_meas(R: float, az: float, el: float, radar_pos_enu: np.ndarray) -> np.ndarray:
        east = R * np.cos(el) * np.sin(az)
        north = R * np.cos(el) * np.cos(az)
        up = R * np.sin(el)
        pos = radar_pos_enu + np.array([east, north, up])
        return np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0])

    def _meas_from_track(self, tr: Track, radar_pos_enu: np.ndarray) -> np.ndarray:
        return CVKalman.h(tr.kf.x, radar_pos_enu)

    def _gate_ok(self, z: np.ndarray, zhat: np.ndarray) -> bool:
        dR = abs(z[0] - zhat[0])
        daz = abs(angle_wrap(z[1] - zhat[1]))
        delv = abs(z[2] - zhat[2])
        return (dR <= self.gate_R) and (daz <= self.gate_az) and (delv <= self.gate_el)

    def associate_and_update(
        self,
        t: float,
        radar_pos_enu: np.ndarray,
        dets: List[Tuple[float, float, float]],
        Rmeas: Optional[np.ndarray] = None,
    ):
        """
        dets: list of (range_m, az_rad, el_rad)
        Rmeas: 3x3 measurement covariance for (R, az, el)
        """
        if Rmeas is None:
            # sensible defaults; tune later
            Rmeas = np.diag([25.0**2, np.deg2rad(0.2)**2, np.deg2rad(0.2)**2])

        used_det = set()

        # ---- update existing tracks (nearest-neighbor) ----
        for tid, tr in list(self.tracks.items()):

            if len(dets) == 0:
                tr.misses += 1
                if tr.misses >= self.drop_misses:
                    self._remove_track(tid)
                continue

            zhat = self._meas_from_track(tr, radar_pos_enu)

            best_i = None
            best_score = 1e9

            for i, (R, az, el) in enumerate(dets):
                if i in used_det:
                    continue
                z = np.array([R, az, el])

                if not self._gate_ok(z, zhat):
                    continue

                # score = weighted residual magnitude
                dR = abs(z[0] - zhat[0]) / max(self.gate_R, 1e-6)
                daz = abs(angle_wrap(z[1] - zhat[1])) / max(self.gate_az, 1e-6)
                delv = abs(z[2] - zhat[2]) / max(self.gate_el, 1e-6)
                score = dR + daz + delv

                if score < best_score:
                    best_score = score
                    best_i = i

            if best_i is None:
                tr.misses += 1
                if tr.misses >= self.drop_misses:
                    self._remove_track(tid)
                continue

            # update with chosen detection
            R, az, el = dets[best_i]
            used_det.add(best_i)

            z = np.array([R, az, el])
            tr.kf.update(z, Rmeas, radar_pos_enu)

            tr.hits += 1
            tr.misses = 0
            tr.last_update_t = t

            if (not tr.confirmed) and (tr.hits >= self.promote_hits):
                tr.confirmed = True
                if tid not in self._rr_queue:
                    self._rr_queue.append(tid)

        # ---- spawn new tracks for unused detections ----
        for i, (R, az, el) in enumerate(dets):
            if i in used_det:
                continue

            x0 = self._init_state_from_meas(R, az, el, radar_pos_enu)
            P0 = np.diag([500.0**2, 500.0**2, 200.0**2, 50.0**2, 50.0**2, 20.0**2])

            kf = CVKalman(x0=x0, P0=P0, q_acc=5.0)

            tid = self._next_id
            self._next_id += 1

            self.tracks[tid] = Track(
                id=tid,
                kf=kf,
                hits=1,
                misses=0,
                confirmed=False,
                last_update_t=t
            )

    def _remove_track(self, tid: int):
        self.tracks.pop(tid, None)
        try:
            self._rr_queue.remove(tid)
        except ValueError:
            pass

    def next_track_for_service(self) -> Optional[Track]:
        """
        Round-robin confirmed tracks.
        """
        if not self._rr_queue:
            return None

        tid = self._rr_queue.popleft()
        self._rr_queue.append(tid)
        return self.tracks.get(tid, None)
