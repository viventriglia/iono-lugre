"""
Geometry + segment mass integration utilities for ray–sphere interactions
and plasmaspheric mass estimation using PyGCPM.

All coordinates are assumed Cartesian GEO in km unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


# ============================================================
# Types
# ============================================================

ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass(frozen=True)
class Subsegment:
    """Parametric subsegment of a ray."""
    start_t: float
    end_t: float
    code: int  # 0: r < R1, 1: R1 ≤ r ≤ R2, 2: r > R2


@dataclass
class SegmentResult:
    """Result container for one segment."""
    result_tec: float
    result_densities: Optional[np.ndarray]
    result_positions: Optional[np.ndarray]


# ============================================================
# Geometry utilities
# ============================================================

def get_proximity_code(r: float, R1: float, R2: float) -> int:
    """Classify distance from origin with respect to two radii."""
    if r < R1:
        return 0
    if r > R2:
        return 2
    return 1


def get_intersection_t_values(
    P1: np.ndarray,
    V: np.ndarray,
    R: float,
    eps: float = 1e-9
) -> List[float]:
    """
    Solve |P1 + t V| = R for t (line–sphere intersections).
    """
    a = float(np.dot(V, V))
    if a < eps:
        return []

    b = float(2.0 * np.dot(P1, V))
    c = float(np.dot(P1, P1) - R * R)

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return []

    sqrt_disc = np.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2.0 * a)
    t2 = (-b - sqrt_disc) / (2.0 * a)

    return [float(t) for t in (t1, t2) if not np.isnan(t)]


def get_segment_colored_subsegments(
    P1: ArrayLike,
    P2: ArrayLike,
    R1: float,
    R2: float,
    eps: float = 1e-9
) -> List[Subsegment]:
    """
    Split a segment into subsegments classified by radial distance.
    """
    P1 = np.asarray(P1, dtype=float).reshape(3)
    P2 = np.asarray(P2, dtype=float).reshape(3)
    V = P2 - P1

    t_vals: List[float] = [0.0, 1.0]
    t_vals += get_intersection_t_values(P1, V, R1, eps)
    t_vals += get_intersection_t_values(P1, V, R2, eps)

    t_vals = sorted({t for t in t_vals if 0.0 <= t <= 1.0})

    if len(t_vals) < 2:
        code = get_proximity_code(np.linalg.norm(P1), R1, R2)
        return [Subsegment(0.0, 1.0, code)]

    subsegments: List[Subsegment] = []
    for t0, t1 in zip(t_vals[:-1], t_vals[1:]):
        if t1 - t0 < eps:
            continue
        t_mid = 0.5 * (t0 + t1)
        P_mid = P1 + t_mid * V
        code = get_proximity_code(np.linalg.norm(P_mid), R1, R2)
        subsegments.append(Subsegment(t0, t1, code))

    return subsegments


# ============================================================
# Coordinate transforms (spacepy)
# ============================================================

def geo_car_to_sm_car(
    P_geo_km: np.ndarray,
    time,
    Coords,
    Ticktock
) -> np.ndarray:
    """Convert GEO Cartesian (km) to SM Cartesian (km)."""
    coords = Coords(P_geo_km, "GEO", "car")
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    coords.ticks = Ticktock([ts] * len(P_geo_km), "ISO")
    return coords.convert("SM", "car").data


# ============================================================
# Mass integration
# ============================================================

def calculate_segment_mass(
    P1: ArrayLike,
    P2: ArrayLike,
    subsegments: Sequence[Subsegment],
    time,
    N_steps: int,
    *,
    R_Earth_km: float,
    kp_extraction_function: Callable[[str], float],
    PyGCPM,
    Coords,
    Ticktock,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Integrate plasmaspheric mass along subsegments with code == 1.
    """
    if N_steps <= 0:
        raise ValueError("N_steps must be positive")

    P1 = np.asarray(P1, float).reshape(3)
    P2 = np.asarray(P2, float).reshape(3)
    V = P2 - P1

    date_int = int(time.strftime("%Y%m%d"))
    ut = time.hour + time.minute / 60.0
    kp = kp_extraction_function(time.strftime("%Y-%m-%d %H:%M:%S"))

    total_mass = 0.0
    rho_all: List[np.ndarray] = []
    pos_all: List[np.ndarray] = []

    for seg in subsegments:
        if seg.code != 1:
            continue

        t = np.linspace(seg.start_t, seg.end_t, N_steps + 1)
        t_mid = 0.5 * (t[:-1] + t[1:])
        P_mid = P1 + t_mid[:, None] * V

        ds = np.linalg.norm(P1 + t[1] * V - (P1 + t[0] * V))

        P_sm = geo_car_to_sm_car(P_mid, time, Coords, Ticktock) / R_Earth_km

        rho, _, _, _ = PyGCPM.GCPM(
            x=P_sm[:, 0],
            y=P_sm[:, 1],
            z=P_sm[:, 2],
            Date=date_int,
            ut=ut,
            Kp=kp,
            Verbose=False,
        )

        dM = rho * 1e6 * ds * 1e3
        total_mass += np.sum(dM)

        rho_all.append(rho)
        pos_all.append(P_mid)

    if rho_all:
        return total_mass, np.concatenate(rho_all), np.concatenate(pos_all)

    return 0.0, np.empty(0), np.empty((0, 3))


# ============================================================
# High-level API
# ============================================================

def run_single_segment_analysis(
    P1: ArrayLike,
    P2: ArrayLike,
    R1: float,
    R2: float,
    time,
    N_steps_mass: int,
    *,
    R_Earth_km: float,
    kp_extraction_function: Callable[[str], float],
    PyGCPM,
    Coords,
    Ticktock,
) -> SegmentResult:
    """Apply geometric rules and integrate mass if needed."""
    subsegments = get_segment_colored_subsegments(P1, P2, R1, R2)
    codes = {s.code for s in subsegments}

    if 0 in codes or codes == {2}:
        return SegmentResult(0.0, None, None)

    mass, rho, pos = calculate_segment_mass(
        P1, P2, subsegments, time, N_steps_mass,
        R_Earth_km=R_Earth_km,
        kp_extraction_function=kp_extraction_function,
        PyGCPM=PyGCPM,
        Coords=Coords,
        Ticktock=Ticktock,
    )
    return SegmentResult(mass, rho, pos)


def run_segment_analysis(
    segments: Iterable[Tuple[ArrayLike, ArrayLike]],
    times: Sequence,
    N_steps_mass: int,
    R1: float,
    R2: float,
    *,
    R_Earth_km: float,
    kp_extraction_function: Callable[[str], float],
    PyGCPM,
    Coords,
    Ticktock,
    tqdm=None,
) -> List[SegmentResult]:
    """Batch segment processing."""
    iterator = tqdm(segments, desc="Segment analysis") if tqdm else segments

    results: List[SegmentResult] = []
    for i, (P1, P2) in enumerate(iterator):
        results.append(
            run_single_segment_analysis(
                P1, P2, R1, R2, times[i], N_steps_mass,
                R_Earth_km=R_Earth_km,
                kp_extraction_function=kp_extraction_function,
                PyGCPM=PyGCPM,
                Coords=Coords,
                Ticktock=Ticktock,
            )
        )
    return results
