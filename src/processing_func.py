"""
Geometry and tec-integration utilities for ray–sphere interactions
and plasmaspheric tec estimation using PyGCPM.

The module:
- splits ray segments by radial distance from Earth,
- applies altitude-dependent adaptive or fixed-step integration,
- integrates plasmaspheric tec using trapezoidal rule.

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
class RaySubsegment:
    """
    Parametric subsegment of a ray.

    Parameters
    ----------
    t_start, t_end : float
        Parametric limits (0 ≤ t ≤ 1) along the original ray.
    region_code : int
        Radial classification:
        0 → r < R_inner
        1 → R_inner ≤ r ≤ R_outer (integration region)
        2 → r > R_outer
    """
    t_start: float
    t_end: float
    region_code: int


@dataclass
class RayIntegrationResult:
    """
    Result container for one ray segment.

    Attributes
    ----------
    total_tec : float
        Integrated tec contribution.
    densities : ndarray or None
        Density averaged over each integration element.
    positions : ndarray or None
        GEO Cartesian midpoints of integration elements (km).
    """
    tec: float
    densities: Optional[np.ndarray]
    positions: Optional[np.ndarray]


# ============================================================
# Geometry utilities
# ============================================================

def classify_radius_region(
    radius_km: float,
    R_inner_km: float,
    R_outer_km: float
) -> int:
    """
    Classify a radial distance with respect to two spherical boundaries.
    """
    if radius_km < R_inner_km:
        return 0
    if radius_km > R_outer_km:
        return 2
    return 1


def line_sphere_intersections_t(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    sphere_radius_km: float,
    eps: float = 1e-9
) -> List[float]:
    """
    Compute parametric intersections between a line and a sphere.

    Solves ||ray_origin + t·ray_direction|| = sphere_radius_km.
    """
    a = float(np.dot(ray_direction, ray_direction))
    if a < eps:
        return []

    b = float(2.0 * np.dot(ray_origin, ray_direction))
    c = float(np.dot(ray_origin, ray_origin) - sphere_radius_km ** 2)

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return []

    s = np.sqrt(discriminant)
    t1 = (-b + s) / (2.0 * a)
    t2 = (-b - s) / (2.0 * a)

    return [float(t) for t in (t1, t2) if not np.isnan(t)]


def split_ray_by_radial_boundaries(
    ray_start: ArrayLike,
    ray_end: ArrayLike,
    R_inner_km: float,
    R_outer_km: float,
    eps: float = 1e-9
) -> List[RaySubsegment]:
    """
    Split a ray segment into radially classified subsegments.
    """
    P0 = np.asarray(ray_start, float).reshape(3)
    P1 = np.asarray(ray_end, float).reshape(3)
    ray_vec = P1 - P0

    t_candidates = [0.0, 1.0]
    t_candidates += line_sphere_intersections_t(P0, ray_vec, R_inner_km, eps)
    t_candidates += line_sphere_intersections_t(P0, ray_vec, R_outer_km, eps)

    t_candidates = sorted({t for t in t_candidates if 0.0 <= t <= 1.0})

    if len(t_candidates) < 2:
        r0 = np.linalg.norm(P0)
        code = classify_radius_region(r0, R_inner_km, R_outer_km)
        return [RaySubsegment(0.0, 1.0, code)]

    subsegments: List[RaySubsegment] = []
    for t0, t1 in zip(t_candidates[:-1], t_candidates[1:]):
        if t1 - t0 < eps:
            continue
        t_mid = 0.5 * (t0 + t1)
        r_mid = np.linalg.norm(P0 + t_mid * ray_vec)
        code = classify_radius_region(r_mid, R_inner_km, R_outer_km)
        subsegments.append(RaySubsegment(t0, t1, code))

    return subsegments


# ============================================================
# Integration control
# ============================================================

INTEGRATION_MODE_PIECEWISE = "piecewise"
INTEGRATION_MODE_FIXED = "fixed"

ALTITUDE_STEP_TABLE_KM = (
    (100.0, 50.0),
    (200.0, 20.0),
    (600.0, 10.0),
    (1000.0, 40.0),
    (2000.0, 50.0),
    (5000.0, 300.0),
    (10000.0, 500.0),
)

DEFAULT_FIXED_STEP_KM = 500.0


def altitude_dependent_step_km(altitude_km: float) -> float:
    """
    Integration step length as a function of altitude.
    """
    for alt_max, step in ALTITUDE_STEP_TABLE_KM:
        if altitude_km < alt_max:
            return step
    return 1000.0


def build_parametric_steps_piecewise(
    ray_origin: np.ndarray,
    ray_vector: np.ndarray,
    t_start: float,
    t_end: float,
    *,
    earth_radius_km: float,
) -> np.ndarray:
    """
    Construct non-uniform parametric steps using altitude-dependent step size.
    """
    t_values = [t_start]
    t = t_start

    ray_length = np.linalg.norm(ray_vector)
    if ray_length < 1e-12:
        return np.array([t_start, t_end])

    while t < t_end:
        position = ray_origin + t * ray_vector
        altitude_km = np.linalg.norm(position) - earth_radius_km
        dt = altitude_dependent_step_km(altitude_km) / ray_length

        t_next = min(t + dt, t_end)
        if t_next <= t:
            break

        t_values.append(t_next)
        t = t_next

    return np.array(t_values)


def build_parametric_steps_fixed(
    ray_vector: np.ndarray,
    t_start: float,
    t_end: float,
    *,
    step_km: float,
) -> np.ndarray:
    """
    Construct uniform parametric steps using a constant physical step length.
    """
    ray_length = np.linalg.norm(ray_vector)
    if ray_length < 1e-12:
        return np.array([t_start, t_end])

    if step_km <= 0:
        raise ValueError("step_km must be positive")

    dt = step_km / ray_length
    n_steps = max(1, int(np.ceil((t_end - t_start) / dt)))
    return np.linspace(t_start, t_end, n_steps + 1)


def build_parametric_steps(
    ray_origin: np.ndarray,
    ray_vector: np.ndarray,
    t_start: float,
    t_end: float,
    *,
    integration_mode: str,
    earth_radius_km: float,
    fixed_step_km: float,
) -> np.ndarray:
    """
    Dispatch parametric step construction based on integration mode.
    """
    if integration_mode == INTEGRATION_MODE_PIECEWISE:
        return build_parametric_steps_piecewise(
            ray_origin, ray_vector, t_start, t_end,
            earth_radius_km=earth_radius_km,
        )

    if integration_mode == INTEGRATION_MODE_FIXED:
        return build_parametric_steps_fixed(
            ray_vector, t_start, t_end,
            step_km=fixed_step_km,
        )

    raise ValueError(f"Unknown integration mode: {integration_mode}")


# ============================================================
# Coordinate transforms
# ============================================================

def convert_geo_to_sm_cartesian(
    positions_geo_km: np.ndarray,
    time,
    Coords,
    Ticktock
) -> np.ndarray:
    """
    Convert GEO Cartesian coordinates to SM Cartesian coordinates.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    coords = Coords(positions_geo_km, "GEO", "car")
    coords.ticks = Ticktock([ts] * len(positions_geo_km), "ISO")
    return coords.convert("SM", "car").data


# ============================================================
# tec integration
# ============================================================

def integrate_ray_density(
    ray_start: ArrayLike,
    ray_end: ArrayLike,
    ray_subsegments: Sequence[RaySubsegment],
    time,
    *,
    integration_mode: str = INTEGRATION_MODE_PIECEWISE,
    fixed_step_km: float = DEFAULT_FIXED_STEP_KM,
    earth_radius_km: float,
    kp_extractor: Callable[[str], float],
    PyGCPM,
    Coords,
    Ticktock,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Integrate plasmaspheric density along valid ray subsegments.
    """
    P0 = np.asarray(ray_start, float).reshape(3)
    P1 = np.asarray(ray_end, float).reshape(3)
    ray_vec = P1 - P0

    date_int = int(time.strftime("%Y%m%d"))
    ut_hours = time.hour + time.minute / 60.0
    kp = kp_extractor(time.strftime("%Y-%m-%d %H:%M:%S"))

    total_tec = 0.0
    densities_all: List[np.ndarray] = []
    positions_all: List[np.ndarray] = []

    for seg in ray_subsegments:
        if seg.region_code != 1:
            continue

        t_steps = build_parametric_steps(
            P0, ray_vec,
            seg.t_start, seg.t_end,
            integration_mode=integration_mode,
            earth_radius_km=earth_radius_km,
            fixed_step_km=fixed_step_km,
        )
        if len(t_steps) < 2:
            continue

        positions_ext = P0 + t_steps[:, None] * ray_vec
        segment_lengths = np.linalg.norm(
            positions_ext[1:] - positions_ext[:-1], axis=1
        )

        positions_sm = (
            convert_geo_to_sm_cartesian(
                positions_ext, time, Coords, Ticktock
            ) / earth_radius_km
        )

        rho_ext, _, _, _ = PyGCPM.GCPM(
            x=positions_sm[:, 0],
            y=positions_sm[:, 1],
            z=positions_sm[:, 2],
            Date=date_int,
            ut=ut_hours,
            Kp=kp,
            Verbose=False,
        )

        rho_avg = 0.5 * (rho_ext[:-1] + rho_ext[1:])
        total_tec += np.sum(rho_avg * 1e6 * segment_lengths * 1e3)

        positions_all.append(
            0.5 * (positions_ext[:-1] + positions_ext[1:])
        )
        densities_all.append(rho_avg)

    if densities_all:
        return (
            total_tec,
            np.concatenate(densities_all),
            np.concatenate(positions_all),
        )

    return 0.0, np.empty(0), np.empty((0, 3))


# ============================================================
# High-level API
# ============================================================

def process_single_ray(
    ray_start: ArrayLike,
    ray_end: ArrayLike,
    R_inner_km: float,
    R_outer_km: float,
    time,
    *,
    integration_mode: str = INTEGRATION_MODE_PIECEWISE,
    fixed_step_km: float = DEFAULT_FIXED_STEP_KM,
    earth_radius_km: float,
    kp_extractor: Callable[[str], float],
    PyGCPM,
    Coords,
    Ticktock,
) -> RayIntegrationResult:
    """
    Apply geometric screening and tec integration to a single ray.
    """
    subsegments = split_ray_by_radial_boundaries(
        ray_start, ray_end, R_inner_km, R_outer_km
    )
    codes = {s.region_code for s in subsegments}

    if 0 in codes or codes == {2}:
        return RayIntegrationResult(0.0, None, None)

    tec, rho, pos = integrate_ray_density(
        ray_start, ray_end, subsegments, time,
        integration_mode=integration_mode,
        fixed_step_km=fixed_step_km,
        earth_radius_km=earth_radius_km,
        kp_extractor=kp_extractor,
        PyGCPM=PyGCPM,
        Coords=Coords,
        Ticktock=Ticktock,
    )

    return RayIntegrationResult(tec, rho, pos)


def process_ray_batch(
    ray_segments: Iterable[Tuple[ArrayLike, ArrayLike]],
    times: Sequence,
    R_inner_km: float,
    R_outer_km: float,
    *,
    integration_mode: str = INTEGRATION_MODE_PIECEWISE,
    fixed_step_km: float = DEFAULT_FIXED_STEP_KM,
    earth_radius_km: float,
    kp_extractor: Callable[[str], float],
    PyGCPM,
    Coords,
    Ticktock,
    tqdm=None,
) -> List[RayIntegrationResult]:
    """
    Batch processing of multiple rays.
    """
    iterator = tqdm(ray_segments, desc="Ray integration") if tqdm else ray_segments

    results: List[RayIntegrationResult] = []
    for i, (ray_start, ray_end) in enumerate(iterator):
        results.append(
            process_single_ray(
                ray_start, ray_end,
                R_inner_km, R_outer_km, times[i],
                integration_mode=integration_mode,
                fixed_step_km=fixed_step_km,
                earth_radius_km=earth_radius_km,
                kp_extractor=kp_extractor,
                PyGCPM=PyGCPM,
                Coords=Coords,
                Ticktock=Ticktock,
            )
        )
    return results
