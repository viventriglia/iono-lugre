"""
utils_func.py

General-purpose utilities:
- GNSS ArcID parsing and constellation inference
- Geomagnetic Kp index extraction
- Ray tangent-point computation
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Union


# ============================================================
# GNSS ArcID utilities
# ============================================================

def parse_gnss_arc_id(arc_id: str) -> Tuple[str, str]:
    """
    Parse a GNSS ArcID and infer the satellite constellation.

    Parameters
    ----------
    arc_id : str
        Arc identifier (e.g. 'G30_OP74_102')

    Returns
    -------
    satellite_id : str
        Satellite identifier (e.g. 'G30')
    constellation : str
        GNSS constellation name ('GPS', 'Galileo', or 'unknown')
    """
    satellite_id = arc_id.split("_")[0]

    if satellite_id.startswith("E"):
        return satellite_id, "Galileo"
    if satellite_id.startswith("G"):
        return satellite_id, "GPS"

    return satellite_id, "unknown"


# ============================================================
# Geomagnetic Kp utilities
# ============================================================

def extract_kp_index(
    utc_time_str: str,
    kp_file_path: Union[str, Path],
) -> float:
    """
    Extract the geomagnetic Kp index for a given UTC time.

    Kp values are defined in fixed 3-hour UT intervals.

    Parameters
    ----------
    utc_time_str : str
        UTC time in format '%Y-%m-%d %H:%M:%S'
    kp_file_path : str or Path
        Path to the standard Kp file (e.g. 'Kp_ap_since_1932.txt')

    Returns
    -------
    float
        Kp value for the corresponding 3-hour interval.
        Returns 0.0 if the interval is not found.
    """
    kp_file_path = Path(kp_file_path)

    if not kp_file_path.exists():
        raise FileNotFoundError(f"Kp file not found: {kp_file_path}")

    dt = datetime.strptime(utc_time_str, "%Y-%m-%d %H:%M:%S")
    interval_hour = (dt.hour // 3) * 3

    with kp_file_path.open("r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(float(parts[3]))  # file stores hh.hh

            if (
                year == dt.year
                and month == dt.month
                and day == dt.day
                and hour == interval_hour
            ):
                return float(parts[7])  # Kp column

    return 0.0


# ============================================================
# Ray tangent-point geometry
# ============================================================

def compute_ray_tangent_points(
    ray_segments,
    *,
    earth_radius_km: float = 6371.0,
):
    """
    Compute tangent (closest-approach) points of rays to Earth's center.

    Parameters
    ----------
    ray_segments : list of tuples
        Each element is (ray_start, ray_end), both 3-element sequences [km].
    earth_radius_km : float, optional
        Earth radius in km (default: 6371.0)

    Returns
    -------
    tangent_points : (N, 3) ndarray
        Cartesian coordinates of tangent points [km]
    tangent_altitudes : (N,) ndarray
        Tangent-point altitudes above Earth's surface [km]
    """

    p_start = np.array([seg[0] for seg in ray_segments])
    p_end   = np.array([seg[1] for seg in ray_segments])

    ray_vectors = p_end - p_start

    dot_p_start_v = np.einsum("ij,ij->i", p_start, ray_vectors)
    dot_v_v       = np.einsum("ij,ij->i", ray_vectors, ray_vectors)

    # Projection parameter along infinite line
    u = -dot_p_start_v / dot_v_v

    tangent_points = p_start + u[:, None] * ray_vectors
    tangent_altitudes = np.linalg.norm(tangent_points, axis=1) - earth_radius_km

    return tangent_points, tangent_altitudes
