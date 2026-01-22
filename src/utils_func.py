"""
utils_func.py

General-purpose utilities:
- ArcID parsing
- Geomagnetic Kp extraction
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Union


# ============================================================
# ArcID utilities
# ============================================================

def parse_arc_id(arc_id: str) -> Tuple[str, str]:
    """
    Parse ArcID and infer GNSS constellation.

    Parameters
    ----------
    arc_id : str
        Arc identifier (e.g. 'G30_OP74_102')

    Returns
    -------
    sv_id : str
        Satellite ID (e.g. 'G30')
    constellation : str
        'GPS', 'Galileo', or 'unknown'
    """
    sv_id = arc_id.split("_")[0]

    if sv_id.startswith("E"):
        return sv_id, "Galileo"
    if sv_id.startswith("G"):
        return sv_id, "GPS"

    return sv_id, "unknown"


# ============================================================
# Kp utilities
# ============================================================

def kp_extraction_function(
    date_str: str,
    file_path: Union[str, Path],
) -> float:
    """
    Extract Kp index for a given UTC time from a standard Kp file.

    Kp values are defined in 3-hour bins.

    Parameters
    ----------
    date_str : str
        UTC time in format '%Y-%m-%d %H:%M:%S'
    file_path : str or Path
        Path to Kp_ap_since_1932.txt

    Returns
    -------
    float
        Kp value for the corresponding 3-hour interval.
        Returns 0.0 if not found.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Kp file not found: {file_path}")

    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    interval_start_hour = (dt.hour // 3) * 3

    with file_path.open("r") as f:
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
                and hour == interval_start_hour
            ):
                return float(parts[7])  # Kp column

    return 0.0


# ============================================================
# tangent point calculation
# ============================================================

def calculate_tangent_points(
    df,
    *,
    R_Earth_km: float = 6371.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute tangent (closest-approach) points of rays to Earth's center.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns p1x,p1y,p1z,p2x,p2y,p2z (km)
    R_Earth_km : float
        Earth radius in km

    Returns
    -------
    tp_coords : (N,3) ndarray
        Tangent point Cartesian coordinates (km)
    tp_height : (N,) ndarray
        Tangent point altitude above Earth (km)
    """
    p1 = df[["p1x", "p1y", "p1z"]].to_numpy()
    p2 = df[["p2x", "p2y", "p2z"]].to_numpy()

    v = p2 - p1

    dot_p1_v = np.einsum("ij,ij->i", p1, v)
    dot_v_v = np.einsum("ij,ij->i", v, v)

    # projection parameter (infinite line)
    u = -dot_p1_v / dot_v_v

    tp = p1 + u[:, None] * v
    tp_height = np.linalg.norm(tp, axis=1) - R_Earth_km

    return tp, tp_height