"""
plotting_func.py

Plot integrated segment results:
- Left panel: modelled TEC vs measured GFLC + IPP altitude colored by latitude
- Right panel: geodetic ray paths with altitude-colored segments and tangent points
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature.nightshade import Nightshade

from typing import Sequence, Union


# ============================================================
# Helpers
# ============================================================

def cartesian_to_latlon(x, y, z):
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
    return lat, lon


def extract_masses(results):
    """
    Accepts either:
    - list of dicts with key 'ResultM'
    - list of objects with attribute ResultM
    """
    masses = []
    for r in results:
        if hasattr(r, "ResultM"):
            masses.append(r.ResultM)
        elif isinstance(r, dict):
            masses.append(r.get("ResultM", np.nan))
        else:
            masses.append(np.nan)
    return np.asarray(masses, dtype=float)


# ============================================================
# Main plotting function
# ============================================================

def plot_integrated_results(
    results,
    times: Sequence,
    sel: pd.DataFrame,
    *,
    title: str = "",
    left_letter: str = "",
    right_letter: str = "",
    font_scale: float = 1.0,
    save: bool = False,
    show_colorbar: bool = True,
    n_geodetics: int = 15,
    label_every: int = 2,
    legend_loc: str = "best",
    PLT_PATH: Union[str, Path, None] = None,
):
    ALPHA_GRID = 0.4
    R_EARTH = 6371.0

    # --------------------------------------------------------
    # 1. Data preparation
    # --------------------------------------------------------

    model_tec = extract_masses(results) / 1e16
    gflc = sel["gflc"].to_numpy()

    shift = model_tec.min() - gflc.min() if len(model_tec) else 0.0
    gflc_shifted = gflc + shift

    # --------------------------------------------------------
    # 2. Figure layout
    # --------------------------------------------------------

    fig = plt.figure(figsize=(20, 9))
    gs = GridSpec(
        2, 2,
        height_ratios=[1, 0.05],
        width_ratios=[1, 1.2],
        hspace=0.2,
        wspace=0.23
    )

    ax_left = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())

    if show_colorbar:
        cax_left = fig.add_subplot(gs[1, 0])
        cax_map = fig.add_subplot(gs[1, 1])

    # --------------------------------------------------------
    # 3. Left panel: TEC + altitude
    # --------------------------------------------------------

    sv_id = title.split("_")[0]
    ax_left.plot(times, model_tec, "k", lw=2, label="Modelled", zorder=10)
    ax_left.plot(sel["time"], gflc_shifted, "tab:red", lw=2, label="Measured", zorder=10)

    ax_left.set_ylabel("TEC [TECu]", fontsize=10 * font_scale)
    ax_left.legend(
        frameon=False,
        loc=legend_loc,
        title=sv_id,
        fontsize=8 * font_scale,
        title_fontsize=8 * font_scale
    )
    ax_left.grid(True, ls="--", alpha=ALPHA_GRID)

    ax_alt = ax_left.twinx()
    norm_lat = plt.Normalize(vmin=-90, vmax=10)

    sc = ax_alt.scatter(
        sel["time"], sel["alt_ip"],
        c=sel["lat_ip"],
        cmap="YlGnBu",
        s=10,
        alpha=0.9,
        norm=norm_lat,
        zorder=1
    )

    ax_alt.set_yscale("log")
    ax_alt.set_ylabel("Altitude [km]", fontsize=10 * font_scale)

    if show_colorbar:
        cb = fig.colorbar(
            sc,
            cax=cax_left,
            orientation="horizontal",
            ticks=np.arange(-90, 11, 20)
        )
        cb.set_label("Latitude", fontsize=9 * font_scale)
        cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}°"))
        cb.ax.tick_params(labelsize=8 * font_scale)

    date_str = sel["time"].dt.date.iloc[0].strftime("%-d %B %Y")
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_left.set_xlabel(date_str, fontsize=10 * font_scale)

    # --------------------------------------------------------
    # 4. Right panel: geodetic rays
    # --------------------------------------------------------

    terminator_date = pd.to_datetime(sel["time"].iloc[len(sel) // 2])
    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, facecolor="#f5f5f5")
    ax_map.add_feature(cfeature.OCEAN, facecolor="#cceeff")
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax_map.add_feature(Nightshade(terminator_date, alpha=0.15))

    gl = ax_map.gridlines(draw_labels=True, lw=0.5, color="gray",
                          alpha=ALPHA_GRID, ls="--")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {"size": 9 * font_scale}
    gl.ylabel_style = {"size": 9 * font_scale}

    colors = plt.get_cmap("inferno")(np.linspace(0, 0.90, 256))
    cmap_h = mcolors.LinearSegmentedColormap.from_list("inferno_cut", colors)
    norm_h = plt.Normalize(vmin=200, vmax=3700)

    idx = np.linspace(0, len(sel) - 1, n_geodetics, dtype=int)
    subset = sel.iloc[idx]

    for i, row in enumerate(subset.itertuples()):
        p1 = np.array([row.p1x, row.p1y, row.p1z])
        p2 = np.array([row.p2x, row.p2y, row.p2z])
        v = p2 - p1

        u = np.linspace(0, 1, 50)
        pts = p1 + u[:, None] * v
        lat, lon = cartesian_to_latlon(pts[:, 0], pts[:, 1], pts[:, 2])
        h = np.linalg.norm(pts, axis=1) - R_EARTH

        for j in range(len(lon) - 1):
            ax_map.plot(
                lon[j:j+2], lat[j:j+2],
                color=cmap_h(norm_h(0.5 * (h[j] + h[j+1]))),
                linewidth=1.5,
                transform=ccrs.Geodetic()
            )

        if i % label_every == 0:
            lt = pd.to_datetime(row.lt_ip).strftime("%H:%M")
            ut = pd.to_datetime(row.ut_time).strftime("%H:%M")
            ax_map.text(lon[0], lat[0], lt,
                        fontsize=7 * font_scale,
                        ha="right", va="bottom",
                        transform=ccrs.PlateCarree())
            ax_map.text(lon[-1], lat[-1], ut,
                        fontsize=7 * font_scale,
                        ha="left", va="bottom",
                        transform=ccrs.PlateCarree())

        u_tp = -np.dot(p1, v) / np.dot(v, v)
        if 0.0 <= u_tp <= 1.0:
            tp = p1 + u_tp * v
            lat_tp, lon_tp = cartesian_to_latlon(*tp)
            ax_map.plot(lon_tp, lat_tp, "ko", ms=3.25,
                        transform=ccrs.Geodetic())

    # --------------------------------------------------------
    # 5. Layout alignment and labels
    # --------------------------------------------------------

    fig.canvas.draw()
    map_pos = ax_map.get_position()
    left_pos = ax_left.get_position()
    ax_left.set_position([left_pos.x0, map_pos.y0,
                          left_pos.width, map_pos.height])

    fig.text(left_pos.x1 - 0.008, map_pos.y0 + 0.05,
             f"{left_letter})",
             fontsize=11 * font_scale,
             fontweight="bold",
             ha="right", va="bottom")

    fig.text(map_pos.x1 - 0.008, map_pos.y0 + 0.05,
             f"{right_letter})",
             fontsize=11 * font_scale,
             fontweight="bold",
             ha="right", va="bottom")

    if show_colorbar:
        cax_left.set_position([left_pos.x0, map_pos.y0 - 0.11,
                               left_pos.width, 0.03])
        cax_map.set_position([map_pos.x0, map_pos.y0 - 0.11,
                              map_pos.width, 0.03])

        sm = plt.cm.ScalarMappable(cmap=cmap_h, norm=norm_h)
        cbm = fig.colorbar(sm, cax=cax_map, orientation="horizontal")
        cbm.set_label("Local height [km]", fontsize=9 * font_scale)
        cbm.ax.tick_params(labelsize=8 * font_scale)

    for ax in (ax_left, ax_alt):
        for side in ax.spines.values():
            side.set_linewidth(0.5)
        ax.tick_params(labelsize=9 * font_scale)

    if save and PLT_PATH is not None:
        PLT_PATH = Path(PLT_PATH)
        PLT_PATH.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLT_PATH / f"{title}.png", dpi=400, bbox_inches="tight")

    plt.show()
