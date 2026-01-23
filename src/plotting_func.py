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


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Computer Modern Serif",
        "DejaVu Serif",
    ],
})

# ============================================================
# Helpers
# ============================================================

def cartesian_to_latlon(x_cart, y_cart, z_cart):
    longitude = np.degrees(np.arctan2(y_cart, x_cart))
    latitude = np.degrees(np.arctan2(z_cart, np.hypot(x_cart, y_cart)))
    return latitude, longitude


def extract_tec_values(results):
    """
    Accepts either:
    - list of dicts with key 'result_tec'
    - list of objects with attribute result_tec
    """
    tec_values = []
    for result in results:
        if hasattr(result, "tec"):
            tec_values.append(result.tec)
        elif isinstance(result, dict):
            tec_values.append(result.get("tec", np.nan))
        else:
            tec_values.append(np.nan)
    return np.asarray(tec_values, dtype=float)


# ============================================================
# Main plotting function
# ============================================================

def plot_integrated_results(
    results,
    times: Sequence,
    selection_df: pd.DataFrame,
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
    EARTH_RADIUS_KM = 6371.0

    # --------------------------------------------------------
    # 1. Data preparation
    # --------------------------------------------------------

    model_tec = extract_tec_values(results) / 1e16
    gflc_values = selection_df["gflc"].to_numpy()

    vertical_shift = model_tec.min() - gflc_values.min() if len(model_tec) else 0.0
    gflc_shifted = gflc_values + vertical_shift

    # --------------------------------------------------------
    # 2. Figure layout
    # --------------------------------------------------------

    fig = plt.figure(figsize=(20, 9))
    grid_spec = GridSpec(
        2, 2,
        height_ratios=[1, 0.05],
        width_ratios=[1, 1.2],
        hspace=0.2,
        wspace=0.23
    )

    ax_left = fig.add_subplot(grid_spec[0, 0])
    ax_map = fig.add_subplot(grid_spec[0, 1], projection=ccrs.PlateCarree())

    if show_colorbar:
        cax_left = fig.add_subplot(grid_spec[1, 0])
        cax_map = fig.add_subplot(grid_spec[1, 1])

    # --------------------------------------------------------
    # 3. Left panel: TEC + altitude
    # --------------------------------------------------------

    satellite_id = title.split("_")[0]
    ax_left.plot(times, model_tec, "k", lw=2, label="Modelled", zorder=10)
    ax_left.plot(selection_df["time"], gflc_shifted, "tab:red", lw=2,
                 label="Measured", zorder=10)

    ax_left.set_ylabel("TEC [TECu]", fontsize=10 * font_scale)
    ax_left.legend(
        frameon=False,
        loc=legend_loc,
        title=satellite_id,
        fontsize=8 * font_scale,
        title_fontsize=8 * font_scale
    )
    ax_left.grid(True, ls="--", alpha=ALPHA_GRID)

    ax_altitude = ax_left.twinx()
    latitude_norm = plt.Normalize(vmin=-90, vmax=10)

    scatter_altitude = ax_altitude.scatter(
        selection_df["time"], selection_df["alt_ip"],
        c=selection_df["lat_ip"],
        cmap="YlGnBu",
        s=10,
        alpha=0.9,
        norm=latitude_norm,
        zorder=1
    )

    ax_altitude.set_yscale("log")
    ax_altitude.set_ylabel("Altitude [km]", fontsize=10 * font_scale)

    if show_colorbar:
        colorbar_left = fig.colorbar(
            scatter_altitude,
            cax=cax_left,
            orientation="horizontal",
            ticks=np.arange(-90, 11, 20)
        )
        colorbar_left.set_label("Latitude", fontsize=9 * font_scale)
        colorbar_left.ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{x:.0f}°")
        )
        colorbar_left.ax.tick_params(labelsize=8 * font_scale)

    date_label = selection_df["time"].dt.date.iloc[0].strftime("%-d %B %Y")
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_left.set_xlabel(date_label, fontsize=10 * font_scale)

    # --------------------------------------------------------
    # 4. Right panel: geodetic rays
    # --------------------------------------------------------

    terminator_time = pd.to_datetime(selection_df["time"].iloc[len(selection_df) // 2])
    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, facecolor="#f5f5f5")
    ax_map.add_feature(cfeature.OCEAN, facecolor="#cceeff")
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax_map.add_feature(Nightshade(terminator_time, alpha=0.15))

    gridlines = ax_map.gridlines(draw_labels=True, lw=0.5, color="gray",
                                 alpha=ALPHA_GRID, ls="--")
    gridlines.top_labels = gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 9 * font_scale}
    gridlines.ylabel_style = {"size": 9 * font_scale}

    inferno_colors = plt.get_cmap("inferno")(np.linspace(0, 0.90, 256))
    altitude_cmap = mcolors.LinearSegmentedColormap.from_list(
        "inferno_cut", inferno_colors
    )
    altitude_norm = plt.Normalize(vmin=200, vmax=3700)

    ray_indices = np.linspace(0, len(selection_df) - 1, n_geodetics, dtype=int)
    ray_subset = selection_df.iloc[ray_indices]

    for ray_idx, ray_row in enumerate(ray_subset.itertuples()):
        ray_start = np.array([ray_row.p1x, ray_row.p1y, ray_row.p1z])
        ray_end = np.array([ray_row.p2x, ray_row.p2y, ray_row.p2z])
        ray_vector = ray_end - ray_start

        ray_param = np.linspace(0, 1, 50)
        ray_points = ray_start + ray_param[:, None] * ray_vector
        latitudes, longitudes = cartesian_to_latlon(
            ray_points[:, 0], ray_points[:, 1], ray_points[:, 2]
        )
        altitudes = np.linalg.norm(ray_points, axis=1) - EARTH_RADIUS_KM

        for seg_idx in range(len(longitudes) - 1):
            ax_map.plot(
                longitudes[seg_idx:seg_idx + 2],
                latitudes[seg_idx:seg_idx + 2],
                color=altitude_cmap(
                    altitude_norm(0.5 * (altitudes[seg_idx] + altitudes[seg_idx + 1]))
                ),
                linewidth=1.5,
                transform=ccrs.Geodetic()
            )

        if ray_idx % label_every == 0:
            local_time = pd.to_datetime(ray_row.lt_ip).strftime("%H:%M")
            universal_time = pd.to_datetime(ray_row.ut_time).strftime("%H:%M")
            ax_map.text(longitudes[0], latitudes[0], local_time,
                        fontsize=7 * font_scale,
                        ha="right", va="bottom",
                        transform=ccrs.PlateCarree())
            ax_map.text(longitudes[-1], latitudes[-1], universal_time,
                        fontsize=7 * font_scale,
                        ha="left", va="bottom",
                        transform=ccrs.PlateCarree())

        tangent_param = -np.dot(ray_start, ray_vector) / np.dot(ray_vector, ray_vector)
        if 0.0 <= tangent_param <= 1.0:
            tangent_point = ray_start + tangent_param * ray_vector
            lat_tp, lon_tp = cartesian_to_latlon(*tangent_point)
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

        scalar_mappable = plt.cm.ScalarMappable(
            cmap=altitude_cmap, norm=altitude_norm
        )
        colorbar_map = fig.colorbar(
            scalar_mappable, cax=cax_map, orientation="horizontal"
        )
        colorbar_map.set_label("Local height [km]", fontsize=9 * font_scale)
        colorbar_map.ax.tick_params(labelsize=8 * font_scale)

    for axis in (ax_left, ax_altitude):
        for side in axis.spines.values():
            side.set_linewidth(0.5)
        axis.tick_params(labelsize=9 * font_scale)

    if save and PLT_PATH is not None:
        PLT_PATH = Path(PLT_PATH)
        PLT_PATH.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLT_PATH / f"{title}.png", dpi=400, bbox_inches="tight")

    plt.show()
