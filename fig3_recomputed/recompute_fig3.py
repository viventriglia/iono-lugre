"""Recompute the Figure 3 GCPM slant TEC with a corrected PyGCPM build.

The published caches (results_<arc>.npy, written 2026-01-26 on Colab) were made
with PyGCPM c2c092a, whose iri/irifun.for declares ionoindx(722), indrz(722).
The ig_rz.dat in use (12/19/2025, 1958-01..2027-11) has 841 entries per series,
so IRI read IG12 out of bounds (~10 instead of ~114 for 2025).

This script keeps the cached ray geometry (GEO midpoints) and re-evaluates GCPM
at every integration node with a build whose arrays are enlarged to 1024. It uses
the repo's own GEO->SM conversion (spacepy), Kp lookup, step table and trapezoid.

usage: python recompute_fig3.py <gcpm_driver.exe> <libpath/> <cache_dir> <obs_tab_new.csv> <Kp_ap_since_1932.txt> <out_dir>
The driver reads "x y z" (SM, Earth radii) lines and prints ne like PyGCPM.GCPM.
"""
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from processing_func import altitude_dependent_step_km, convert_geo_to_sm_cartesian  # noqa: E402
from utils_func import extract_kp_index  # noqa: E402
from spacepy import coordinates as coord  # noqa: E402
from spacepy.time import Ticktock  # noqa: E402

RE = 6371.0
ARCS = ["E02_OP38_100", "E26_OP38_100", "G24_OP74_100", "G30_OP74_102"]


def rebuild_nodes(centers):
    """Integration nodes from cached GEO midpoints (single region-1 subsegment)."""
    d = centers[-1] - centers[0]
    d /= np.linalg.norm(d)
    step = altitude_dependent_step_km(np.linalg.norm(centers[0]) - RE)
    for _ in range(5):  # first step is set by the altitude of the first node
        step = altitude_dependent_step_km(np.linalg.norm(centers[0] - 0.5 * step * d) - RE)
    nodes = np.empty((len(centers) + 1, 3))
    nodes[0] = centers[0] - 0.5 * step * d
    for j, c in enumerate(centers):
        nodes[j + 1] = 2 * c - nodes[j]
    return nodes


def gcpm(driver, libpath, sm, t, kp):
    out = subprocess.run(
        [driver, libpath, str(t.year), str(t.dayofyear), f"{t.hour + t.minute / 60.0:.8f}", f"{kp:.4f}"],
        input="\n".join(f"{x:.9g} {y:.9g} {z:.9g}" for x, y, z in sm), capture_output=True, text=True, check=True,
    ).stdout.split()
    ne = np.array(out, float)
    if len(ne) != len(sm) or not np.all(np.isfinite(ne)) or np.any(ne < 0):
        raise ValueError(f"invalid GCPM output at {t}")
    return ne


def main():
    driver, libpath, cache_dir, obs_csv, kp_file, out_dir = sys.argv[1:7]
    cache_dir, out_dir = Path(cache_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(obs_csv, usecols=["time", "ArcID", "gflc"])
    obs = obs[obs["ArcID"].isin(ARCS)].copy()
    obs["time"] = pd.to_datetime(obs["time"], format="%d-%b-%Y %H:%M:%S")
    kp_memo = {}

    summary, panels = [], {}
    for arc in ARCS:
        sel = obs[obs["ArcID"] == arc].sort_values("time")
        times = pd.date_range(sel["time"].min(), sel["time"].max(), freq="30s")
        cached = np.load(cache_dir / f"results_{arc}.npy", allow_pickle=True)
        if len(cached) != len(times):
            raise ValueError(f"{arc}: {len(cached)} cached rays vs {len(times)} epochs")
        rows = []
        for t, r in zip(times, cached):
            key = (t.date(), t.hour // 3)
            if key not in kp_memo:
                kp_memo[key] = extract_kp_index(t.strftime("%Y-%m-%d %H:%M:%S"), kp_file)
            kp = kp_memo[key]
            old = r.tec / 1e16
            if r.positions is None or len(r.positions) == 0:
                rows.append(dict(time_utc=t, kp=kp, tec_published=old, tec_recomputed=0.0, audit_err=0.0, steps=0))
                continue
            nodes = rebuild_nodes(np.asarray(r.positions, float))
            lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
            audit_err = float(np.dot(np.asarray(r.densities, float), lengths) * 1e-7 - old)
            if abs(audit_err) > 0.01:
                raise ValueError(f"{arc} {t}: node rebuild fails TEC audit by {audit_err:.4f} TECU")
            sm = convert_geo_to_sm_cartesian(nodes, t.to_pydatetime(), coord.Coords, Ticktock) / RE
            ne = gcpm(driver, libpath, sm, t, kp)
            new = float(np.dot(0.5 * (ne[:-1] + ne[1:]), lengths) * 1e-7)
            rows.append(dict(time_utc=t, kp=kp, tec_published=old, tec_recomputed=new, audit_err=audit_err, steps=len(lengths)))
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"gcpm_stec_recomputed_{arc}.csv", index=False, float_format="%.4f")

        g = sel["gflc"].to_numpy()
        shift_pub = df["tec_published"].min() - g.min()
        shift_new = df["tec_recomputed"].min() - g.min()
        panels[arc] = (df, sel["time"].to_numpy(), g, shift_pub, shift_new)
        # model-vs-shifted-GFLC misfit on the 30-s grid (nearest observation)
        gi = pd.Series(g, index=sel["time"]).groupby(level=0).mean().reindex(df["time_utc"], method="nearest")
        summary.append(dict(
            arc=arc, epochs=len(df),
            published_min=df.tec_published.min(), published_max=df.tec_published.max(), published_last=df.tec_published.iloc[-1],
            recomputed_min=df.tec_recomputed.min(), recomputed_max=df.tec_recomputed.max(), recomputed_last=df.tec_recomputed.iloc[-1],
            max_ratio_recomputed_over_published=float((df.tec_recomputed / df.tec_published.where(df.tec_published > 0.5)).max()),
            gflc_shift_published=shift_pub, gflc_shift_recomputed=shift_new,
            rms_model_minus_gflc_published=float(np.sqrt(np.mean((df.tec_published.to_numpy() - (gi.to_numpy() + shift_pub)) ** 2))),
            rms_model_minus_gflc_recomputed=float(np.sqrt(np.mean((df.tec_recomputed.to_numpy() - (gi.to_numpy() + shift_new)) ** 2))),
            max_audit_err=float(df.audit_err.abs().max()),
        ))
        print(f"{arc}: {len(df)} epochs, published max {df.tec_published.max():.2f} -> recomputed max {df.tec_recomputed.max():.2f} TECU", flush=True)

    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False, float_format="%.3f")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, arc in zip(axes.flat, ARCS):
        df, tg, g, sp, sn = panels[arc]
        ax.plot(tg, g + sp, color="#e8a0a0", lw=1, label="GFLC shifted to published model min")
        ax.plot(tg, g + sn, color="#c0392b", lw=1, label="GFLC shifted to recomputed model min")
        ax.plot(df.time_utc, df.tec_published, "--", color="0.45", lw=2, label="GCPM published (overflowed IRI indices)")
        ax.plot(df.time_utc, df.tec_recomputed, "k", lw=2, label="GCPM recomputed (IRI arrays fixed)")
        ax.set_title(arc)
        ax.set_ylabel("Slant TEC [TECU]")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", labelrotation=20)
    axes.flat[0].legend(fontsize=8, loc="best")
    fig.savefig(out_dir / "figure3_gcpm_recomputed.png", dpi=150)


if __name__ == "__main__":
    main()
