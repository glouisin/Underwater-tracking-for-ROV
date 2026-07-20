#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# analyze_run.py — turns monitoring.py logs into article-ready tables/figures.
#
#   python3 analyze_run.py logs/htp_dav3                       # single run
#   python3 analyze_run.py logs/cpu_dav3 logs/htp_dav3 ...     # comparison
#
# Outputs, written into ./analysis_<timestamp>/ :
#   table_latency.csv / .md      per-stage mean/p50/p95/p99/max  (Table: latency)
#   table_runs.csv   / .md       one row per run: FPS, duty, thermals, control
#   fig_latency_breakdown.png    stacked mean time per stage
#   fig_latency_cdf.png          CDF of end-to-end frame time (shows tail)
#   fig_thermal.png              temps + frequencies vs time, latency overlaid
#   fig_cpu.png                  per-core utilisation vs time
#   fig_control.png              dx/dy/ez vs time + RMS annotation
#   fig_compare_fps.png          bar chart across runs (comparison mode)
#
# Only numpy + matplotlib + pandas required (run this on the Mac, not the board).
# -----------------------------------------------------------------------------
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGE_ORDER = [
    "capture_wait", "preproc_yolo", "preproc_depth", "yolo_invoke", "depth_invoke",
    "postproc_yolo", "nms", "tracking", "depth_sample", "render", "udp_send",
    "gst_push", "throttle_sleep",
]
# Sentinel temperatures of inactive sensors on this SoC.
def _is_active_temp(series):
    v = series.dropna()
    if v.empty:
        return False
    return not (v.max() <= 0 or (v.max() - v.min() < 0.5 and abs(v.mean() - 25) < 2))


def load(run_dir):
    r = {"tag": os.path.basename(os.path.normpath(run_dir)), "dir": run_dir}
    r["frames"] = pd.read_csv(os.path.join(run_dir, "frames.csv"))
    p_sys = os.path.join(run_dir, "system.csv")
    r["system"] = pd.read_csv(p_sys) if os.path.exists(p_sys) else pd.DataFrame()
    for k, f in (("summary", "summary.json"), ("meta", "meta.json")):
        p = os.path.join(run_dir, f)
        r[k] = json.load(open(p)) if os.path.exists(p) else {}
    return r


def stats(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return dict(mean=np.nan, p50=np.nan, p95=np.nan, p99=np.nan, max=np.nan)
    return dict(mean=v.mean(), p50=np.percentile(v, 50), p95=np.percentile(v, 95),
                p99=np.percentile(v, 99), max=v.max())


def table_latency(runs, out):
    rows = []
    for r in runs:
        df = r["frames"]
        for s in STAGE_ORDER + ["npu", "cpu", "frame"]:
            col = {"npu": "t_npu_ms", "cpu": "t_cpu_ms",
                   "frame": "dt_frame_ms"}.get(s, f"t_{s}_ms")
            if col not in df.columns:
                continue
            v = df[col].values
            if np.nanmax(v) <= 0:
                continue
            st = stats(v)
            rows.append(dict(run=r["tag"], stage=s, **{k: round(x, 3) for k, x in st.items()}))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "table_latency.csv"), index=False)
    with open(os.path.join(out, "table_latency.md"), "w") as f:
        f.write(t.to_markdown(index=False))
    return t


def table_runs(runs, out):
    rows = []
    for r in runs:
        df, sysdf, s = r["frames"], r["system"], r.get("summary", {})
        row = {
            "run": r["tag"],
            "frames": s.get("frames_processed", len(df)),
            "duration_s": s.get("duration_s"),
            "fps_true": s.get("fps_true"),
            "loop_rate_hz": s.get("loop_rate_hz"),
            "drop_%": round(100 * (s.get("frame_drop_ratio") or 0), 2),
            "npu_duty_%": round(100 * (s.get("npu_duty_cycle") or 0), 1),
            "lat_e2e_p50": round(stats(df.get("dt_frame_ms", []))["p50"], 2),
            "lat_e2e_p95": round(stats(df.get("dt_frame_ms", []))["p95"], 2),
            "lat_npu_mean": round(stats(df.get("t_npu_ms", []))["mean"], 2),
            "mon_overhead_us": (s.get("monitor_overhead_us") or {}).get("p95"),
        }
        if not sysdf.empty:
            if "cpu_total_pct" in sysdf:
                row["cpu_mean_%"] = round(sysdf["cpu_total_pct"].mean(), 1)
                row["cpu_max_%"] = round(sysdf["cpu_total_pct"].max(), 1)
            if "proc_cpu_pct" in sysdf:
                row["proc_cpu_mean_%"] = round(sysdf["proc_cpu_pct"].mean(), 1)
            if "proc_rss_mb" in sysdf:
                row["rss_max_mb"] = round(sysdf["proc_rss_mb"].max(), 1)
            npu_cols = [c for c in sysdf.columns if "nspss" in c and _is_active_temp(sysdf[c])]
            if npu_cols:
                row["npu_temp_max_C"] = round(sysdf[npu_cols].max().max(), 1)
                row["npu_temp_rise_C"] = round(
                    sysdf[npu_cols].max(axis=1).iloc[-5:].mean()
                    - sysdf[npu_cols].max(axis=1).iloc[:5].mean(), 1)
            cpu_t = [c for c in sysdf.columns
                     if c.startswith("temp_cpu") and _is_active_temp(sysdf[c])]
            if cpu_t:
                row["cpu_temp_max_C"] = round(sysdf[cpu_t].max().max(), 1)
        for axis in ("dx", "dy", "ez"):
            c = (s.get("control") or {}).get(axis)
            if c:
                row[f"{axis}_rms"] = c["rms"]
                row[f"{axis}_p95"] = c["abs_p95"]
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "table_runs.csv"), index=False)
    with open(os.path.join(out, "table_runs.md"), "w") as f:
        f.write(t.to_markdown(index=False))
    return t


def fig_breakdown(runs, out):
    stages = [s for s in STAGE_ORDER
              if any(f"t_{s}_ms" in r["frames"] and r["frames"][f"t_{s}_ms"].max() > 0
                     for r in runs)]
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(runs) + 3))
    bottom = np.zeros(len(runs))
    cmap = plt.get_cmap("tab20")
    for i, s in enumerate(stages):
        vals = np.array([r["frames"].get(f"t_{s}_ms", pd.Series([0])).mean() for r in runs])
        ax.barh([r["tag"] for r in runs], vals, left=bottom, label=s, color=cmap(i % 20))
        bottom += vals
    ax.set_xlabel("mean time per frame (ms)")
    ax.set_title("Per-stage latency breakdown")
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_latency_breakdown.png"), dpi=160)
    plt.close(fig)


def fig_cdf(runs, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in runs:
        v = r["frames"].get("dt_frame_ms")
        if v is None:
            continue
        v = np.sort(v.values[1:])
        ax.plot(v, np.linspace(0, 1, v.size), label=r["tag"])
    ax.set_xlabel("end-to-end frame time (ms)")
    ax.set_ylabel("CDF")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    ax.set_title("Latency distribution (tail matters for control stability)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_latency_cdf.png"), dpi=160)
    plt.close(fig)


def fig_thermal(r, out):
    sysdf = r["system"]
    if sysdf.empty:
        return
    temps = [c for c in sysdf.columns if c.startswith("temp_") and _is_active_temp(sysdf[c])]
    if not temps:
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for c in temps:
        style = "-" if "nspss" in c else "--"
        lw = 2 if "nspss" in c else 1
        axes[0].plot(sysdf["t_rel"], sysdf[c], style, lw=lw, label=c.replace("temp_", ""))
    axes[0].set_ylabel("°C")
    axes[0].legend(ncol=4, fontsize=7)
    axes[0].grid(alpha=.3)
    axes[0].set_title(f"{r['tag']} — thermals (solid = NPU/nspss), frequencies, latency")

    fcols = [c for c in sysdf.columns if c.startswith("freq_cpu")]
    for c in fcols:
        axes[1].plot(sysdf["t_rel"], sysdf[c], lw=1, label=c.replace("freq_", "").replace("_mhz", ""))
    axes[1].set_ylabel("MHz")
    axes[1].legend(ncol=4, fontsize=7)
    axes[1].grid(alpha=.3)

    df = r["frames"]
    if "t_rel" in df:
        w = max(1, len(df) // 200)
        axes[2].plot(df["t_rel"], df["dt_frame_ms"].rolling(w).mean(), label="frame time")
        if "t_npu_ms" in df:
            axes[2].plot(df["t_rel"], df["t_npu_ms"].rolling(w).mean(), label="NPU inference")
    axes[2].set_ylabel("ms")
    axes[2].set_xlabel("t (s)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig_thermal_{r['tag']}.png"), dpi=160)
    plt.close(fig)


def fig_cpu(r, out):
    sysdf = r["system"]
    cols = [c for c in sysdf.columns if c.startswith("cpu") and c.endswith("_pct")
            and c != "cpu_total_pct"]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for c in cols:
        ax.plot(sysdf["t_rel"], sysdf[c], lw=1, label=c)
    if "cpu_total_pct" in sysdf:
        ax.plot(sysdf["t_rel"], sysdf["cpu_total_pct"], "k", lw=2, label="total")
    if "proc_cpu_pct" in sysdf:
        ax.plot(sysdf["t_rel"], sysdf["proc_cpu_pct"], "r--", lw=1.5, label="pipeline process")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("% utilisation")
    ax.set_title(f"{r['tag']} — CPU load (process vs system)")
    ax.legend(ncol=5, fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig_cpu_{r['tag']}.png"), dpi=160)
    plt.close(fig)


def fig_control(r, out):
    df = r["frames"]
    if "dx" not in df.columns:
        return
    d = df[df.get("vision_valid", True) == True] if "vision_valid" in df else df
    if d.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, c, lbl in zip(axes, ["dx", "dy", "ez"],
                          ["dx (norm.)", "dy (norm.)", "ez (norm.)"]):
        if c not in d:
            continue
        v = d[c].astype(float)
        ax.plot(d["t_rel"], v, lw=.8)
        ax.axhline(0, color="k", lw=.8)
        rms = float(np.sqrt((v ** 2).mean()))
        ax.axhline(rms, color="r", ls=":", lw=.8)
        ax.axhline(-rms, color="r", ls=":", lw=.8)
        ax.set_ylabel(lbl)
        ax.grid(alpha=.3)
        ax.set_title(f"RMS = {rms:.4f}   |e|p95 = {np.percentile(np.abs(v), 95):.4f}",
                     fontsize=9, loc="right")
    axes[-1].set_xlabel("t (s)")
    fig.suptitle(f"{r['tag']} — stationkeeping error signals")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig_control_{r['tag']}.png"), dpi=160)
    plt.close(fig)


def fig_compare_fps(runs, out):
    if len(runs) < 2:
        return
    tags = [r["tag"] for r in runs]
    fps = [r["summary"].get("fps_true", np.nan) for r in runs]
    npu = [r["frames"]["t_npu_ms"].mean() if "t_npu_ms" in r["frames"] else np.nan
           for r in runs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.bar(tags, fps, color="#3b7dd8")
    a1.set_ylabel("FPS (true)")
    a1.tick_params(axis="x", rotation=30)
    a1.grid(alpha=.3, axis="y")
    a2.bar(tags, npu, color="#d8733b")
    a2.set_ylabel("mean NPU inference (ms)")
    a2.tick_params(axis="x", rotation=30)
    a2.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_compare_fps.png"), dpi=160)
    plt.close(fig)


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__ or "usage: analyze_run.py <run_dir> [run_dir ...]")
        sys.exit(1)
    out = f"analysis_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out, exist_ok=True)
    runs = [load(d) for d in dirs]

    t1 = table_latency(runs, out)
    t2 = table_runs(runs, out)
    fig_breakdown(runs, out)
    fig_cdf(runs, out)
    fig_compare_fps(runs, out)
    for r in runs:
        fig_thermal(r, out)
        fig_cpu(r, out)
        fig_control(r, out)

    print(t2.to_string(index=False))
    print(f"\nWritten to ./{out}/")


if __name__ == "__main__":
    main()