#!/usr/bin/env python3
"""
plot_thermal_log.py

Plots RB3 Gen2 (QCS8550) thermal_zone CSV logs like the ones produced during
NPU inference / thermal throttling tests.

Usage:
    python3 plot_thermal_log.py path/to/log.csv [--out out.png] [--top 10]

What it does:
  - Groups the ~100 thermal_zone columns into subsystem categories
    (cpu_big, cpu_little, gpu, npu/nspss, camera, video, ddr, aoss,
     battery/pmic, modem/rf, other) based on the column name.
  - Draws one subplot per category (elapsed_s on x-axis, deg C on y-axis).
  - Prints a ranked table of the hottest zones (peak temp) so you can
    quickly see which sensor is driving throttling.
  - Skips columns that are entirely empty (many RF/modem zones are NaN
    when the modem isn't active).
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- category rules -------------------------------------------------------
# Order matters: first matching pattern wins.
CATEGORY_RULES = [
    ("NPU (nspss)",     re.compile(r"nspss")),
    ("GPU",              re.compile(r"gpuss")),
    ("CPU big (cpu-1)",  re.compile(r"cpu-1-|cpuss-[1-3]\b")),
    ("CPU little (cpu-0)", re.compile(r"cpu-0-|cpuss-0\b")),
    ("Camera",           re.compile(r"camera")),
    ("Video",            re.compile(r"video")),
    ("DDR",              re.compile(r"\bddr\b")),
    ("AOSS",             re.compile(r"aoss")),
    ("Battery / PMIC",   re.compile(r"battery|pm8550|pmr735|bcl|vbat|ibat")),
    ("Modem / RF",       re.compile(r"mdmss|sdr|mmw|pa1?\b|pa-therm")),
    ("USB / Wireless",   re.compile(r"usb|wls|wireless")),
]


def categorize(col_name: str) -> str:
    for label, pattern in CATEGORY_RULES:
        if pattern.search(col_name):
            return label
    return "Other"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", type=Path, help="Path to the thermal log CSV")
    ap.add_argument("--out", type=Path, default=None,
                     help="Output image path (default: <csv_stem>_thermal.png)")
    ap.add_argument("--top", type=int, default=15,
                     help="How many hottest zones to list in the summary")
    ap.add_argument("--threshold", type=float, default=None,
                     help="Optional throttle threshold (deg C) to draw as a red line")
    args = ap.parse_args()

    if not args.csv_path.exists():
        sys.exit(f"File not found: {args.csv_path}")

    df = pd.read_csv(args.csv_path)

    if "elapsed_s" not in df.columns:
        sys.exit("Expected an 'elapsed_s' column, not found in this CSV.")

    # thermal_zone*_C columns only
    zone_cols = [c for c in df.columns if c.startswith("thermal_zone") and c.endswith("_C")]
    if not zone_cols:
        sys.exit("No thermal_zone*_C columns found.")

    # drop columns that are entirely NaN (e.g. inactive modem/RF sensors)
    zone_cols = [c for c in zone_cols if df[c].notna().any()]

    # --- summary: hottest zones -------------------------------------------
    peaks = df[zone_cols].max().sort_values(ascending=False)
    print(f"\nTop {args.top} hottest zones (peak deg C):")
    print("-" * 45)
    for name, val in peaks.head(args.top).items():
        # strip the boilerplate prefix/suffix for readability
        short = re.sub(r"^thermal_zone\d+_", "", name).replace("_C", "")
        print(f"  {short:<30s} {val:6.1f} C")

    # --- grouping -----------------------------------------------------------
    groups = {}
    for col in zone_cols:
        cat = categorize(col)
        groups.setdefault(cat, []).append(col)

    # keep a stable, useful ordering; push "Other" to the end
    ordered_cats = [c for c, _ in CATEGORY_RULES if c in groups]
    if "Other" in groups:
        ordered_cats.append("Other")

    n = len(ordered_cats)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    x = df["elapsed_s"]

    for ax, cat in zip(axes, ordered_cats):
        cols = groups[cat]
        for col in cols:
            short = re.sub(r"^thermal_zone\d+_", "", col).replace("_C", "")
            ax.plot(x, df[col], linewidth=0.8, alpha=0.8, label=short)
        ax.set_ylabel("deg C")
        ax.set_title(f"{cat}  ({len(cols)} zones)")
        ax.grid(True, alpha=0.3)
        if args.threshold is not None:
            ax.axhline(args.threshold, color="red", linestyle="--", linewidth=1,
                        label=f"threshold {args.threshold:.0f}C")
        # avoid a huge legend when there are many zones in a group
        if len(cols) <= 12:
            ax.legend(fontsize=7, ncol=min(len(cols), 4), loc="upper left")

    axes[-1].set_xlabel("elapsed_s")
    fig.suptitle(f"Thermal log: {args.csv_path.name}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = args.out or args.csv_path.with_name(args.csv_path.stem + "_thermal.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot -> {out_path}")


if __name__ == "__main__":
    main()