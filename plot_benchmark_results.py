#!/usr/bin/env python3
"""
Trace les courbes de calibration/benchmark à partir des CSV produits par
benchmark_depth_model.py.

Usage:
    python plot_benchmark_results.py results_midas.csv
    python plot_benchmark_results.py results_midas.csv results_depth_anything_v2.csv

Génère un PNG multi-panneaux :
  1. Nuage de points brut : distance réelle vs z_corrected, par modèle
  2. Courbe de calibration ajustée : fit linéaire ET fit en espace disparité,
     superposés aux points réels -- pour voir visuellement laquelle colle
     le mieux (cf. discussion sur le fit en espace 1/distance)
  3. Résidus (erreur de prédiction) par palier de distance
  4. Latence d'inférence par modèle (boxplot)

Un panneau supplémentaire "bruit par palier" (écart-type de z_corrected à
distance réelle fixe) est ajouté si plusieurs frames existent par palier.
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_results(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "frame_id": int(row["frame_id"]),
                "model_name": row["model_name"],
                "distance_cm_reelle": float(row["distance_cm_reelle"]),
                "z_raw": float(row["z_raw"]),
                "z_corrected": float(row["z_corrected"]),
                "latency_ms": float(row["latency_ms"]),
            })
    return rows


def fit_linear(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return a, b


def fit_disparity(z, dist):
    """1/dist ~= a*z + b, retourne les coefficients pour reconstruire
    dist_pred = 1/(a*z+b)."""
    z = np.asarray(z, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)
    inv_dist = 1.0 / dist
    a, b = fit_linear(z, inv_dist)
    return a, b


def main():
    parser = argparse.ArgumentParser(description="Trace les courbes de calibration/benchmark de profondeur")
    parser.add_argument("csv_files", nargs="+", help="CSV produits par benchmark_depth_model.py")
    parser.add_argument("--output", default="benchmark_plots.png", help="fichier image de sortie")
    args = parser.parse_args()

    all_data = {}
    for path in args.csv_files:
        rows = load_results(path)
        if rows:
            all_data[rows[0]["model_name"]] = rows

    if not all_data:
        print("Aucune donnée valide trouvée.")
        return

    colors = plt.cm.tab10.colors
    model_names = list(all_data.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_scatter, ax_calib, ax_residuals, ax_latency = axes.flatten()

    # ---- Panel 1: raw scatter plot ----
    for i, (model, rows) in enumerate(all_data.items()):
        dists = [r["distance_cm_reelle"] for r in rows]
        zs = [r["z_corrected"] for r in rows]
        ax_scatter.scatter(dists, zs, s=8, alpha=0.4, color=colors[i % 10], label=model)
    ax_scatter.set_xlabel("Ground-truth distance (cm)")
    ax_scatter.set_ylabel("z_corrected (raw)")
    ax_scatter.set_title("Raw scatter plot")
    ax_scatter.legend()
    ax_scatter.grid(alpha=0.3)

    # ---- Panel 2: fitted calibration (linear vs disparity) ----
    for i, (model, rows) in enumerate(all_data.items()):
        dists = np.array([r["distance_cm_reelle"] for r in rows])
        zs = np.array([r["z_corrected"] for r in rows])

        ax_calib.scatter(zs, dists, s=8, alpha=0.3, color=colors[i % 10], label=f"{model} (measured)")

        z_range = np.linspace(zs.min(), zs.max(), 200)

        a_lin, b_lin = fit_linear(zs, dists)
        pred_lin = a_lin * z_range + b_lin
        ax_calib.plot(z_range, pred_lin, "--", color=colors[i % 10], alpha=0.8,
                      label=f"{model} linear fit")

        a_disp, b_disp = fit_disparity(zs, dists)
        denom = a_disp * z_range + b_disp
        denom_safe = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        pred_disp = 1.0 / denom_safe
        ax_calib.plot(z_range, pred_disp, "-", color=colors[i % 10], alpha=0.9,
                      label=f"{model} disparity fit")

    ax_calib.set_xlabel("z_corrected (raw)")
    ax_calib.set_ylabel("Ground-truth distance (cm)")
    ax_calib.set_title("Calibration: linear vs disparity fit")
    ax_calib.legend(fontsize=8)
    ax_calib.grid(alpha=0.3)

    # ---- Panel 3: residuals per tier (disparity fit) ----
    for i, (model, rows) in enumerate(all_data.items()):
        dists = np.array([r["distance_cm_reelle"] for r in rows])
        zs = np.array([r["z_corrected"] for r in rows])

        a_disp, b_disp = fit_disparity(zs, dists)
        denom = a_disp * zs + b_disp
        denom_safe = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        pred_disp = 1.0 / denom_safe
        residuals = pred_disp - dists

        # Mean residual per ground-truth distance tier (discrete tiers)
        per_tier = defaultdict(list)
        for d, r in zip(dists, residuals):
            per_tier[d].append(r)
        tiers_sorted = sorted(per_tier.keys())
        means = [np.mean(per_tier[t]) for t in tiers_sorted]
        stds = [np.std(per_tier[t]) for t in tiers_sorted]

        ax_residuals.errorbar(tiers_sorted, means, yerr=stds, marker="o",
                               color=colors[i % 10], label=model, capsize=3)

    ax_residuals.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax_residuals.set_xlabel("Ground-truth distance (cm)")
    ax_residuals.set_ylabel("Residual (predicted - actual, cm)")
    ax_residuals.set_title("Calibration error per tier (disparity fit)")
    ax_residuals.legend()
    ax_residuals.grid(alpha=0.3)

    # ---- Panel 4: latency per model ----
    latency_data = [[r["latency_ms"] for r in rows] for rows in all_data.values()]
    bp = ax_latency.boxplot(latency_data, tick_labels=model_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax_latency.set_ylabel("Inference latency (ms)")
    ax_latency.set_title("Latency distribution per model")
    ax_latency.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plots saved -> {args.output}")


if __name__ == "__main__":
    main()