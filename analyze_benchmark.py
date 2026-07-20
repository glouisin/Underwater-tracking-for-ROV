#!/usr/bin/env python3
"""
Analyse comparative de plusieurs modèles de profondeur à partir des CSV
produits par benchmark_depth_model.py.

Usage :
    python analyze_benchmark.py results_midas.csv results_depth_anything_v2.csv [...]

Pour chaque CSV (= un modèle), calcule :
  - Régression linéaire z_corrected -> distance réelle (moindres carrés, a/b)
  - RMSE post-calibration
  - Corrélation de Pearson (force de la relation, indépendamment de l'échelle)
  - Inversions de monotonicité (paires de paliers consécutifs où z_corrected
    ne varie pas dans le même sens que la distance réelle -> signe de bruit
    ou de saturation du modèle)
  - Bruit par palier de distance (écart-type de z_corrected à distance
    réelle fixe -> stabilité du modèle à distance constante, indépendant
    de la qualité de la calibration globale)
  - Latence moyenne / p95

Puis affiche un tableau de synthèse trié par RMSE croissant (meilleur en
haut) pour comparer les modèles entre eux.
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np


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


def fit_affine_lstsq(x, y):
    """y ≈ a*x + b, moindres carrés classiques (pas robuste -- benchmark
    contrôlé donc pas d'outliers attendus ; utiliser RANSAC en conditions
    terrain réelles, cf. calibration en production)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return a, b


def compute_rmse(pred, true):
    pred = np.asarray(pred)
    true = np.asarray(true)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def compute_monotonicity_inversions(per_tier_means):
    """per_tier_means : liste de (distance_reelle, z_corrected_moyen) triée
    par distance réelle croissante. Compte les paires consécutives où
    z_corrected ne décroît pas alors que la distance croît (on s'attend à
    z_corrected décroissant avec la distance, cf. convention disparité)."""
    tiers = sorted(per_tier_means, key=lambda t: t[0])
    inversions = 0
    for i in range(len(tiers) - 1):
        _, z_a = tiers[i]
        _, z_b = tiers[i + 1]
        if z_b >= z_a:  # devrait décroître ; sinon inversion
            inversions += 1
    return inversions, len(tiers) - 1


def analyze_model(rows):
    model_name = rows[0]["model_name"]
    dists = [r["distance_cm_reelle"] for r in rows]
    zs = [r["z_corrected"] for r in rows]
    latencies = [r["latency_ms"] for r in rows]

    # Régression + RMSE post-calibration -- FIT LINÉAIRE DIRECT
    # distance ~= a*z + b. Simple mais suppose une relation affine entre
    # z_corrected et la distance réelle, ce qui n'est PAS garanti (MiDaS/
    # Depth Anything sont scale-shift invariants en espace DISPARITÉ
    # (1/distance), pas en espace métrique direct -- cf. discussion sur la
    # formulation du modèle). Si la vraie relation est non-linéaire, l'erreur
    # de ce fit peut dominer le RMSE et masquer les vraies différences de
    # bruit entre modèles.
    a, b = fit_affine_lstsq(zs, dists)
    pred = [a * z + b for z in zs]
    rmse = compute_rmse(pred, dists)

    # Régression + RMSE post-calibration -- FIT EN ESPACE DISPARITÉ
    # 1/distance ~= a_disp*z + b_disp, puis distance = 1/(a_disp*z + b_disp).
    # Cohérent avec la convention scale-shift invariant réelle du modèle ;
    # à comparer au RMSE direct ci-dessus pour trancher laquelle des deux
    # formulations décrit mieux les données réelles collectées.
    dists_arr = np.asarray(dists, dtype=np.float64)
    zs_arr = np.asarray(zs, dtype=np.float64)
    inv_dists = 1.0 / dists_arr
    a_disp, b_disp = fit_affine_lstsq(zs_arr, inv_dists)
    denom = a_disp * zs_arr + b_disp
    # Évite une division par ~0 si le fit dégénère sur des données de test
    denom_safe = np.where(np.abs(denom) < 1e-9, np.sign(denom) * 1e-9 + 1e-12, denom)
    pred_disp = 1.0 / denom_safe
    rmse_disp = compute_rmse(pred_disp, dists_arr)

    # Corrélation de Pearson
    pearson = float(np.corrcoef(dists, zs)[0, 1])

    # Bruit par palier (regroupement par distance réelle -- suppose des
    # paliers discrets comme produits par benchmark_depth_model.py)
    per_tier = defaultdict(list)
    for d, z in zip(dists, zs):
        per_tier[d].append(z)

    per_tier_std = {d: float(np.std(v)) for d, v in per_tier.items()}
    per_tier_mean = [(d, float(np.mean(v))) for d, v in per_tier.items()]
    noise_moyen = float(np.mean(list(per_tier_std.values())))

    inversions, n_pairs = compute_monotonicity_inversions(per_tier_mean)

    return {
        "model_name": model_name,
        "n_frames": len(rows),
        "a": a,
        "b": b,
        "rmse_cm": rmse,
        "a_disp": a_disp,
        "b_disp": b_disp,
        "rmse_disp_cm": rmse_disp,
        "pearson": pearson,
        "noise_moyen_par_palier": noise_moyen,
        "monotonicity_inversions": f"{inversions}/{n_pairs}",
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p95_ms": float(sorted(latencies)[int(0.95 * len(latencies))]),
        "per_tier_std": per_tier_std,
    }


def print_summary(results):
    results_sorted = sorted(results, key=lambda r: min(r["rmse_cm"], r["rmse_disp_cm"]))

    print("=" * 104)
    print(f"{'Modèle':<18}{'RMSE lin(cm)':>13}{'RMSE disp(cm)':>14}{'Pearson':>9}{'Bruit/palier':>13}{'Inversions':>11}{'Lat.moy(ms)':>12}{'Lat.p95(ms)':>12}")
    print("-" * 104)
    for r in results_sorted:
        print(f"{r['model_name']:<18}{r['rmse_cm']:>13.2f}{r['rmse_disp_cm']:>14.2f}{r['pearson']:>9.3f}"
              f"{r['noise_moyen_par_palier']:>13.2f}{r['monotonicity_inversions']:>11}"
              f"{r['latency_mean_ms']:>12.2f}{r['latency_p95_ms']:>12.2f}")
    print("=" * 104)
    print("RMSE lin  = fit direct distance = a*z + b")
    print("RMSE disp = fit en espace disparité, distance = 1/(a*z + b) -- souvent plus fidèle")
    print("            à la formulation scale-shift invariant réelle du modèle")

    print("\nDétail calibration linéaire (z_corrected -> distance_cm = a*z_corrected + b) :")
    for r in results_sorted:
        print(f"  {r['model_name']:<20} a={r['a']:.5f}  b={r['b']:.3f}")

    print("\nDétail calibration disparité (1/distance_cm = a_disp*z_corrected + b_disp) :")
    for r in results_sorted:
        print(f"  {r['model_name']:<20} a_disp={r['a_disp']:.6f}  b_disp={r['b_disp']:.6f}")

    print("\nBruit par palier de distance (écart-type de z_corrected, cm-équivalent) :")
    for r in results_sorted:
        print(f"  {r['model_name']}:")
        for d, std in sorted(r["per_tier_std"].items()):
            print(f"    distance={d:.0f}cm  std={std:.3f}")

    print("\nMeilleur candidat (RMSE le plus bas) :", results_sorted[0]["model_name"])
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Compare plusieurs modèles de profondeur à partir de leurs CSV de benchmark")
    parser.add_argument("csv_files", nargs="+", help="Un ou plusieurs CSV produits par benchmark_depth_model.py")
    args = parser.parse_args()

    all_results = []
    for path in args.csv_files:
        rows = load_results(path)
        if not rows:
            print(f"[!] {path} est vide, ignoré", file=sys.stderr)
            continue
        all_results.append(analyze_model(rows))

    if not all_results:
        print("Aucun résultat valide à analyser.", file=sys.stderr)
        sys.exit(1)

    print_summary(all_results)


if __name__ == "__main__":
    main()