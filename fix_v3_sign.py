"""
Corrige la convention de signe de Depth Anything V3 (profondeur directe,
contrairement à V2/MiDaS qui sortent une disparité) en inversant
z_corrected, pour rendre le CSV comparable aux deux autres modèles.
"""
import csv
import sys

def fix_sign(input_path, output_path):
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    zs = [float(r["z_corrected"]) for r in rows]
    z_max_local = max(zs)  # approx : suffisant pour ré-inverser dans la même plage

    for r in rows:
        z = float(r["z_corrected"])
        r["z_corrected"] = str(z_max_local - z)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} lignes corrigées -> {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_v3_sign.py results_depth_anything_v3.csv results_depth_anything_v3_fixed.csv")
        sys.exit(1)
    fix_sign(sys.argv[1], sys.argv[2])