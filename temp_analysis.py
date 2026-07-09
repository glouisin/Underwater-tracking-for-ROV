#!/usr/bin/env python3
"""
Log les temperatures des thermal zones du board (RB3 Gen2 / QCS8550)
dans un CSV, pour tracer une courbe de temperature dans le temps.

Usage:
    python3 log_temperature.py                     # log toutes les 2s, indefiniment (Ctrl+C pour arreter)
    python3 log_temperature.py --interval 5         # toutes les 5s
    python3 log_temperature.py --duration 600       # arret automatique apres 600s
    python3 log_temperature.py --out temps.csv      # nom de fichier custom

Le CSV contient une colonne par thermal_zone detecte + timestamp + elapsed_s.
Fonctionne sur rov2 et rov-test (meme mecanisme sysfs sur les deux boards).

Sur ce board, le NPU/Hexagon (HTP) est expose sous le nom "nspss" (Neural
Signal Processor) : nspss-0 a nspss-3 (thermal_zone20 a 23) -- pas sous les
noms plus communement documentes (npu/hexagon/cdsp/adsp/hvx). Les deux
familles de mots-cles sont incluses ci-dessous pour rester robuste si tu
reutilises ce script sur un autre board/firmware.
"""

import argparse
import csv
import glob
import os
import time
import signal
import sys

THERMAL_BASE = "/sys/class/thermal"

# Mots-cles typiques des zones thermiques liees au NPU/DSP sur les SoC Qualcomm
# (le HTP/NPU tourne sur le coeur Hexagon, expose parfois comme cdsp/adsp/hvx,
# ou "nspss"/"nsp" -- Neural Signal Processor -- sur ce board precis)
NPU_KEYWORDS = ["npu", "hexagon", "cdsp", "adsp", "hvx", "dsp", "nspss", "nsp"]

# Valeurs sentinelles observees sur les zones inactives/non cablees de ce
# firmware : a exclure de l'analyse (pas du log -- on log tout, mais on
# previent l'utilisateur) car elles ne refletent aucune mesure physique
# reelle (capteur absent, non alimente, ou lecture par defaut).
SENTINEL_TEMPS_C = [-40.0, 0.0]
SENTINEL_TOLERANCE_C = 0.5


def discover_zones():
    """Trouve tous les thermal_zoneN disponibles et leur 'type' (nom lisible)."""
    zones = {}
    for path in sorted(glob.glob(f"{THERMAL_BASE}/thermal_zone*")):
        zone_id = os.path.basename(path)
        type_path = os.path.join(path, "type")
        temp_path = os.path.join(path, "temp")
        if not os.path.exists(temp_path):
            continue
        try:
            with open(type_path) as f:
                zone_type = f.read().strip()
        except OSError:
            zone_type = zone_id
        zones[zone_id] = {"type": zone_type, "temp_path": temp_path}
    return zones


def read_temp_c(temp_path):
    """Lit une temperature en millidegres et convertit en degres C."""
    try:
        with open(temp_path) as f:
            raw = int(f.read().strip())
        return raw / 1000.0
    except (OSError, ValueError):
        return None


def is_sentinel(temp_c):
    """Detecte une valeur sentinelle probable (capteur inactif/non cable)."""
    if temp_c is None:
        return False
    return any(abs(temp_c - s) <= SENTINEL_TOLERANCE_C for s in SENTINEL_TEMPS_C)


def main():
    parser = argparse.ArgumentParser(description="Log temperature -> CSV")
    parser.add_argument("--interval", type=float, default=2.0, help="secondes entre mesures (defaut: 2.0)")
    parser.add_argument("--duration", type=float, default=None, help="duree totale en secondes (defaut: illimite)")
    parser.add_argument("--out", type=str, default="temperature_log.csv", help="fichier CSV de sortie")
    args = parser.parse_args()

    zones = discover_zones()
    if not zones:
        print("Aucun thermal_zone trouve sous /sys/class/thermal. Verifie que tu es bien sur le board.")
        sys.exit(1)

    zone_ids = list(zones.keys())
    print("Zones detectees:")
    npu_zone_ids = []
    for zid in zone_ids:
        ztype = zones[zid]["type"]
        is_npu = any(kw in ztype.lower() for kw in NPU_KEYWORDS)
        tag = "  <-- probable NPU/DSP" if is_npu else ""
        if is_npu:
            npu_zone_ids.append(zid)
        print(f"  {zid} -> {ztype}{tag}")

    if not npu_zone_ids:
        print(
            "\nAucune zone ne matche les mots-cles NPU/DSP "
            f"({', '.join(NPU_KEYWORDS)}). Le HTP n'expose peut-etre pas de "
            "thermal_zone dedie sur ce firmware -- toutes les zones sont "
            "quand meme loguees ci-dessous, verifie 'cpu-1-x-usr' ou "
            "'aoss-0-usr' qui peuvent refleter la charge globale du SoC."
        )
    else:
        print(f"\n{len(npu_zone_ids)} zone(s) NPU/DSP detectee(s) : {', '.join(npu_zone_ids)}")

    # Verification rapide des valeurs sentinelles au demarrage (capteurs
    # inactifs/non cables) -- purement informatif, n'affecte pas le log.
    sentinel_zones = []
    for zid in zone_ids:
        t0_check = read_temp_c(zones[zid]["temp_path"])
        if is_sentinel(t0_check):
            sentinel_zones.append((zid, t0_check))
    if sentinel_zones:
        print(
            f"\n[!] {len(sentinel_zones)} zone(s) affichent une valeur "
            f"sentinelle probable (capteur inactif) : "
            + ", ".join(f"{zid}={t:.1f}C" for zid, t in sentinel_zones)
        )
        print("    Ces zones sont quand meme loguees, mais a exclure de toute analyse ulterieure.")

    header = ["timestamp", "elapsed_s"] + [f"{zid}_{zones[zid]['type']}_C" for zid in zone_ids]

    stop = {"flag": False}

    def handle_sigint(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, handle_sigint)

    t0 = time.time()
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        f.flush()

        print(f"\nLogging vers {args.out} toutes les {args.interval}s (Ctrl+C pour arreter)...")

        while not stop["flag"]:
            now = time.time()
            elapsed = now - t0
            row = [f"{now:.3f}", f"{elapsed:.3f}"]
            for zid in zone_ids:
                temp = read_temp_c(zones[zid]["temp_path"])
                row.append(f"{temp:.1f}" if temp is not None else "")
            writer.writerow(row)
            f.flush()

            if args.duration is not None and elapsed >= args.duration:
                break

            time.sleep(args.interval)

    print(f"\nTermine. {args.out} pret a etre trace (ex: plot_thermal_log.py).")


if __name__ == "__main__":
    main()