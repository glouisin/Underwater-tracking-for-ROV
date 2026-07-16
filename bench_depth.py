#!/usr/bin/env python3
"""
Benchmark d'un modèle de profondeur (TFLite) sur une vidéo de référence.

Usage :
    python benchmark_depth_model.py \
        --model models/depth_anything_v2-tflite-float/depth_anything_v2.tflite \
        --model-name depth_anything_v2 \
        --video calib_video.mp4 \
        --calibration calibration.csv \
        --pixel-x 640 --pixel-y 360 \
        --use-htp \
        --output results_depth_anything_v2.csv

Le fichier de calibration (CSV) associe des plages de frames à une distance
réelle connue (mesurée manuellement au ruban gradué) :
    frame_start,frame_end,distance_cm
    0,150,30
    151,300,50
    301,450,70
    ...

Le script échantillonne systématiquement la profondeur à une coordonnée
pixel FIXE (--pixel-x/--pixel-y), pas via une détection YOLO — l'objectif
est d'isoler la précision du modèle de profondeur du bruit de détection.
L'opérateur doit donc avoir centré/positionné l'objet cible à cette
coordonnée pixel manuellement pendant l'enregistrement de la vidéo de
référence.

Chaque modèle testé (MiDaS, Depth Anything V2, V3...) se lance séparément
avec ce script sur la MÊME vidéo de référence, produisant un CSV comparable.
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np

try:
    import ai_edge_litert.interpreter as tflite
    _USING_TF_FALLBACK = False
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        _USING_TF_FALLBACK = False
    except ImportError:
        import tensorflow.lite as tflite
        _USING_TF_FALLBACK = True


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark d'un modèle de profondeur monoculaire")
    p.add_argument("--model", required=True, help="Chemin vers le modèle .tflite")
    p.add_argument("--model-name", required=True, help="Nom court utilisé dans les résultats (ex: midas, depth_anything_v2)")
    p.add_argument("--video", required=True, help="Vidéo de référence (même vidéo pour tous les modèles comparés)")
    p.add_argument("--calibration", required=True, help="CSV frame_start,frame_end,distance_cm")
    p.add_argument("--pixel-x", type=int, required=True, help="Coordonnée X (résolution native vidéo) où échantillonner la profondeur")
    p.add_argument("--pixel-y", type=int, required=True, help="Coordonnée Y (résolution native vidéo) où échantillonner la profondeur")
    p.add_argument("--use-htp", action="store_true", help="Utiliser le délégué QNN HTP (NPU) au lieu du CPU")
    p.add_argument("--delegate-path", default="/opt/qcom/qirp-sdk/lib/aarch64-oe-linux-gcc11.2/libQnnTFLiteDelegate.so")
    p.add_argument("--adsp-library-path", default="/opt/qcom/qairt-new/qairt/2.48.0.260626/lib/hexagon-v75/unsigned")
    p.add_argument("--z-max", type=float, default=1000.0, help="Z_MAX utilisé pour les modèles float non quantifiés (fallback normalisation)")
    p.add_argument("--output", required=True, help="CSV de sortie")
    return p.parse_args()


def load_calibration(path):
    """Retourne une liste de (frame_start, frame_end, distance_cm)."""
    ranges = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranges.append((int(row["frame_start"]), int(row["frame_end"]), float(row["distance_cm"])))
    return ranges


def distance_for_frame(frame_idx, calibration_ranges):
    for start, end, dist in calibration_ranges:
        if start <= frame_idx <= end:
            return dist
    return None  # frame hors plage de calibration connue -> ignorée


def build_interpreter(model_path, use_htp, delegate_path, adsp_library_path):
    # ADSP_LIBRARY_PATH doit être fixé AVANT toute init de délégué NPU
    if use_htp:
        os.environ["ADSP_LIBRARY_PATH"] = adsp_library_path

    if use_htp:
        if _USING_TF_FALLBACK:
            # tensorflow.lite n'expose pas load_delegate directement (contrairement
            # à ai_edge_litert/tflite_runtime) -- c'est sous tf.lite.experimental,
            # avec la même signature (chemin .so + options dict).
            delegate = tflite.experimental.load_delegate(delegate_path, options={"backend_type": "htp", "htp_performance_mode": "1"})
        else:
            delegate = tflite.load_delegate(delegate_path, options={"backend_type": "htp", "htp_performance_mode": "1"})
        interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[delegate])
    else:
        interpreter = tflite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()
    return interpreter


def main():
    args = parse_args()
    calibration_ranges = load_calibration(args.calibration)

    interpreter = build_interpreter(args.model, args.use_htp, args.delegate_path, args.adsp_library_path)
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

    in_h, in_w = in_det[0]["shape"][1], in_det[0]["shape"][2]
    is_quantized = in_det[0]["dtype"] == np.uint8

    in_scale, in_zp = (in_det[0]["quantization"] if is_quantized else (None, None))
    out_scale, out_zp = (out_det[0]["quantization"] if is_quantized else (None, None))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {args.video}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_x = in_w / frame_w
    scale_y = in_h / frame_h
    px_model = int(args.pixel_x * scale_x)
    py_model = int(args.pixel_y * scale_y)
    px_model = min(max(px_model, 0), in_w - 1)
    py_model = min(max(py_model, 0), in_h - 1)

    rows = []
    frame_idx = 0
    warmup_done = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dist_true = distance_for_frame(frame_idx, calibration_ranges)
        if dist_true is None:
            frame_idx += 1
            continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (in_w, in_h))

        if is_quantized:
            input_tensor = (img_resized.astype(np.float32) / in_scale + in_zp).astype(np.uint8)
        else:
            input_tensor = (img_resized.astype(np.float32) / 255.0)

        input_tensor = np.expand_dims(input_tensor, axis=0)

        t0 = time.perf_counter()
        interpreter.set_tensor(in_det[0]["index"], input_tensor)
        interpreter.invoke()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not warmup_done:
            print(f"[{args.model_name}] Première inférence : {latency_ms:.1f} ms (peut inclure un coût de warmup)")
            warmup_done = True

        output_raw = interpreter.get_tensor(out_det[0]["index"])[0]

        if is_quantized:
            depth_map = (output_raw.astype(np.float32) - out_zp) * out_scale
            z_raw = float(depth_map[py_model, px_model])
            z_max_local = 255.0 * out_scale
        else:
            raw = np.squeeze(output_raw).astype(np.float32)
            d_min, d_max = raw.min(), raw.max()
            depth_map = (raw - d_min) / (d_max - d_min + 1e-6) * args.z_max
            z_raw = float(depth_map[py_model, px_model])
            z_max_local = args.z_max

        z_corrected = z_max_local - z_raw  # convention disparité : plus grand brut = plus proche

        rows.append({
            "frame_id": frame_idx,
            "model_name": args.model_name,
            "distance_cm_reelle": dist_true,
            "z_raw": z_raw,
            "z_corrected": z_corrected,
            "latency_ms": latency_ms,
        })

        frame_idx += 1

    cap.release()

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "model_name", "distance_cm_reelle", "z_raw", "z_corrected", "latency_ms"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[{args.model_name}] {len(rows)} frames analysées -> {args.output}")

    # ---- Résumé rapide en console (analyse fine à faire séparément, cf. analyze_benchmark.py) ----
    if rows:
        import statistics
        latencies = [r["latency_ms"] for r in rows]
        print(f"[{args.model_name}] Latence moyenne : {statistics.mean(latencies):.2f} ms  "
              f"(p95={sorted(latencies)[int(0.95 * len(latencies))]:.2f} ms)")

        dists = [r["distance_cm_reelle"] for r in rows]
        zs = [r["z_corrected"] for r in rows]
        corr = np.corrcoef(dists, zs)[0, 1]
        print(f"[{args.model_name}] Corrélation z_corrected vs distance réelle : {corr:.3f}")


if __name__ == "__main__":
    main()