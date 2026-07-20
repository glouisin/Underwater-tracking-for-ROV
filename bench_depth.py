#!/usr/bin/env python3
"""
Benchmark d'un modèle de profondeur (TFLite) sur une vidéo de référence.

Deux modes d'échantillonnage du pixel de profondeur :

1. PIXEL FIXE (--pixel-x/--pixel-y) : suppose que l'objet reste exactement
   au même pixel à chaque palier -- fiable seulement si le recentrage
   manuel est précis, ce qui devient difficile à courte distance (l'effet
   de parallax amplifie toute erreur de positionnement physique en un
   grand déplacement pixel).

2. YOLO (--yolo-model) : détecte l'objet à chaque frame et échantillonne
   la profondeur au centroïde réel de la détection -- même logique que le
   pipeline de production (rov_vision_*.py), robuste au fait que l'objet
   n'est pas parfaitement recentré à la main à chaque palier.

Usage (mode pixel fixe) :
    python benchmark_depth_model.py \
        --model models/depth_anything_v2-tflite-float/depth_anything_v2.tflite \
        --model-name depth_anything_v2 \
        --video calib_video.mp4 \
        --calibration calibration.csv \
        --pixel-x 640 --pixel-y 360 \
        --use-htp \
        --output results_depth_anything_v2.csv

Usage (mode YOLO) :
    python benchmark_depth_model.py \
        --model models/depth_anything_v2-tflite-float/depth_anything_v2.tflite \
        --model-name depth_anything_v2 \
        --video calib_video.mp4 \
        --calibration calibration.csv \
        --yolo-model models/deepbox-tflite-float/yolov8n_saved_model/int8/yolov8n_full_integer_quant.tflite \
        --conf-thres 0.05 \
        --use-htp \
        --output results_depth_anything_v2.csv

Le fichier de calibration (CSV) associe des plages de frames à une distance
réelle connue (mesurée manuellement au ruban gradué) :
    frame_start,frame_end,distance_cm
    0,150,30
    151,300,50
    301,450,70
    ...

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

    sampling_group = p.add_argument_group("Échantillonnage du pixel de profondeur (choisir un mode)")
    sampling_group.add_argument("--pixel-x", type=int, help="Mode pixel fixe : coordonnée X (résolution native vidéo)")
    sampling_group.add_argument("--pixel-y", type=int, help="Mode pixel fixe : coordonnée Y (résolution native vidéo)")
    sampling_group.add_argument("--yolo-model", help="Mode YOLO : chemin vers le modèle YOLOv8n .tflite (int8 full-integer quant)")
    sampling_group.add_argument("--conf-thres", type=float, default=0.05, help="Seuil de confiance YOLO (mode YOLO uniquement, défaut: 0.05)")
    sampling_group.add_argument("--yolo-min-cy", type=int, default=0, help="Rejette les détections dont le centre y (pixel natif vidéo) est sous ce seuil (mode YOLO, filtre les fantômes -- même logique que MIN_CY_VALID en production)")
    sampling_group.add_argument(
        "--exclude-zone", action="append", default=[],
        metavar="x_min,y_min,x_max,y_max",
        help=(
            "Mode YOLO : exclut toute détection dont le centroïde (coordonnées natives vidéo) "
            "tombe dans ce rectangle. Utile pour un second objet parasite identifié dans le "
            "champ (ex: élément de décor dans un coin, détecté de façon stable par YOLO mais "
            "faussant l'échantillonnage de profondeur). Répéter l'option pour plusieurs zones. "
            "Exemple: --exclude-zone 0,380,120,480"
        )
    )

    p.add_argument("--use-htp", action="store_true", help="Utiliser le délégué QNN HTP (NPU) au lieu du CPU")
    p.add_argument("--delegate-path", default="/opt/qcom/qairt-new/qairt/2.48.0.260626/lib/aarch64-oe-linux-gcc11.2/libQnnTFLiteDelegate.so")
    p.add_argument("--adsp-library-path", default=(
        "/opt/qcom/qairt-new/qairt/2.48.0.260626/lib/hexagon-v73/unsigned;"
        "/opt/qcom/qirp-sdk/lib/aarch64-oe-linux-gcc11.2;"
        "/opt/qcom/qirp-sdk/lib/hexagon-v73/unsigned"
    ))
    p.add_argument("--z-max", type=float, default=1000.0, help="Z_MAX utilisé pour les modèles float non quantifiés (fallback normalisation)")
    p.add_argument(
        "--depth-invert", action="store_true",
        help=(
            "Inverse la convention de sortie du modèle (z_corrected = z_raw au lieu de "
            "Z_MAX - z_raw). Nécessaire pour Depth Anything V3, qui sort une profondeur "
            "directe (brut élevé = loin) contrairement à MiDaS/Depth Anything V2 qui sortent "
            "une disparité (brut élevé = proche). Vérifie le signe du Pearson affiché en fin "
            "de run : une corrélation positive avec la distance signale une convention "
            "inversée par rapport à MiDaS/V2, à corriger avec ce flag plutôt qu'en post-traitement."
        )
    )
    p.add_argument("--output", required=True, help="CSV de sortie")

    args = p.parse_args()

    use_fixed_pixel = args.pixel_x is not None and args.pixel_y is not None
    use_yolo = args.yolo_model is not None

    if use_fixed_pixel and use_yolo:
        p.error("Choisis un seul mode d'échantillonnage : --pixel-x/--pixel-y OU --yolo-model, pas les deux.")
    if not use_fixed_pixel and not use_yolo:
        p.error("Il faut spécifier soit --pixel-x/--pixel-y (mode pixel fixe), soit --yolo-model (mode détection automatique).")
    if use_fixed_pixel and (args.pixel_x is None or args.pixel_y is None):
        p.error("--pixel-x et --pixel-y doivent être fournis ensemble.")

    exclude_zones = []
    for zone_str in args.exclude_zone:
        try:
            x_min, y_min, x_max, y_max = (int(v) for v in zone_str.split(","))
        except ValueError:
            p.error(f"--exclude-zone invalide : '{zone_str}' -- format attendu x_min,y_min,x_max,y_max")
        exclude_zones.append((x_min, y_min, x_max, y_max))
    args.exclude_zones_parsed = exclude_zones

    return args


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


def in_any_exclude_zone(cx, cy, exclude_zones):
    """True si le point (cx, cy), en coordonnées natives vidéo, tombe dans
    au moins une des zones d'exclusion (x_min, y_min, x_max, y_max)."""
    for x_min, y_min, x_max, y_max in exclude_zones:
        if x_min <= cx <= x_max and y_min <= cy <= y_max:
            return True
    return False


def detect_best_box(yolo_interpreter, in_det_yolo, out_det_yolo, frame_rgb, frame_w, frame_h,
                     conf_thres, min_cy, exclude_zones=None):
    """Fait tourner YOLO sur une frame et retourne (cx, cy, confidence) du
    meilleur candidat (score le plus élevé après filtrage), ou None si rien
    ne passe le seuil. Même logique de dequantization/parsing que le
    pipeline de production, simplifiée (pas de tracker -- juste la
    meilleure détection frame par frame, suffisant pour un benchmark où
    l'objet est seul et connu dans le champ).

    exclude_zones : liste de rectangles (x_min, y_min, x_max, y_max) en
    coordonnées natives vidéo -- toute détection dont le centroïde tombe
    dans une de ces zones est ignorée avant la sélection du meilleur score.
    Utile pour un second objet parasite identifié dans la scène (cf. le cas
    du coin bas-gauche détecté de façon stable par YOLO mais correspondant
    à un objet différent de la cible réelle)."""
    if exclude_zones is None:
        exclude_zones = []

    in_h_yolo, in_w_yolo = in_det_yolo[0]["shape"][1:3]
    in_scale_yolo, in_zp_yolo = in_det_yolo[0]["quantization"]
    out_scale_yolo, out_zp_yolo = out_det_yolo[0]["quantization"]

    resized = cv2.resize(frame_rgb, (in_w_yolo, in_h_yolo))
    buf = resized.astype(np.float32) / (255.0 * in_scale_yolo) + in_zp_yolo
    np.clip(buf, -128, 127, out=buf)
    input_tensor = np.expand_dims(buf.astype(np.int8), axis=0)

    yolo_interpreter.set_tensor(in_det_yolo[0]["index"], input_tensor)
    yolo_interpreter.invoke()
    output_raw = yolo_interpreter.get_tensor(out_det_yolo[0]["index"])[0]

    output = (output_raw.astype(np.float32) - out_zp_yolo) * out_scale_yolo
    output = output.transpose()  # [84, 8400] -> [8400, 84]

    boxes = output[:, :4]
    class_scores = output[:, 4:]
    scores = np.max(class_scores, axis=1)

    mask = scores > conf_thres
    if not np.any(mask):
        return None

    boxes = boxes[mask]
    scores = scores[mask]

    cx = boxes[:, 0] * frame_w
    cy = boxes[:, 1] * frame_h

    valid = cy >= min_cy
    if not np.any(valid):
        return None
    cx, cy, scores = cx[valid], cy[valid], scores[valid]

    if exclude_zones:
        keep = np.array([not in_any_exclude_zone(x, y, exclude_zones) for x, y in zip(cx, cy)])
        if not np.any(keep):
            return None
        cx, cy, scores = cx[keep], cy[keep], scores[keep]

    best_idx = int(np.argmax(scores))
    return float(cx[best_idx]), float(cy[best_idx]), float(scores[best_idx])


def main():
    args = parse_args()
    calibration_ranges = load_calibration(args.calibration)
    use_yolo = args.yolo_model is not None

    interpreter = build_interpreter(args.model, args.use_htp, args.delegate_path, args.adsp_library_path)
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

    in_h, in_w = in_det[0]["shape"][1], in_det[0]["shape"][2]
    is_quantized = in_det[0]["dtype"] == np.uint8

    in_scale, in_zp = (in_det[0]["quantization"] if is_quantized else (None, None))
    out_scale, out_zp = (out_det[0]["quantization"] if is_quantized else (None, None))

    yolo_interpreter = None
    in_det_yolo = out_det_yolo = None
    if use_yolo:
        print(f"[{args.model_name}] Mode YOLO actif : {args.yolo_model}")
        if args.exclude_zones_parsed:
            for zone in args.exclude_zones_parsed:
                print(f"[{args.model_name}] Zone d'exclusion active : x={zone[0]}-{zone[2]}, y={zone[1]}-{zone[3]}")
        yolo_interpreter = build_interpreter(args.yolo_model, args.use_htp, args.delegate_path, args.adsp_library_path)
        in_det_yolo = yolo_interpreter.get_input_details()
        out_det_yolo = yolo_interpreter.get_output_details()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {args.video}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_x = in_w / frame_w
    scale_y = in_h / frame_h

    if not use_yolo:
        px_model = int(args.pixel_x * scale_x)
        py_model = int(args.pixel_y * scale_y)
        px_model = min(max(px_model, 0), in_w - 1)
        py_model = min(max(py_model, 0), in_h - 1)

    rows = []
    frame_idx = 0
    warmup_done = False
    skipped_no_detection = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dist_true = distance_for_frame(frame_idx, calibration_ranges)
        if dist_true is None:
            frame_idx += 1
            continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detection_conf = None
        if use_yolo:
            detection = detect_best_box(
                yolo_interpreter, in_det_yolo, out_det_yolo, img, frame_w, frame_h,
                args.conf_thres, args.yolo_min_cy, args.exclude_zones_parsed
            )
            if detection is None:
                skipped_no_detection += 1
                frame_idx += 1
                continue
            cx_native, cy_native, detection_conf = detection
            px_model = int(np.clip(cx_native * scale_x, 0, in_w - 1))
            py_model = int(np.clip(cy_native * scale_y, 0, in_h - 1))

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

        # z_corrected suit par défaut la convention disparité (brut élevé = proche,
        # cf. MiDaS/Depth Anything V2). --depth-invert inverse cette convention pour
        # les modèles qui sortent une profondeur directe (brut élevé = loin), comme
        # Depth Anything V3 -- cf. doc officielle du modèle.
        if args.depth_invert:
            z_corrected = z_raw
        else:
            z_corrected = z_max_local - z_raw

        row = {
            "frame_id": frame_idx,
            "model_name": args.model_name,
            "distance_cm_reelle": dist_true,
            "z_raw": z_raw,
            "z_corrected": z_corrected,
            "latency_ms": latency_ms,
        }
        if use_yolo:
            row["yolo_confidence"] = detection_conf
            row["pixel_x_used"] = px_model
            row["pixel_y_used"] = py_model
        rows.append(row)

        frame_idx += 1

    cap.release()

    fieldnames = ["frame_id", "model_name", "distance_cm_reelle", "z_raw", "z_corrected", "latency_ms"]
    if use_yolo:
        fieldnames += ["yolo_confidence", "pixel_x_used", "pixel_y_used"]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[{args.model_name}] {len(rows)} frames analysées -> {args.output}")
    if use_yolo and skipped_no_detection > 0:
        print(f"[{args.model_name}] {skipped_no_detection} frames ignorées (aucune détection YOLO au-dessus de --conf-thres={args.conf_thres})")

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

        # Avertissement automatique : par convention (disparité), z_corrected doit
        # décroître avec la distance, donc Pearson attendu négatif. Un Pearson
        # positif signale probablement une convention de sortie inversée pour ce
        # modèle (cf. Depth Anything V3, qui sort une profondeur directe et non
        # une disparité) -- à corriger avec --depth-invert plutôt qu'en post-traitement.
        if corr > 0.3 and not args.depth_invert:
            print(f"[{args.model_name}] ATTENTION : corrélation positive ({corr:.3f}) alors qu'une "
                  f"corrélation négative est attendue (convention disparité). Ce modèle sort "
                  f"peut-être une profondeur directe plutôt qu'une disparité -- relance avec "
                  f"--depth-invert et compare le nouveau signe du Pearson.")
        elif corr < -0.3 and args.depth_invert:
            print(f"[{args.model_name}] ATTENTION : --depth-invert est actif mais la corrélation "
                  f"est déjà négative ({corr:.3f}) -- ce modèle suit peut-être déjà la convention "
                  f"disparité standard, --depth-invert pourrait être inutile ou incorrect ici.")


if __name__ == "__main__":
    main()