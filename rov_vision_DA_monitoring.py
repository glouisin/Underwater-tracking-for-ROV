# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
# ROV Vision Pipeline — INSTRUMENTED BUILD
#
# Identical to rov_vision_FINAL.py except for measurement instrumentation.
# Every change is marked with  # [MON]  so you can diff against your current
# file and port the edits if your version has drifted.
#
# Behavioural fixes included (see integration_patch.md §0):
#   [MON-FIX-1] depth invoke was timed OUTSIDE the inference timer
#   [MON-FIX-2] last_seq was never updated -> duplicate-frame guard was dead
#   [MON-FIX-3] throttle / DEBUG_DEPTH / HTP now env-driven for benchmark runs
#   [MON-FIX-4] 'depth_map' in dir() was always True
#   [MON-FIX-5] clean shutdown so summary.json is always written
# =============================================================================

#!/usr/bin/env python3

import os

os.environ["ADSP_LIBRARY_PATH"] = (
    "/opt/qcom/qairt-new/qairt/2.48.0.260626/lib/hexagon-v73/unsigned;"
    "/opt/qcom/qirp-sdk/lib/aarch64-oe-linux-gcc11.2;"
    "/opt/qcom/qirp-sdk/lib/hexagon-v73/unsigned;"
    + os.environ.get("ADSP_LIBRARY_PATH", "")
)
import sys
import math
import json
import signal
import socket
import time
import cv2
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import ai_edge_litert.interpreter as tflite
import threading

from monitoring import RunMonitor          # [MON]

Z_DIAG = False
if Z_DIAG:
    from z_diagnostic import sample_depth, ZDiagnosticLogger
Gst.init(None)


# -------------------- [MON] Benchmark switches --------------------
# Env-driven so that a run is fully described by its shell command, and so the
# same binary produces every ablation of the paper without editing source.
RUN_TAG       = os.environ.get("RUN_TAG", "dev")
NO_THROTTLE   = os.environ.get("NO_THROTTLE", "0") == "1"
FORCE_DEPTH   = os.environ.get("FORCE_DEPTH", "0") == "1"
DEPTH_USE_HTP = os.environ.get("DEPTH_USE_HTP", "1") == "1"
DEBUG_DEPTH   = os.environ.get("DEBUG_DEPTH", "1") == "0"


# -------------------- Auto IP connection --------------------
def get_ssh_client_ip():
    conn = os.environ.get("SSH_CONNECTION")
    if conn:
        return conn.split()[0]
    client = os.environ.get("SSH_CLIENT")
    if client:
        return client.split()[0]
    return None


# -------------------- class for Camera thread --------------------
class CameraStream:
    def __init__(self, src=2):
        self.seq = 0
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.frame   = None
        self.ok      = False
        self.lock    = threading.Lock()
        self.running = True
        self.ok, self.frame = self.cap.read()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok    = ok
                self.frame = frame
                self.seq += 1

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None, self.seq
            return self.ok, self.frame.copy(), self.seq

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()


# -------------------- Parameters --------------------
MODEL_PATH_YOLO  = "/root/ai-rov/src/models/deepbox-tflite-float/yolov8n_saved_model/int8/yolov8n_full_integer_quant.tflite"
MODEL_PATH_DEPTH = "/root/ai-rov/src/models/depth_anything_v3-tflite-float/depth_anything_v3.tflite"
DELEGATE_PATH    = "/opt/qcom/qairt-new/qairt/2.48.0.260626/lib/aarch64-oe-linux-gcc11.2/libQnnTFLiteDelegate.so"

MAC_IP           = get_ssh_client_ip()
UDP_IP           = "127.0.0.1"
UDP_PORT         = 17002
SELECT_PORT      = 17001
LIST_PORT        = 17003
VIDEO_IN         = "13759780_2160_3840_60fps.mp4"
TEST_VIDEO       = os.environ.get("TEST_VIDEO", "0") == "1"      # [MON]

FRAME_W       = 1280
FRAME_H       = 720
FPS_OUT       = 15
CONF_THRES    = 0.2
MIN_CY_VALID  = 60
NMS_IOU_THRES = 0.1

ALPHA_XY        = 0.15
ALPHA_Z         = 0.05
ALPHA_CXY_DEPTH = 0.25

DEBUG_EXCLUSION_ZONE = False

INFER_FPS   = 15
MIN_LOOP_DT = 1.0 / INFER_FPS

MAX_BOX_AREA_RATIO = 0.8
DEPTH_INVERT       = False

# -------------------- Fusion profondeur --------------------
# Fusion du signal monoculaire avec l'aire de la boîte englobante, DANS
# L'ESPACE DISPARITE (valeur haute = proche), avant l'inversion :
#
#     z_bb    = z_raw_desired * sqrt(area / area_desired)
#     z_fused = a * z_model + (1 - a) * z_bb
#
# Base physique : la projection perspective donne A ∝ 1/d², la disparité
# monoculaire varie en 1/d, donc disparité ∝ sqrt(A). Les deux signaux vivent
# bien dans le même espace, la combinaison linéaire est licite. Fusionner
# APRES l'inversion (z_corrected = Z_MAX - z) casserait cette relation.
#
# FUSION_ALPHA pilote l'ablation de l'article sans toucher au code :
#   1.0 -> modèle seul (référence)   0.0 -> boîte seule (méthode Safa et al.)
FUSION_ALPHA   = float(os.environ.get("FUSION_ALPHA", "0.6"))
ALPHA_AREA     = 0.20    # EMA sur l'aire : sans lissage, z_bb hérite du jitter YOLO
BORDER_MARGIN_PX = 8     # boîte tronquée au bord -> aire fausse -> repli modèle seul

# Arrondi du setpoint à la centaine supérieure (marge de sécurité de l'ancienne
# version). 0.0 = désactivé, ce qui est requis pour des mesures propres :
# l'arrondi injecte un offset constant que le PID passe son temps à rattraper,
# et ez ne part pas de zéro à la sélection. Remettre 100.0 pour l'ancien
# comportement.
Z_DESIRED_ROUND_TO = float(os.environ.get("Z_DESIRED_ROUND_TO", "0"))

# -------------------- Paramètres tracker --------------------
# Déclarés ici (et non plus après update_tracks) car CONFIG les référence.
MAX_DIST_PIXELS = 450     # distance max entre centroïdes pour un même objet
TRACK_TTL_S     = 2.0     # [FIX-B] expiration en SECONDES (indépendante du FPS)

# -------------------- Object detection exclusion --------------------
CLAW_EXCLUSION_ZONE_PX = {
    "x_min": 448, "x_max": 832,
    "y_min": 576, "y_max": 720,
}


def is_in_claw_zone(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return (CLAW_EXCLUSION_ZONE_PX["x_min"] <= cx <= CLAW_EXCLUSION_ZONE_PX["x_max"] and
            CLAW_EXCLUSION_ZONE_PX["y_min"] <= cy <= CLAW_EXCLUSION_ZONE_PX["y_max"])


def is_box_too_big(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h
    return (box_area / frame_area) > MAX_BOX_AREA_RATIO


# -------------------- Load Delegate --------------------
delegate_options = {'backend_type': 'htp', 'htp_performance_mode': '1'}
delegate = tflite.load_delegate(DELEGATE_PATH, options=delegate_options)

# -------------------- Load YOLO --------------------
yolo_interpreter = tflite.Interpreter(
    model_path=MODEL_PATH_YOLO,
    experimental_delegates=[delegate]
)
yolo_interpreter.allocate_tensors()

in_det_yolo  = yolo_interpreter.get_input_details()
out_det_yolo = yolo_interpreter.get_output_details()

in_h_yolo, in_w_yolo        = in_det_yolo[0]["shape"][1:3]
in_scale_yolo, in_zp_yolo   = in_det_yolo[0]["quantization"]
out_scale_yolo, out_zp_yolo = out_det_yolo[0]["quantization"]

print(f"YOLO  — entrée : {in_w_yolo}x{in_h_yolo}  dtype: {in_det_yolo[0]['dtype']}")

# -------------------- Load Depth model --------------------
delegate_options_depth = None
if DEPTH_USE_HTP:
    delegate_options_depth = {'backend_type': 'htp', 'htp_performance_mode': '1'}
    delegate_depth = tflite.load_delegate(DELEGATE_PATH, options=delegate_options_depth)
    depth_interpreter = tflite.Interpreter(
        model_path=MODEL_PATH_DEPTH,
        experimental_delegates=[delegate_depth]
    )
else:
    depth_interpreter = tflite.Interpreter(model_path=MODEL_PATH_DEPTH)
depth_interpreter.allocate_tensors()

in_det_depth  = depth_interpreter.get_input_details()
out_det_depth = depth_interpreter.get_output_details()

in_h_depth, in_w_depth        = in_det_depth[0]["shape"][1:3]
in_scale_depth, in_zp_depth   = in_det_depth[0]["quantization"]
out_scale_depth, out_zp_depth = out_det_depth[0]["quantization"]

print(f"Depth — entrée : {in_w_depth}x{in_h_depth}  dtype: {in_det_depth[0]['dtype']}")

DEPTH_INPUT_DTYPE  = in_det_depth[0]['dtype']
DEPTH_OUTPUT_DTYPE = out_det_depth[0]['dtype']
DEPTH_IS_QUANTIZED = DEPTH_INPUT_DTYPE in (np.uint8, np.int8) and in_scale_depth != 0

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

scale_x_depth = in_w_depth / FRAME_W
scale_y_depth = in_h_depth / FRAME_H

Z_MAX_FLOAT = 1000.0
Z_MAX = (255.0 * out_scale_depth) if DEPTH_IS_QUANTIZED else Z_MAX_FLOAT


# ==================== [MON] Monitor construction ====================
# Placed after the interpreters so model metadata can be registered, and
# before the GStreamer pipeline so startup cost is not attributed to frames.
CONFIG = dict(
    FRAME_W=FRAME_W, FRAME_H=FRAME_H, FPS_OUT=FPS_OUT, INFER_FPS=INFER_FPS,
    CONF_THRES=CONF_THRES, NMS_IOU_THRES=NMS_IOU_THRES,
    ALPHA_XY=ALPHA_XY, ALPHA_Z=ALPHA_Z, ALPHA_CXY_DEPTH=ALPHA_CXY_DEPTH,
    MIN_CY_VALID=MIN_CY_VALID, MAX_BOX_AREA_RATIO=MAX_BOX_AREA_RATIO,
    MAX_DIST_PIXELS=MAX_DIST_PIXELS, TRACK_TTL_S=TRACK_TTL_S,
    FUSION_ALPHA=FUSION_ALPHA, ALPHA_AREA=ALPHA_AREA,
    BORDER_MARGIN_PX=BORDER_MARGIN_PX, Z_DESIRED_ROUND_TO=Z_DESIRED_ROUND_TO,
    DEPTH_INVERT=DEPTH_INVERT, DEPTH_USE_HTP=DEPTH_USE_HTP,
    DEBUG_DEPTH=DEBUG_DEPTH, NO_THROTTLE=NO_THROTTLE, FORCE_DEPTH=FORCE_DEPTH,
    TEST_VIDEO=TEST_VIDEO, Z_MAX=float(Z_MAX),
    DEPTH_IS_QUANTIZED=bool(DEPTH_IS_QUANTIZED),
)
mon = RunMonitor(run_tag=RUN_TAG, config=CONFIG,
                 extra_env=["ADSP_LIBRARY_PATH", "LD_LIBRARY_PATH"])
mon.register_model("yolov8n", yolo_interpreter, MODEL_PATH_YOLO, delegate_options)
mon.register_model("depth", depth_interpreter, MODEL_PATH_DEPTH,
                   delegate_options_depth if DEPTH_USE_HTP else "cpu")
# ====================================================================

# -------------------- GStreamer Pipeline --------------------
pipeline = Gst.parse_launch(
    f'appsrc name=src '
    f'is-live=true block=true format=time '
    f'caps=video/x-raw,format=BGR,width={FRAME_W},height={FRAME_H},framerate={FPS_OUT}/1 '
    '! queue max-size-buffers=3 leaky=downstream '
    '! videoconvert '
    '! qtic2venc control-rate=1 target-bitrate=1500000 '
    '! h264parse config-interval=1 '
    '! mpegtsmux '
    f'! udpsink host={MAC_IP} port=5000 buffer-size=4194304'
)
appsrc = pipeline.get_by_name('src')
pipeline.set_state(Gst.State.PLAYING)

# -------------------- Video Input --------------------
if not TEST_VIDEO:
    cap = CameraStream(src=2)
else:
    cap = cv2.VideoCapture(VIDEO_IN)

frame_rs           = np.empty((FRAME_H, FRAME_W, 3), np.uint8)
input_tensor_yolo  = np.empty((1, in_h_yolo,  in_w_yolo,  3), np.int8)
input_tensor_depth = np.empty((1, in_h_depth, in_w_depth, 3), DEPTH_INPUT_DTYPE)

YOLO_SCALE  = 1.0 / (255.0 * in_scale_yolo)
DEPTH_SCALE = 1.0 / (255.0 * in_scale_depth) if DEPTH_IS_QUANTIZED else None
buf_yolo_f32  = np.empty((in_h_yolo,  in_w_yolo,  3), np.float32)
buf_depth_f32 = np.empty((in_h_depth, in_w_depth, 3), np.float32)

# -------------------- Stats --------------------
frame_cnt    = 0
t_total_inf  = 0.0
t_start_loop = time.time()

tracked_objects = {}
next_id = 0


def update_tracks(current_boxes, frame_cnt):
    """Tracker par distance euclidienne entre centroïdes.

    Assignation gloutonne au plus proche voisin. Un track ne peut être
    apparié qu'à UNE détection par frame : d'où used_tracks, alimenté aussi
    bien lors d'un appariement que lors d'une création. Sans cela, plusieurs
    détections reçoivent le même identifiant.
    """
    global tracked_objects, next_id
    now = time.time()
    assignments = {}
    used_tracks = set()

    centroids = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in current_boxes]

    for idx, (cx, cy) in enumerate(centroids):
        best_id   = None
        best_dist = MAX_DIST_PIXELS

        for tid, tobj in tracked_objects.items():
            if tid in used_tracks:
                continue
            dx   = cx - tobj['cx']
            dy   = cy - tobj['cy']
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best_id   = tid

        if best_id is not None:
            assignments[idx] = best_id
            t = tracked_objects[best_id]
            t['cx']        = cx
            t['cy']        = cy
            t['bbox']      = current_boxes[idx]
            t['last_seen'] = frame_cnt
            t['last_t']    = now                          # [FIX-B]
            t['x_f'] = (ALPHA_XY * (cx - FRAME_W / 2)
                        + (1 - ALPHA_XY) * t.get('x_f', cx - FRAME_W / 2))
            t['y_f'] = (ALPHA_XY * (FRAME_H / 2 - cy)
                        + (1 - ALPHA_XY) * t.get('y_f', FRAME_H / 2 - cy))
            used_tracks.add(best_id)                      # [FIX-A] track apparié
        else:
            assignments[idx] = next_id
            tracked_objects[next_id] = {
                'cx': cx, 'cy': cy,
                'bbox': current_boxes[idx],
                'last_seen': frame_cnt,
                'last_t': now,                            # [FIX-B]
                'created_t': now,
                'x_f': cx - FRAME_W / 2,
                'y_f': FRAME_H / 2 - cy,
                'z_f': None,
            }
            used_tracks.add(next_id)                      # [FIX-A] track créé
            next_id += 1

    # [FIX-B] expiration en temps réel : le comportement du tracker ne dépend
    # plus du FPS, donc reste comparable entre les runs de benchmark.
    to_delete = [tid for tid, tobj in tracked_objects.items()
                 if now - tobj.get('last_t', now) > TRACK_TTL_S]
    for tid in to_delete:
        del tracked_objects[tid]

    return assignments


# -------------------- UDP Sockets --------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sel_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sel_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sel_sock.bind(("0.0.0.0", SELECT_PORT))
sel_sock.setblocking(False)

selected_id = None
z_desired = None
z_desired_pending = False
# Ancres de fusion, capturées à la première mesure réelle après sélection
z_raw_desired = None      # disparité du modèle au setpoint
area_desired  = None      # aire de boîte lissée au setpoint


def poll_selection():
    global selected_id, z_desired, z_desired_pending
    global z_raw_desired, area_desired
    while True:
        try:
            data, _ = sel_sock.recvfrom(64)
        except BlockingIOError:
            break
        msg = data.decode(errors="ignore").strip()
        if msg == "clear":
            selected_id = None
            z_desired = None
            z_desired_pending = False
            z_raw_desired = None
            area_desired = None
        elif msg.startswith("select,"):
            try:
                selected_id = int(msg.split(",")[1])
                z_desired = None
                z_desired_pending = True
                z_raw_desired = None
                area_desired = None
            except (ValueError, IndexError):
                pass


if Z_DIAG:
    zlog = ZDiagnosticLogger("z_diag.csv")


# -------------------- [MON-FIX-5] Clean shutdown --------------------
def _shutdown(signum=None, frame=None):
    """Guarantees summary.json is written even on Ctrl-C / SIGTERM."""
    try:
        mon.close()
    except Exception as e:
        print(f"[MON] close failed: {e}")
    try:
        if Z_DIAG:
            zlog.stop()
        appsrc.emit('end-of-stream')
        pipeline.set_state(Gst.State.NULL)
        if not TEST_VIDEO:
            cap.stop()
    except Exception:
        pass
    sys.exit(0)


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# -------------------- Main Loop --------------------
last_seq = -1
loop_cnt = 0
while True:
    t_start_iter = time.time()
    loop_cnt += 1

    if cap.seq == last_seq:
        mon.mark_loop(False)                     # [MON] idle spin, no new frame
        time.sleep(0.002)
        continue
    mon.mark_loop(True)                          # [MON]

    with mon.stage("capture_wait"):              # [MON]
        ok, frame, seq = cap.read()
    if not ok or frame is None:
        continue
    last_seq = seq                               # [MON-FIX-2]

    frame_cnt += 1

    # [MON] reset per-frame application fields so frame_row never sees stale
    # values from a previous frame (and never raises NameError).
    dx = dy = ez = None
    z_raw = None                                 # disparité modèle (cible)
    z_bb = z_fused = None                        # [FUSION] signaux séparés
    area_sel = None
    fusion_fallback = 0
    n_raw = n_conf = n_after_cy = n_after_claw = n_after_big = n_final = 0
    n_dup_ids = 0                                # [MON]

    poll_selection()
    list_data = []

    no_x = FRAME_W / 2
    no_y = FRAME_H / 2

    # ---------- Preprocessing ----------
    with mon.stage("preproc_yolo"):              # [MON]
        cv2.resize(frame, (FRAME_W, FRAME_H), dst=frame_rs)
        resized_yolo = cv2.resize(frame_rs, (in_w_yolo, in_h_yolo))
        buf_yolo_f32[:] = resized_yolo
        buf_yolo_f32   *= YOLO_SCALE
        buf_yolo_f32   += in_zp_yolo
        np.clip(buf_yolo_f32, -128, 127, out=buf_yolo_f32)
        input_tensor_yolo[0] = buf_yolo_f32.astype(np.int8)

    run_depth = (selected_id is not None) or FORCE_DEPTH   # [MON]

    if run_depth:
        with mon.stage("preproc_depth"):         # [MON]
            resized_depth = cv2.resize(frame_rs, (in_w_depth, in_h_depth))
            resized_rgb   = cv2.cvtColor(resized_depth, cv2.COLOR_BGR2RGB)
            if DEPTH_IS_QUANTIZED:
                buf_depth_f32[:] = resized_rgb
                buf_depth_f32   *= DEPTH_SCALE
                buf_depth_f32   += in_zp_depth
                np.clip(buf_depth_f32, 0, 255, out=buf_depth_f32)
                input_tensor_depth[0] = buf_depth_f32.astype(np.uint8)
            else:
                buf_depth_f32[:] = resized_rgb
                buf_depth_f32   *= (1.0 / 255.0)
                buf_depth_f32   -= IMAGENET_MEAN
                buf_depth_f32   /= IMAGENET_STD
                input_tensor_depth[0] = buf_depth_f32.astype(DEPTH_INPUT_DTYPE)

    # ---------- Inference ----------
    # [MON-FIX-1] the depth invoke is now inside a timed stage. In the original
    # it ran before t0, so it was silently excluded from every latency figure.
    if run_depth:
        depth_interpreter.set_tensor(in_det_depth[0]['index'], input_tensor_depth)
        with mon.stage("depth_invoke"):          # [MON]
            depth_interpreter.invoke()

    yolo_interpreter.set_tensor(in_det_yolo[0]['index'], input_tensor_yolo)
    t0 = time.time()
    with mon.stage("yolo_invoke"):               # [MON]
        yolo_interpreter.invoke()
    t_inf = time.time() - t0
    t_total_inf += t_inf

    # ---------- Postprocessing ----------
    with mon.stage("postproc_yolo"):             # [MON]
        depth_map = None
        output_raw_yolo = yolo_interpreter.get_tensor(out_det_yolo[0]['index'])[0]
        if run_depth:
            output_raw_depth = depth_interpreter.get_tensor(out_det_depth[0]['index'])[0]
            if DEPTH_IS_QUANTIZED:
                depth_map = (output_raw_depth.astype(np.float32) - out_zp_depth) * out_scale_depth
            else:
                raw = np.squeeze(output_raw_depth).astype(np.float32)
                d_min, d_max = raw.min(), raw.max()
                depth_map = (raw - d_min) / (d_max - d_min + 1e-6) * Z_MAX_FLOAT
            if DEPTH_INVERT:
                depth_map = Z_MAX - depth_map

        output = (output_raw_yolo.astype(np.float32) - out_zp_yolo) * out_scale_yolo
        output = output.transpose()

        boxes        = output[:, :4]
        class_scores = output[:, 4:]
        scores       = np.max(class_scores, axis=1)
        classes      = np.argmax(class_scores, axis=1)
        n_raw = int(len(boxes))                  # [MON]

    # ---------- Depth visualisation ----------
    with mon.stage("render"):                    # [MON]
        if DEBUG_DEPTH and depth_map is not None:
            depth_norm    = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_color   = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)
            depth_display = cv2.resize(depth_color, (FRAME_W, FRAME_H))
            output_frame  = cv2.addWeighted(frame_rs, 0.5, depth_display, 0.5, 0)
        else:
            output_frame = frame_rs

    # ---------- Stats every 30 frames ----------
    if frame_cnt % 30 == 0:
        elapsed_loop = time.time() - t_start_loop
        # [MON-FIX-4] 'depth_map' in dir() was always True
        depth_info = (f"{depth_map.min():.3f} / {depth_map.max():.3f}"
                      if depth_map is not None else "N/A")
        print(f"\n{'='*45}")
        print(f"  Stats @ frame {frame_cnt}")
        print(f"  Loop tour/s  : {loop_cnt / elapsed_loop:.1f}")
        print(f"  FPS          : {frame_cnt / elapsed_loop:.1f}")
        print(f"  Latence YOLO : {(t_total_inf / frame_cnt)*1000:.2f} ms")
        print(f"  Depth min/max: {depth_info}")
        print(f"  Cible        : {selected_id}")
        print(f"{'='*45}\n")

    # ---------- Confidence filtering ----------
    mask    = scores > CONF_THRES
    boxes   = boxes[mask]
    scores  = scores[mask]
    classes = classes[mask]
    n_conf  = int(mask.sum())                    # [MON]

    if len(boxes):
        cx = boxes[:, 0] * FRAME_W
        cy = boxes[:, 1] * FRAME_H
        w  = boxes[:, 2] * FRAME_W
        h  = boxes[:, 3] * FRAME_H

        x1 = (cx - w / 2).astype(np.int32)
        y1 = (cy - h / 2).astype(np.int32)
        x2 = (cx + w / 2).astype(np.int32)
        y2 = (cy + h / 2).astype(np.int32)
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        boxes_cv2 = np.column_stack((x1, y1, x2 - x1, y2 - y1))
        cx_int = cx.astype(np.int32)
        cy_int = cy.astype(np.int32)

        valid_zone = cy_int >= MIN_CY_VALID
        boxes_xyxy = boxes_xyxy[valid_zone]
        boxes_cv2  = boxes_cv2[valid_zone]
        scores     = scores[valid_zone]
        cx_int     = cx_int[valid_zone]
        cy_int     = cy_int[valid_zone]
        cx         = cx[valid_zone]
        cy         = cy[valid_zone]
        n_after_cy = len(boxes_xyxy)             # [MON]

        if len(boxes_xyxy):
            claw_mask = np.array([not is_in_claw_zone(b, FRAME_W, FRAME_H)
                                  for b in boxes_xyxy], dtype=bool)
            boxes_xyxy = boxes_xyxy[claw_mask]
            boxes_cv2  = boxes_cv2[claw_mask]
            scores     = scores[claw_mask]
            cx_int     = cx_int[claw_mask]
            cy_int     = cy_int[claw_mask]
            cx         = cx[claw_mask]
            cy         = cy[claw_mask]
        n_after_claw = len(boxes_xyxy)           # [MON]

        if len(boxes_xyxy):
            big_mask = np.array([not is_box_too_big(b, FRAME_W, FRAME_H)
                                 for b in boxes_xyxy], dtype=bool)
            boxes_xyxy = boxes_xyxy[big_mask]
            boxes_cv2  = boxes_cv2[big_mask]
            scores     = scores[big_mask]
            cx_int     = cx_int[big_mask]
            cy_int     = cy_int[big_mask]
            cx         = cx[big_mask]
            cy         = cy[big_mask]
        n_after_big = len(boxes_xyxy)            # [MON]

        with mon.stage("nms"):                   # [MON]
            if len(boxes_xyxy):
                idx_cv2 = cv2.dnn.NMSBoxes(
                    boxes_cv2.tolist(), scores.tolist(), CONF_THRES, NMS_IOU_THRES
                )
            else:
                idx_cv2 = ()
        n_final = len(idx_cv2)                   # [MON]

        if len(idx_cv2):
            indx = idx_cv2.flatten()

            current_boxes_list = [boxes_xyxy[i].tolist() for i in indx]
            with mon.stage("tracking"):          # [MON]
                track_assignments = update_tracks(current_boxes_list, frame_cnt)

            # [MON] garde de non-régression : deux détections ne doivent jamais
            # partager un identifiant. Coût ~2 us/frame. Logué plutôt que levé
            # en assert, pour ne pas tuer une session ROV en cours.
            n_dup_ids = len(track_assignments) - len(set(track_assignments.values()))
            if n_dup_ids:
                print(f"[TRACK] {n_dup_ids} identifiant(s) dupliqué(s) "
                      f"frame {frame_cnt}: {track_assignments}")

            for local_idx, i in enumerate(indx):
                cxi = cx_int[i]
                cyi = cy_int[i]
                x1i, y1i, x2i, y2i = boxes_xyxy[i]
                sc = scores[i]
                track_id = track_assignments[local_idx]
                tracked_objects[track_id]['conf'] = float(sc)

                with mon.stage("depth_sample"):  # [MON]
                    cx_m_raw = np.clip(cxi * scale_x_depth, 0, in_w_depth - 1)
                    cy_m_raw = np.clip(cyi * scale_y_depth, 0, in_h_depth - 1)

                    prev_cx_m = tracked_objects[track_id].get('cx_m_f')
                    prev_cy_m = tracked_objects[track_id].get('cy_m_f')

                    cx_m_f = cx_m_raw if prev_cx_m is None else (
                        ALPHA_CXY_DEPTH * cx_m_raw + (1 - ALPHA_CXY_DEPTH) * prev_cx_m)
                    cy_m_f = cy_m_raw if prev_cy_m is None else (
                        ALPHA_CXY_DEPTH * cy_m_raw + (1 - ALPHA_CXY_DEPTH) * prev_cy_m)
                    tracked_objects[track_id]['cx_m_f'] = cx_m_f
                    tracked_objects[track_id]['cy_m_f'] = cy_m_f

                    cx_m = int(round(cx_m_f))
                    cy_m = int(round(cy_m_f))

                    if Z_DIAG:
                        z_px, z_med, z_std = sample_depth(depth_map, cx_m, cy_m, win=5)
                        zlog.log(frame_cnt, valid=True, conf=float(sc),
                                 cx_m=cx_m, cy_m=cy_m,
                                 z_px=z_px, z_med=z_med, z_std=z_std)

                    if depth_map is not None:
                        z_raw_i = float(depth_map[cy_m, cx_m])
                    else:
                        z_raw_i = 0.0

                    # ---------- [FUSION] aire de boîte lissée ----------
                    # EMA sur l'aire : la boîte de YOLO tremble d'une frame à
                    # l'autre et sqrt() propagerait ce bruit dans z_bb.
                    area_i = float(max(1.0, (x2i - x1i) * (y2i - y1i)))
                    prev_a = tracked_objects[track_id].get('area_f')
                    area_f = area_i if prev_a is None else (
                        ALPHA_AREA * area_i + (1 - ALPHA_AREA) * prev_a)
                    tracked_objects[track_id]['area_f'] = area_f

                    # Boîte tronquée par un bord de l'image : l'aire visible
                    # n'est plus proportionnelle à la taille apparente, le
                    # terme géométrique devient faux. On repasse au modèle seul.
                    at_border = (x1i <= BORDER_MARGIN_PX
                                 or y1i <= BORDER_MARGIN_PX
                                 or x2i >= FRAME_W - BORDER_MARGIN_PX
                                 or y2i >= FRAME_H - BORDER_MARGIN_PX)

                    # ---------- [FUSION] combinaison en espace disparité ----------
                    z_bb_i = None
                    a_eff = 1.0
                    if (track_id == selected_id
                            and z_raw_desired is not None
                            and area_desired is not None
                            and area_desired > 0 and z_raw_desired > 0):
                        z_bb_i = z_raw_desired * math.sqrt(area_f / area_desired)
                        a_eff = 1.0 if at_border else FUSION_ALPHA
                        z_fused_i = a_eff * z_raw_i + (1.0 - a_eff) * z_bb_i
                    else:
                        z_fused_i = z_raw_i

                    z_corrected = Z_MAX - z_fused_i

                if depth_map is not None:
                    prev_z = tracked_objects[track_id].get('z_f')
                    tracked_objects[track_id]['z_f'] = (
                        z_corrected if prev_z is None
                        else ALPHA_Z * z_corrected + (1 - ALPHA_Z) * prev_z
                    )
                    # [FUSION] capture différée des DEUX ancres, sur la même
                    # frame : la disparité du modèle et l'aire lissée doivent
                    # décrire le même instant, sinon z_bb est biaisé dès le
                    # départ.
                    if z_desired_pending and track_id == selected_id:
                        z_raw_desired = z_raw_i
                        area_desired  = area_f
                        z_set = Z_MAX - z_raw_i
                        if Z_DESIRED_ROUND_TO > 0:
                            z_set = math.ceil(z_set / Z_DESIRED_ROUND_TO) * Z_DESIRED_ROUND_TO
                        z_desired = z_set
                        print(f"[FUSION] ancres : z_raw={z_raw_desired:.2f} "
                              f"area={area_desired:.0f} px2 -> setpoint z={z_desired:.1f}")
                        z_desired_pending = False

                if track_id == selected_id:      # [MON] signaux de la cible
                    z_raw    = z_raw_i
                    z_bb     = z_bb_i
                    z_fused  = z_fused_i
                    area_sel = area_f
                    fusion_fallback = int(at_border and z_bb_i is not None)

                list_data.append(
                    f"{track_id},{int(cxi - no_x)},{int(no_y - cyi)},"
                    f"{z_corrected:.2f},{float(sc):.2f}"
                )

                # --- Drawing ---
                with mon.stage("render"):        # [MON]
                    is_sel = (track_id == selected_id)
                    color = (0, 255, 0) if z_corrected >= 800 else (0, 0, 255)
                    thick = 2
                    if is_sel:
                        color = (0, 255, 255)
                        thick = 4
                    cv2.rectangle(output_frame, (x1i, y1i), (x2i, y2i), color, thick)
                    cv2.circle(output_frame, (cxi, cyi), 6, (0, 0, 255), -1)
                    cv2.putText(output_frame,
                                f"x:{int(cxi-no_x)} y:{int(no_y-cyi)} z:{z_corrected:.2f}",
                                (x1i, max(10, y1i - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(output_frame, f"{sc:.2f}", (x1i, max(10, y1i - 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(output_frame, f"Object:{track_id}", (x1i, max(10, y1i - 50)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 200), 2)
                    if DEBUG_EXCLUSION_ZONE:
                        cv2.rectangle(
                            output_frame,
                            (CLAW_EXCLUSION_ZONE_PX["x_min"], CLAW_EXCLUSION_ZONE_PX["y_min"]),
                            (CLAW_EXCLUSION_ZONE_PX["x_max"], CLAW_EXCLUSION_ZONE_PX["y_max"]),
                            (255, 0, 255), 2)
                    if is_sel:
                        cv2.putText(output_frame, "CIBLE", (x1i, max(10, y1i - 80)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # ---------- Envois UDP ----------
    with mon.stage("udp_send"):                  # [MON]
        sock.sendto(";".join(list_data).encode(), (MAC_IP, LIST_PORT))

        t_sel = tracked_objects.get(selected_id) if selected_id is not None else None
        if t_sel is not None and t_sel.get('z_f') is not None:
            dx = float(np.clip(t_sel['x_f'] / (FRAME_W / 2), -1.0, 1.0))
            dy = float(np.clip(t_sel['y_f'] / (FRAME_H / 2), -1.0, 1.0))
            ez = 0.0
            if z_desired is not None:
                ez = float(np.clip((z_desired - t_sel['z_f']) / Z_MAX, -1.0, 1.0))
            packet = json.dumps({
                "vision_valid":     True,
                "confidence":       float(t_sel.get('conf', 1.0)),
                "dx_normalized":    dx,
                "dy_normalized":    dy,
                "scale_error_norm": ez,
            }).encode()
            sock.sendto(packet, (UDP_IP, UDP_PORT))
        else:
            packet = json.dumps({
                "vision_valid":     False,
                "confidence":       0.0,
                "dx_normalized":    0.0,
                "dy_normalized":    0.0,
                "scale_error_norm": 0.0,
            }).encode()
            sock.sendto(packet, (UDP_IP, UDP_PORT))

    # ---------- Video Output ----------
    with mon.stage("gst_push"):                  # [MON]
        cv2.circle(output_frame, (int(FRAME_W/2), int(FRAME_H/2)), 6, (255, 0, 0), -1)
        cv2.line(output_frame, (0, int(FRAME_H/2)), (FRAME_W, int(FRAME_H/2)), (230, 0, 0), 1)
        cv2.putText(output_frame, "(x: 0 , y : 0 , z : 0)",
                    (int(FRAME_W/2) + 10, int(FRAME_H/2) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        data = output_frame.tobytes()
        buf = Gst.Buffer.new_wrapped(data)
        buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, FPS_OUT)
        buf.pts = frame_cnt * buf.duration
        buf.dts = buf.pts
        appsrc.emit('push-buffer', buf)

    # ---------- [MON] one CSV row per processed frame ----------
    bbox = t_sel.get('bbox') if t_sel else None
    area = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) if bbox else None

    mon.frame_row(
        frame_cnt=frame_cnt, seq=seq,
        run_depth=int(run_depth),
        n_raw=n_raw, n_conf=n_conf, n_after_cy=n_after_cy,
        n_after_claw=n_after_claw, n_after_big=n_after_big, n_final=n_final,
        n_tracks=len(tracked_objects), n_dup_ids=n_dup_ids,
        track_age_s=(round(time.time() - t_sel['created_t'], 2)
                     if t_sel and 'created_t' in t_sel else None),
        selected_id=(selected_id if selected_id is not None else -1),
        vision_valid=bool(t_sel is not None
                          and t_sel.get('z_f') is not None
                          and t_sel.get('last_seen') == frame_cnt),
        conf=(t_sel.get('conf') if t_sel else None),
        dx=dx, dy=dy, ez=ez,
        z_raw=z_raw,                 # disparité brute du modèle
        z_bb=z_bb,                   # [FUSION] estimation géométrique
        z_fused=z_fused,             # [FUSION] signal effectivement utilisé
        fusion_alpha=FUSION_ALPHA,
        fusion_fallback=fusion_fallback,
        area_sel=area_sel,
        area_desired=area_desired,
        z_raw_desired=z_raw_desired,
        z_f=(t_sel.get('z_f') if t_sel else None),
        z_desired=z_desired,
        bbox_area=area,
        bbox_area_ratio=(area / (FRAME_W * FRAME_H)) if area else None,
        cx=(t_sel.get('cx') if t_sel else None),
        cy=(t_sel.get('cy') if t_sel else None),
        depth_min=(float(depth_map.min()) if depth_map is not None else None),
        depth_max=(float(depth_map.max()) if depth_map is not None else None),
        depth_mean=(float(depth_map.mean()) if depth_map is not None else None),
    )

    # ---------- Throttle ----------
    if not NO_THROTTLE:                          # [MON-FIX-3]
        elapsed = time.time() - t_start_iter
        if elapsed < MIN_LOOP_DT:
            with mon.stage("throttle_sleep"):    # [MON]
                time.sleep(MIN_LOOP_DT - elapsed)