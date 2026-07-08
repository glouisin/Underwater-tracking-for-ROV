# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


# =============================================================================
# ROV Vision Pipeline
#
# This script is the main perception loop running on the ROV (Remotely Operated
# Vehicle). It grabs camera frames, runs two AI models in parallel:
#   - YOLOv8n         → detects objects and gives their 2D bounding boxes
#   - Depth Anything V2 → estimates depth (the z-axis) from a single RGB image
#
# Detected objects are tracked across frames using a simple centroid tracker.
# The operator selects a target from the Mac interfaace; the ROV then streams
# normalized (dx, dy, dz) error signals to the PID controller via UDP.
# The annotated video feed is simultaneously streamed to the Mac over GStreamer.
# =============================================================================

#!/usr/bin/env python3

import os

# The Qualcomm HTP (Hexagon Tensor Processor) delegate needs these shared
# libraries at runtime. We prepend them to any existing path so they are
# found before anything else on the system.
os.environ["ADSP_LIBRARY_PATH"] = (
    "/opt/qcom/qairt-new/qairt/2.48.0.260626/lib/hexagon-v73/unsigned;"
    "/opt/qcom/qirp-sdk/lib/aarch64-oe-linux-gcc11.2;"
    "/opt/qcom/qirp-sdk/lib/hexagon-v73/unsigned;"
    + os.environ.get("ADSP_LIBRARY_PATH", "")
)

import json
import socket
import time
import cv2
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import ai_edge_litert.interpreter as tflite  # TFLite runtime with delegate support
import threading
#Activate for z jitter diagnostic
Z_DIAG = False
if Z_DIAG:

    from z_diagnostic import sample_depth, ZDiagnosticLogger
Gst.init(None)


# -------------------- Auto IP connection --------------------
def get_ssh_client_ip():
    # SSH_CONNECTION format: "client_ip client_port server_ip server_port"
    conn = os.environ.get("SSH_CONNECTION")
    if conn:
        return conn.split()[0]

    # Fallback : SSH_CLIENT format: "client_ip client_port server_port"
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

# INT-8 quantized YOLOv8n model — small and fast, good trade-off on edge hardware
MODEL_PATH_YOLO  = "/root/ai-rov/src/models/deepbox-tflite-float/yolov8n_saved_model/int8/yolov8n_full_integer_quant.tflite"

# --- CHANGEMENT ---
# Depth Anything V2 quantifié (à adapter au chemin réel de ton .tflite converti).
# NOTE : vérifie la variante utilisée (vits/vitb) et sa résolution d'entrée native
# (souvent 518x518 côté modèle original, mais un export TFLite quantifié peut
# avoir été redimensionné, ex: 256x256 ou 384x384). Peu importe : le code lit
# la shape et les paramètres de quantization directement depuis l'interpreter,
# donc aucune autre modification n'est nécessaire tant que le fichier .tflite
# est correct.
MODEL_PATH_DEPTH = "/root/ai-rov/src/models/depth_anything_v2-tflite-float/depth_anything_v2.tflite"

# Qualcomm QNN TFLite delegate — offloads inference to the HTP hardware accelerator
DELEGATE_PATH    = "/opt/qcom/qairt-new/qairt/2.48.0.260626/lib/aarch64-oe-linux-gcc11.2/libQnnTFLiteDelegate.so"

MAC_IP           = get_ssh_client_ip() # IP for streaming
UDP_IP           = "127.0.0.1"
UDP_PORT         = 17002         # Coordinates transmission
SELECT_PORT      = 17001         # Listening selections (Mac -> ROV)
LIST_PORT        = 17003          # Diffusion of object list (ROV -> Mac)
VIDEO_IN      = "13759780_2160_3840_60fps.mp4"
# Activate for video test
TEST_VIDEO = False

FRAME_W       = 1280  # output / display resolution
FRAME_H       = 720
FPS_OUT       = 15   # GStreamer stream framerate (can exceed camera capture rate)
CONF_THRES    = 0.02  # detections below this confidence score are discarded

MIN_CY_VALID  = 60      # NOUVEAU : rejette les détections trop hautes dans l'image (fantômes de surface)

NMS_IOU_THRES = 0.1 # IoU threshold for Non-Maximum Suppression — lower = keep more overlapping boxes
#EMA parameters
# Exponential Moving Average smoothing: alpha close to 0 → very smooth but slow to react;
# alpha close to 1 → reacts quickly but jittery. Tune independently per axis.
ALPHA_XY =  0.15     # reactivity x,y
ALPHA_Z  = 0.05   # reactivity z
ALPHA_CXY_DEPTH = 0.25   # reactivity centroids (ex ALPHA_CXY_MIDAS)
#Activate for color map (debugging, uses CPU ressoucrces)
DEBUG_DEPTH = False
DEBUG_EXCLUSION_ZONE = True

INFER_FPS = 15 # limite le rythme d'inférence, indépendamment de la vitesse caméra
MIN_LOOP_DT = 1.0 / INFER_FPS

MAX_BOX_AREA_RATIO = 0.8

# --- CHANGEMENT / A VERIFIER ---
# Depth Anything V2 (mode relatif) suit la même convention que MiDaS : une
# valeur de sortie plus élevée = objet plus proche (disparité / inverse depth).
# C'est une hypothèse à valider empiriquement dès le premier run (voir le
# print de calibration ajouté plus bas, même logique que ce qui a été fait
# pour MiDaS). Si les couleurs proche/loin sont inversées à l'écran, passe
# DEPTH_INVERT à True.
DEPTH_INVERT = False

# --- TEMPORAIRE ---
# Le delegate QNN HTP ne supporte pas l'op GELU utilisée dans le backbone
# ViT/DINOv2 de Depth Anything V2 (graph_prepare.cc échoue sur QNN_Gelu,
# "no properties registered for q::QNN_Gelu"). En attendant une résolution
# côté modèle (re-export avec activation compatible HTP, ou rapport de
# compatibilité QAI Hub), on fait tourner ce modèle en CPU pur pour valider
# d'abord si la qualité de profondeur en eau trouble justifie l'effort de
# portage NPU. YOLO reste sur HTP.
DEPTH_USE_HTP = True

# -------------------- Object detection exclusion --------------------
# Zone d'exclusion pour la pince (à calibrer selon ta résolution, ex: 1280x720)
CLAW_EXCLUSION_ZONE_PX = {
    "x_min": 448 , "x_max": 832,  # centré horizontalement
    "y_min": 576, "y_max": 720,   # bas de l'image
}

def is_in_claw_zone(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return (CLAW_EXCLUSION_ZONE_PX["x_min"] <= cx <= CLAW_EXCLUSION_ZONE_PX["x_max"] and
            CLAW_EXCLUSION_ZONE_PX["y_min"] <= cy <= CLAW_EXCLUSION_ZONE_PX["y_max"])

def is_box_too_big(box, frame_w, frame_h):
    x1, y1, x2,y2 = box
    box_area = (x2-x1)*(y2-y1)
    frame_area = (frame_w)*(frame_h)
    return (box_area/frame_area) > MAX_BOX_AREA_RATIO
# -------------------- Load Delegate --------------------
# HTP = Hexagon Tensor Processor, the DSP-based AI accelerator on Qualcomm SoCs.
# Loading it once here and reusing it for both models avoids reloading overhead.
delegate_options = {'backend_type': 'htp', 'htp_performance_mode': '1',}
delegate = tflite.load_delegate(DELEGATE_PATH, options=delegate_options)

# -------------------- Load YOLO --------------------
# Both models share the same delegate instance so inference can be dispatched
# to the HTP back-to-back without re-initialisation cost.
yolo_interpreter = tflite.Interpreter(
    model_path=MODEL_PATH_YOLO,
    experimental_delegates=[delegate]
)
yolo_interpreter.allocate_tensors()

in_det_yolo  = yolo_interpreter.get_input_details()
out_det_yolo = yolo_interpreter.get_output_details()

# Extract input spatial size and quantization parameters once so we can
# reuse them every frame without querying the interpreter each time.
in_h_yolo, in_w_yolo       = in_det_yolo[0]["shape"][1:3]
in_scale_yolo, in_zp_yolo  = in_det_yolo[0]["quantization"]   # scale and zero-point for INT8 input
out_scale_yolo, out_zp_yolo = out_det_yolo[0]["quantization"]  # same for the output tensor

print(f"YOLO  — entrée : {in_w_yolo}x{in_h_yolo}  dtype: {in_det_yolo[0]['dtype']}")

# -------------------- Load Depth Anything V2 --------------------
# --- CHANGEMENT --- (remplace le bloc MiDaS)
# Depth Anything V2 (comme MiDaS) sort une carte de profondeur *relative*
# (inverse depth) : valeur brute plus grande = pixel plus proche de la caméra.
# On garde la même logique d'inversion plus bas (z_corrected = Z_MAX - z_raw)
# sous réserve de validation (voir DEPTH_INVERT ci-dessus).
if DEPTH_USE_HTP:
    delegate_options_depth = {
        'backend_type': 'htp',
        'htp_performance_mode': '1',
    }
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

in_h_depth, in_w_depth         = in_det_depth[0]["shape"][1:3]
in_scale_depth, in_zp_depth    = in_det_depth[0]["quantization"]
out_scale_depth, out_zp_depth  = out_det_depth[0]["quantization"]

print(f"Depth Anything V2 — entrée : {in_w_depth}x{in_h_depth}  dtype: {in_det_depth[0]['dtype']}")

# --- CHANGEMENT ---
# Contrairement à MiDaS (quantifié int8/uint8 full-integer), ce modèle DA-V2
# peut être exporté en float32 (pas de quantization d'entrée/sortie). On
# détecte le cas au runtime pour éviter de diviser par un scale nul.
DEPTH_INPUT_DTYPE  = in_det_depth[0]['dtype']
DEPTH_OUTPUT_DTYPE = out_det_depth[0]['dtype']
DEPTH_IS_QUANTIZED = DEPTH_INPUT_DTYPE in (np.uint8, np.int8) and in_scale_depth != 0

# Normalisation ImageNet standard (RGB), utilisée par la plupart des modèles
# ViT/DINOv2 dont Depth Anything V2 — à vérifier contre le script d'export
# si tu l'as (certains pipelines utilisent une normalisation [-1,1] simple
# à la place).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Facteurs d'échelle pour convertir cx/cy → espace d'entrée du modèle de profondeur
# Pixel coordinates from YOLO are in 1280×720 space; we need to map them
# into the depth model's smaller input grid to sample the depth at the right location.
scale_x_depth = in_w_depth / FRAME_W
scale_y_depth = in_h_depth / FRAME_H

# --- CHANGEMENT ---
# En quantifié, Z_MAX = 255 * out_scale donnait la valeur max représentable
# par le TFLite converter (calibrée sur le representative dataset). En
# float32, il n'y a pas d'équivalent : la sortie est une carte de profondeur
# relative à l'échelle arbitraire, généralement normalisée par frame
# (min/max) dans les démos Depth Anything. On fixe donc une constante
# Z_MAX_FLOAT arbitraire pour rester compatible avec le reste du pipeline
# (seuils couleur à 850, normalisation de ez, etc.) — À RECALIBRER après un
# premier test, ces seuils ont été réglés pour la plage MiDaS quantifiée et
# n'ont aucune raison de rester pertinents ici.
Z_MAX_FLOAT = 1000.0
Z_MAX = (255.0 * out_scale_depth) if DEPTH_IS_QUANTIZED else Z_MAX_FLOAT

# -------------------- GStreamer Pipeline --------------------
# appsrc lets us push raw BGR frames from Python into the pipeline.
# From there: convert colour space → hardware H.264 encoder (qtic2venc) →
# wrap in MPEG-TS → send over UDP to the Mac operator station.
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
    cap= cv2.VideoCapture(VIDEO_IN)
# Preallocate buffers
# Allocating numpy arrays once outside the loop avoids triggering the Python
# garbage collector on every frame, which would add unpredictable latency.
frame_rs           = np.empty((FRAME_H, FRAME_W, 3), np.uint8)
input_tensor_yolo  = np.empty((1, in_h_yolo,  in_w_yolo,  3), np.int8)
input_tensor_depth = np.empty((1, in_h_depth, in_w_depth, 3), DEPTH_INPUT_DTYPE)
# ---- Reusable buffers for quantization (optimization) ----
# Pre-compute the combined pixel→quantized-int8 scale factor to replace
# a division and a multiplication each frame with a single multiplication.
YOLO_SCALE  = 1.0 / (255.0 * in_scale_yolo)
DEPTH_SCALE = 1.0 / (255.0 * in_scale_depth) if DEPTH_IS_QUANTIZED else None
buf_yolo_f32  = np.empty((in_h_yolo,  in_w_yolo,  3), np.float32)
buf_depth_f32 = np.empty((in_h_depth, in_w_depth, 3), np.float32)
# -------------------- Stats --------------------

frame_cnt    = 0
t_total_inf  = 0.0
t_start_loop = time.time()

# -------------------- Paramètres tracker --------------------
MAX_DIST_PIXELS  = 450 # max dist between two centroids to be considered the same object
MAX_LOST_FRAMES  = 120    # max frames before suppression of the object

# -------------------- Tracker centroïde --------------------
tracked_objects = {}  # {track_id: {'cx': , 'cy': , 'bbox': , 'last_seen': }}
next_id = 0

def update_tracks(current_boxes, frame_cnt):
    """
    Tracker based on euclidian distance between centroids

    Better than IoU (see midas_yolo_iou.py) because of yolo's jitter
    """
    # Greedy nearest-neighbour assignment: for each new detection we find
    # the existing track whose centroid is closest. If the closest track is
    # within MAX_DIST_PIXELS we link them; otherwise we create a new track.
    # This runs in O(detections × tracks) which is fine at typical ROV scene
    # densities (usually < 20 objects at once).
    global tracked_objects, next_id
    assignments  = {}
    used_tracks  = set()

    # Calcul des centroïdes des détections courantes
    centroids = [
        ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        for b in current_boxes
    ]

    for idx, (cx, cy) in enumerate(centroids):
        best_id   = None
        best_dist = MAX_DIST_PIXELS  # treshold in pixel

        for tid, tobj in tracked_objects.items():
            if tid in used_tracks:
                continue
            dx   = cx - tobj['cx']
            dy   = cy - tobj['cy']
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < best_dist:
                best_dist = dist
                best_id   = tid

        if best_id is not None:
            # Objet connu — mise à jour position
            assignments[idx]                      = best_id
            tracked_objects[best_id]['cx']        = cx
            tracked_objects[best_id]['cy']        = cy
            tracked_objects[best_id]['bbox']      = current_boxes[idx]
            tracked_objects[best_id]['last_seen'] = frame_cnt
            used_tracks.add(best_id)
            # Filtre exponentiel sur les coordonnées envoyées au PID
            # EMA on x_f / y_f (expressed as offsets from image centre) so the
            # PID controller receives a smoothed error signal rather than raw
            # noisy pixel positions.
            tracked_objects[best_id]['x_f'] = (
                ALPHA_XY * (cx - FRAME_W/2)
                + (1 - ALPHA_XY) * tracked_objects[best_id].get('x_f', cx - FRAME_W/2)
            )
            tracked_objects[best_id]['y_f'] = (
                ALPHA_XY * (FRAME_H/2 - cy)
                + (1 - ALPHA_XY) * tracked_objects[best_id].get('y_f', FRAME_H/2 - cy)
            )

        else:
            # Nouvel objet
            # First time we see this object: initialise with raw values (no
            # smoothing yet) and assign a new monotonically increasing ID.
            assignments[idx] = next_id
            tracked_objects[next_id] = {
                'cx':        cx,
                'cy':        cy,
                'bbox':      current_boxes[idx],
                'last_seen': frame_cnt,
                'x_f':       cx - FRAME_W/2,
                'y_f':       FRAME_H/2 - cy,
                'z_f':       None,   # initialisé au premier z_corrected
            }
            next_id += 1

    # Nettoyage des tracks perdus
    # Remove any track that hasn't been matched for too long — they've either
    # left the scene or were spurious detections.
    to_delete = [
        tid for tid, tobj in tracked_objects.items()
        if frame_cnt - tobj['last_seen'] > MAX_LOST_FRAMES
    ]
    for tid in to_delete:
        del tracked_objects[tid]

    return assignments

# -------------------- UDP Sockets --------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   # envois PID + liste Mac

# Socket de réception des sélections (non bloquante : ne ralentit pas la boucle)
# Non-blocking so that polling it each frame adds near-zero overhead even when
# there are no incoming messages from the operator.
sel_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sel_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sel_sock.bind(("0.0.0.0", SELECT_PORT))
sel_sock.setblocking(False)

selected_id = None   # track_id verrouillé pour le PID (None = aucune cible)

z_desired = None # lock the detected z
def poll_selection():
    """Vide la socket de sélection et met à jour selected_id (sélection par ID)."""
    # Drains the socket entirely each call (loop until BlockingIOError) so
    # that if multiple messages arrived between two frames we always end up
    # with the most recent operator intent.
    global selected_id, z_desired
    while True:
        try:
            data, _ = sel_sock.recvfrom(64)
        except BlockingIOError:
            break                                   # plus rien à lire
        msg = data.decode(errors="ignore").strip()
        if msg == "clear":
            # Operator deselected the target — go back to idle mode
            selected_id = None
            z_desired = None
        elif msg.startswith("select,"):
            try:
                new_id = int(msg.split(",")[1])
                selected_id = new_id
                # Snapshot the target's current depth so the ROV tries to
                # maintain the same distance (z_desired acts as a setpoint).
                if new_id in tracked_objects and tracked_objects[new_id].get('z_f'):
                        z_desired = tracked_objects[new_id]['z_f']
            except (ValueError, IndexError):
                pass

if Z_DIAG:
    zlog = ZDiagnosticLogger("z_diag.csv")
# -------------------- Main Loop --------------------
# Everything from here runs as fast as the hardware allows (no explicit sleep).
# Each iteration: grab frame → preprocess → infer → postprocess → track → send → stream.
last_seq=-1
loop_cnt =0
while True:
    t_start_iter = time.time()
    loop_cnt+=1
    if cap.seq == last_seq:
        time.sleep(0.002)
        continue
    ok, frame, seq = cap.read()
    if not ok or frame is None:
        continue

    frame_cnt += 1

    poll_selection()          # met à jour selected_id selon les clics du Mac
    list_data = []            # liste complète des objets -> interface Mac

    # Center of frame as new origin
    # Redefine pixel (0,0) as the image centre so the error signals we send
    # to the PID are naturally zero when the target is centred on screen.
    no_x = FRAME_W / 2
    no_y = FRAME_H / 2

    # ---------- Preprocessing ----------
    # Resize to the output resolution first; all subsequent resizes branch
    # from this common base to avoid redundant full-resolution copies.
    cv2.resize(frame, (FRAME_W, FRAME_H), dst=frame_rs)

    # YOLO — BGR
    # Scale pixel values [0,255] to the INT8 quantized range expected by the model,
    # applying the model's zero-point offset, then clip to prevent overflow.
    resized_yolo = cv2.resize(frame_rs, (in_w_yolo, in_h_yolo))
    buf_yolo_f32[:] = resized_yolo
    buf_yolo_f32   *= YOLO_SCALE
    buf_yolo_f32   += in_zp_yolo
    np.clip(buf_yolo_f32, -128, 127, out=buf_yolo_f32)
    input_tensor_yolo[0] = buf_yolo_f32.astype(np.int8)

    # Depth Anything V2 — RGB
    # Comme MiDaS, le modèle attend du RGB, donc conversion depuis le BGR
    # d'OpenCV. Deux chemins selon le type de modèle chargé :
    #   - quantifié (int8/uint8)  → même logique que MiDaS
    #   - float32 (non quantifié) → normalisation ImageNet standard
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
    # Both models are invoked back-to-back on the HTP. On this hardware they
    # run sequentially (not truly parallel), but the delegate batches the DSP
    # dispatch efficiently.
    yolo_interpreter.set_tensor(in_det_yolo[0]['index'],  input_tensor_yolo)
    run_depth = selected_id is not None
    if run_depth: # correction
        depth_interpreter.set_tensor(in_det_depth[0]['index'], input_tensor_depth)
        depth_interpreter.invoke()


    t0 = time.time()
    yolo_interpreter.invoke()

    t_inf = time.time() - t0
    t_total_inf += t_inf

    # ---------- Postprocessing ----------
    depth_map = None
    output_raw_yolo  = yolo_interpreter.get_tensor(out_det_yolo[0]['index'])[0]
    if run_depth:
        output_raw_depth = depth_interpreter.get_tensor(out_det_depth[0]['index'])[0]
        if DEPTH_IS_QUANTIZED:
            # Dequantize Depth Anything V2 → depth map
            # Result is a 2-D array (same H×W as model input) where each value is a
            # relative inverse depth. Higher = closer; we invert later to get z_corrected.
            depth_map = (output_raw_depth.astype(np.float32) - out_zp_depth) * out_scale_depth
        else:
            # Sortie float32 brute : carte de profondeur relative à échelle
            # arbitraire. On la squeeze au besoin (shape peut être (H,W) ou
            # (H,W,1) selon l'export) puis on normalise min/max par frame
            # pour la ramener dans une plage comparable à Z_MAX_FLOAT.
            raw = np.squeeze(output_raw_depth).astype(np.float32)
            d_min, d_max = raw.min(), raw.max()
            depth_map = (raw - d_min) / (d_max - d_min + 1e-6) * Z_MAX_FLOAT
        if DEPTH_INVERT:
            depth_map = Z_MAX - depth_map

    # Dequantize YOLO
    # Convert raw INT8 tensor back to float32 using the stored scale/zero-point,
    # then transpose from YOLOv8's native [84, 8400] layout to [8400, 84] so
    # each row is one candidate detection with [cx, cy, w, h, cls0…cls79].
    output = (output_raw_yolo.astype(np.float32) - out_zp_yolo) * out_scale_yolo
    output = output.transpose()   # [84, 8400] → [8400, 84]



    boxes        = output[:, :4]   # centre-x, centre-y, width, height (normalised 0–1)
    class_scores = output[:, 4:]   # per-class confidence scores
    scores       = np.max(class_scores, axis=1)    # best class score for each detection
    classes      = np.argmax(class_scores, axis=1) # corresponding class index

    # ---------- Depth visualisation ----------
    # Blend the MAGMA-coloured depth map over the camera feed so engineers can
    # visually verify depth quality. Disabled by default — costs ~5 ms of CPU.
    if DEBUG_DEPTH :
        depth_norm    = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_color   = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)
        depth_display = cv2.resize(depth_color, (FRAME_W, FRAME_H))
        output_frame  = cv2.addWeighted(frame_rs, 0.5, depth_display, 0.5, 0)
    else :
        output_frame = frame_rs
    # ---------- Stats every 30 frames ----------
    if frame_cnt % 30 == 0:
        elapsed_loop = time.time() - t_start_loop
        print(f"\n{'='*45}")
        print(f"  Stats @ frame {frame_cnt}")

        print(f"  Loop tour/s  : {loop_cnt / elapsed_loop:.1f}")   # devrait etre eleve
        print(f"  FPS : {frame_cnt / elapsed_loop:.1f}")  # ~7-9
        print(f"  FPS inference only  : {1.0 / (t_total_inf / frame_cnt):.1f}")
        print(f"  Latence moy. inf.   : {(t_total_inf / frame_cnt)*1000:.2f} ms")
        print(f"  Latence last frame  : {t_inf*1000:.2f} ms")
        depth_info = f"{depth_map.min():.3f} / {depth_map.max():.3f}" if run_depth and 'depth_map' in dir() else "N/A"
        print(f"  Depth min/max       : {depth_info}")
        print(f"  Cible verrouillee   : {selected_id}")
        print(f"{'='*45}\n")

    # ---------- Confidence filtering ----------
    # Drop everything below CONF_THRES before doing anything more expensive.
    mask    = scores > CONF_THRES
    boxes   = boxes[mask]
    scores  = scores[mask]
    classes = classes[mask]

    if len(boxes):
        # Convert YOLOv8 normalised [cx, cy, w, h] to absolute pixel coordinates
        cx = boxes[:, 0] * FRAME_W
        cy = boxes[:, 1] * FRAME_H
        w  = boxes[:, 2] * FRAME_W
        h  = boxes[:, 3] * FRAME_H

        x1 = (cx - w / 2).astype(np.int32)
        y1 = (cy - h / 2).astype(np.int32)
        x2 = (cx + w / 2).astype(np.int32)
        y2 = (cy + h / 2).astype(np.int32)
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        boxes_cv2 = np.column_stack((x1, y1, x2 - x1, y2 - y1))  # OpenCV NMS expects [x, y, w, h]
        cx_int = cx.astype(np.int32)
        cy_int = cy.astype(np.int32)

        # rejet des fantômes en haut de frame
        valid_zone = cy_int >= MIN_CY_VALID
        boxes_xyxy  = boxes_xyxy[valid_zone]
        boxes_cv2   = boxes_cv2[valid_zone]
        scores      = scores[valid_zone]
        cx_int      = cx_int[valid_zone]
        cy_int      = cy_int[valid_zone]
        cx          = cx[valid_zone]
        cy          = cy[valid_zone]
        # rejet de la zone de la pince
        if len(boxes_xyxy):

            claw_mask = np.array([not is_in_claw_zone(box, FRAME_W, FRAME_H) for box in boxes_xyxy], dtype= bool)
            boxes_xyxy  = boxes_xyxy[claw_mask]
            boxes_cv2   = boxes_cv2[claw_mask]
            scores      = scores[claw_mask]
            cx_int      = cx_int[claw_mask]
            cy_int      = cy_int[claw_mask]
            cx          = cx[claw_mask]
            cy          = cy[claw_mask]
        # rejet des box trop larges
        if len(boxes_xyxy):

            big_mask = np.array([not is_box_too_big(box, FRAME_W, FRAME_H) for box in boxes_xyxy], dtype = bool)
            boxes_xyxy  = boxes_xyxy[big_mask]
            boxes_cv2   = boxes_cv2[big_mask]
            scores      = scores[big_mask]
            cx_int      = cx_int[big_mask]
            cy_int      = cy_int[big_mask]
            cx          = cx[big_mask]
            cy          = cy[big_mask]

        # Non-Maximum Suppression: removes duplicate boxes that cover the same object.
        # Returns the indices of the boxes to keep.
        if len(boxes_xyxy):
            idx_cv2 = cv2.dnn.NMSBoxes(
                boxes_cv2.tolist(), scores.tolist(), CONF_THRES, NMS_IOU_THRES
            )
        else:
            idx_cv2= ()



        if len(idx_cv2):

            indx = idx_cv2.flatten()

            # Build current boxes list and update tracker
            current_boxes_list = [boxes_xyxy[i].tolist() for i in indx]
            track_assignments  = update_tracks(current_boxes_list, frame_cnt)


            for local_idx, i in enumerate(indx):
                cxi                  = cx_int[i]
                cyi                  = cy_int[i]
                x1i, y1i, x2i, y2i  = boxes_xyxy[i]
                sc                   = scores[i]
                track_id             = track_assignments[local_idx]
                tracked_objects[track_id]['conf'] = float(sc)

                # --- Z extraction (espace natif Depth Anything V2) ---
                # Map the detection centroid from 1280×720 pixel space into the
                # depth model's coordinate system, then apply EMA smoothing
                # separately on the depth-space coordinates to reduce quantisation
                # noise before we sample the depth value.
                cx_m_raw = np.clip(cxi * scale_x_depth, 0, in_w_depth - 1)
                cy_m_raw = np.clip(cyi * scale_y_depth, 0, in_h_depth - 1)

                tid_for_smoothing = track_assignments[local_idx]
                prev_cx_m = tracked_objects[tid_for_smoothing].get('cx_m_f')
                prev_cy_m = tracked_objects[tid_for_smoothing].get('cy_m_f')

                cx_m_f = cx_m_raw if prev_cx_m is None else (
                    ALPHA_CXY_DEPTH * cx_m_raw + (1 - ALPHA_CXY_DEPTH) * prev_cx_m
                )
                cy_m_f = cy_m_raw if prev_cy_m is None else (
                    ALPHA_CXY_DEPTH * cy_m_raw + (1 - ALPHA_CXY_DEPTH) * prev_cy_m
                )
                tracked_objects[tid_for_smoothing]['cx_m_f'] = cx_m_f
                tracked_objects[tid_for_smoothing]['cy_m_f'] = cy_m_f

                cx_m = int(round(cx_m_f))
                cy_m = int(round(cy_m_f))
                if Z_DIAG:
                    z_px, z_med, z_std = sample_depth(depth_map, cx_m, cy_m, win=5)
                    z_raw = z_px
                    zlog.log(frame_cnt, valid=True, conf=float(sc),
                             cx_m=cx_m, cy_m=cy_m,
                             z_px=z_px, z_med=z_med, z_std=z_std)
                if depth_map is not None:
                    z_raw = float(depth_map[cy_m, cx_m])
                else:
                    z_raw = 0.0
                z_corrected = Z_MAX - z_raw  # large = far, small = close
                tid = track_assignments[local_idx]
                prev_z = tracked_objects[tid].get('z_f')
                # EMA on depth: very low alpha (0.05) because depth is
                # particularly noisy — we accept slow reaction in exchange
                # for a stable z signal going to the PID controller.
                tracked_objects[tid]['z_f'] = (
                        z_corrected if prev_z is None
                        else ALPHA_Z * z_corrected + (1 - ALPHA_Z) * prev_z
                )

                # --- Liste pour l'interface Mac : id,x,y,z,conf ---
                # Raw (not smoothed) coordinates are reported to the UI so the
                # operator sees a responsive bounding box, while the PID receives
                # the smoothed values (x_f, y_f, z_f) for stable control.
                list_data.append(
                    f"{track_id},{int(cxi - no_x)},{int(no_y - cyi)},"
                    f"{z_corrected:.2f},{float(sc):.2f}"
                )

                # --- Drawing ---
                is_sel = (track_id == selected_id)
                # Color encodes distance: green = far (safe clearance), red = close.
                # Selected target overrides to yellow so the operator always knows
                # which object the ROV is tracking.
                color  = (0, 255, 0) if z_corrected >= 850 else (0, 0, 255)
                thick  = 2
                if is_sel:
                    color = (0, 255, 255)   # jaune = cible verrouillée
                    thick = 4
                cv2.rectangle(output_frame, (x1i, y1i), (x2i, y2i), color, thick)
                cv2.circle(output_frame, (cxi, cyi), radius=6, color=(0, 0, 255), thickness=-1)
                cv2.putText(
                    output_frame,
                    f"x:{int(cxi-no_x)} y:{int(no_y-cyi)} z:{z_corrected:.2f}",
                    (x1i, max(10, y1i - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
                cv2.putText(
                    output_frame,
                    f"{sc:.2f}",
                    (x1i, max(10, y1i - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
                cv2.putText(
                    output_frame,
                    f"Object:{track_id}",
                    (x1i, max(10, y1i - 50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 200), 2

                )
                if DEBUG_EXCLUSION_ZONE:
                    cv2.putText(
                            output_frame,
                            "EXCLUSION ZONE",
                            (CLAW_EXCLUSION_ZONE_PX["x_min"], max(10, CLAW_EXCLUSION_ZONE_PX["y_min"] - 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2
                        )
                    cv2.putText(
                            output_frame,
                            f"x_min:{CLAW_EXCLUSION_ZONE_PX['x_min']}, "
                            f"y_min:{CLAW_EXCLUSION_ZONE_PX['y_min']}, "
                            f"x_max:{CLAW_EXCLUSION_ZONE_PX['x_max']}, "
                            f"y_max:{CLAW_EXCLUSION_ZONE_PX['y_max']}",
                            (CLAW_EXCLUSION_ZONE_PX["x_min"], max(10, CLAW_EXCLUSION_ZONE_PX["y_min"] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2
                        )


                    cv2.rectangle(
                        output_frame,
                        (CLAW_EXCLUSION_ZONE_PX["x_min"], CLAW_EXCLUSION_ZONE_PX["y_min"]),
                        (CLAW_EXCLUSION_ZONE_PX["x_max"], CLAW_EXCLUSION_ZONE_PX["y_max"]),
                        (255, 0, 255), 2
                    )
                if is_sel:
                    cv2.putText(
                        output_frame, "CIBLE",
                        (x1i, max(10, y1i - 80)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
                    )


    # ---------- Envois UDP ----------

    # Liste complète des objets -> interface Mac (toujours, même vide pour vider la liste)
    # Send even when empty so the Mac UI clears stale entries if all objects left the scene.
    sock.sendto(";".join(list_data).encode(), (MAC_IP, LIST_PORT))

    # Cible verrouillée uniquement -> contrôleur PID (valeurs filtrées)
    if selected_id is not None and selected_id in tracked_objects:
        t = tracked_objects[selected_id]
        if t.get('z_f') is not None:
           # Normalization
           # Clamp to [-1, 1] so the PID gains don't need to change with resolution.
           dx = float(np.clip(t['x_f']/(FRAME_W/2),-1.0,1.0))
           dy = float(np.clip(t['y_f']/(FRAME_H/2),-1.0,1.0))
           #depth error
           # ez > 0 means target is farther than desired → ROV should move forward.
           # ez < 0 means target is closer than desired → ROV should back up.
           ez =0.0
           if z_desired is not None:
                ez = float(np.clip((z_desired - t['z_f']) / Z_MAX, -1.0, 1.0))
        #json formatting
           packet = json.dumps({
            "vision_valid":           True,
            "confidence":             float(t.get('conf', 1.0)),
            "dx_normalized":          dx,
            "dy_normalized":          dy,
            "scale_error_normalized": ez,
            }).encode()

           sock.sendto(packet, (UDP_IP, UDP_PORT))
            #print("PID:", pid_msg)
    else:
        # no target
        # Send zeros so the PID controller knows there is nothing to track and
        # can hold position rather than acting on stale error values.
        packet = json.dumps({
            "vision_valid": False,
            "confidence":   0.0,
            "dx_normalized": 0.0,
            "dy_normalized": 0.0,
            "scale_error_normalized": 0.0,
        }).encode()
        sock.sendto(packet, (UDP_IP, UDP_PORT))
    # ---------- Video Output ----------
    t_vid = time.time()
    # Draw a crosshair at the image centre to mark the (0,0) reference point
    # for the coordinate system displayed in the overlay text.
    cv2.circle(output_frame, (int(FRAME_W/2), int(FRAME_H/2)), radius=6, color=(255, 0, 0), thickness=-1)
    cv2.line(output_frame, (0, int(FRAME_H/2)), (FRAME_W, int(FRAME_H/2)), (230, 0, 0), 1)
    cv2.putText(
        output_frame,
        "(x: 0 , y : 0 , z : 0)",
        (int(FRAME_W/2) + 10, int(FRAME_H/2) + 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
    )
    # Push the annotated frame into the GStreamer pipeline which encodes and
    # streams it to the Mac. PTS/DTS are computed from frame count so the
    # decoder can reconstruct the correct playback timeline.
    data = output_frame.tobytes()
    buf = Gst.Buffer.new_wrapped(data)
    # PTS based on frame_cnt rather than wall clock time: produces perfectly regular
    # timestamps for the H.264 decoder regardless of inference timing jitter.
    # This prevents the Mac decoder from freezing when two consecutive frames
    # arrive with an irregular time gap (e.g. after a slow HTP inference round).
    buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, FPS_OUT)
    buf.pts      = frame_cnt * buf.duration  # monotonically increasing, always regular
    buf.dts      = buf.pts                   # dts = pts for intra-only / low-latency streams
    appsrc.emit('push-buffer', buf)
    t_vid_out = time.time() - t_vid
    elapsed = time.time() - t_start_iter  # nécessite de capturer t_start_iter en début de boucle
    if elapsed < MIN_LOOP_DT:
        time.sleep(MIN_LOOP_DT - elapsed)

if Z_DIAG:
    zlog.stop()
# -------------------- Finish --------------------
elapsed_total = time.time() - t_start_loop
print(f"\n{'='*45}")
print(f"  RÉSUMÉ FINAL")
print(f"  Frames totales      : {frame_cnt}")
print(f"  Durée totale        : {elapsed_total:.1f} s")
print(f"  FPS moyen global    : {frame_cnt / elapsed_total:.1f}")
print(f"  FPS moyen inference : {1.0 / (t_total_inf / frame_cnt):.1f}")
print(f"  Latence moy. inf.   : {(t_total_inf / frame_cnt)*1000:.2f} ms")
print(f"{'='*45}")

appsrc.emit('end-of-stream')
pipeline.set_state(Gst.State.NULL)
cap.stop()
print("Done – video streamed through GStreamer")