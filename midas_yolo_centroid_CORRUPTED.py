# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

#!/usr/bin/env python3

import os

os.environ["ADSP_LIBRARY_PATH"] = (
    "/opt/qcom/qirp-sdk/lib/aarch64-oe-linux-gcc11.2;"
    "/opt/qcom/qirp-sdk/lib/hexagon-v73/unsigned;"
    + os.environ.get("ADSP_LIBRARY_PATH", "")
)
import socket
import time
import cv2
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import ai_edge_litert.interpreter as tflite
import threading

Gst.init(None)

class CameraStream:
    def __init__(self, src=2):
        self.cap = cv2.VideoCapture(src)
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

    def read(self):
        with self.lock:
            return self.ok, self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# -------------------- Parameters --------------------
MODEL_PATH_YOLO  = "models/deepbox-tflite-float/yolov8n_saved_model/int8/yolov8n_full_integer_quant.tflite"
MODEL_PATH_MIDAS = "/root/ai-rov/src/models/midas-tflite-w8a8/midas.tflite"
DELEGATE_PATH    = "/opt/qcom/qirp-sdk/lib/aarch64-oe-linux-gcc11.2/libQnnTFLiteDelegate.so"
MAC_IP           = "192.168.5.171"
UDP_IP           = "127.0.0.1"
UDP_PORT         = 17001

FRAME_W       = 1600
FRAME_H       = 900
FPS_OUT       = 60
CONF_THRES    = 0.20

NMS_IOU_THRES = 0.50

ALPHA_XY = 0.4   # réactivité position (0 = figé, 1 = brut)
ALPHA_Z  = 0.2   # réactivité profondeur (plus faible car MiDaS est plus bruité)

# -------------------- Load Delegate --------------------
delegate_options = {'backend_type': 'htp'}
delegate = tflite.load_delegate(DELEGATE_PATH, options=delegate_options)

# -------------------- Load YOLO --------------------
yolo_interpreter = tflite.Interpreter(
    model_path=MODEL_PATH_YOLO,
    experimental_delegates=[delegate]
)
yolo_interpreter.allocate_tensors()

in_det_yolo  = yolo_interpreter.get_input_details()
out_det_yolo = yolo_interpreter.get_output_details()

in_h_yolo, in_w_yolo       = in_det_yolo[0]["shape"][1:3]
in_scale_yolo, in_zp_yolo  = in_det_yolo[0]["quantization"]
out_scale_yolo, out_zp_yolo = out_det_yolo[0]["quantization"]

print(f"YOLO  — entrée : {in_w_yolo}x{in_h_yolo}  dtype: {in_det_yolo[0]['dtype']}")

# -------------------- Load MiDaS --------------------
midas_interpreter = tflite.Interpreter(
    model_path=MODEL_PATH_MIDAS,
    experimental_delegates=[delegate]
)
midas_interpreter.allocate_tensors()

in_det_midas  = midas_interpreter.get_input_details()
out_det_midas = midas_interpreter.get_output_details()

in_h_midas, in_w_midas         = in_det_midas[0]["shape"][1:3]
in_scale_midas, in_zp_midas    = in_det_midas[0]["quantization"]
out_scale_midas, out_zp_midas  = out_det_midas[0]["quantization"]

print(f"MiDaS — entrée : {in_w_midas}x{in_h_midas}  dtype: {in_det_midas[0]['dtype']}")

# Facteurs d'échelle pour convertir cx/cy → espace MiDaS 256×256
scale_x_midas = in_w_midas / FRAME_W
scale_y_midas = in_h_midas / FRAME_H
Z_MAX = 255.0 * out_scale_midas

# -------------------- GStreamer Pipeline --------------------
pipeline = Gst.parse_launch(
    f'appsrc name=src '
    f'is-live=true '
    f'block=true '
    f'format=time '
    f'caps=video/x-raw,format=BGR,width={FRAME_W},height={FRAME_H},framerate={FPS_OUT}/1 '
    '! videoconvert '
    '! qtic2venc '
    '! h264parse '
    '! mpegtsmux '
    f'! udpsink host={MAC_IP} port=5000'
)
appsrc = pipeline.get_by_name('src')
pipeline.set_state(Gst.State.PLAYING)

# -------------------- Video Input --------------------
cap = CameraStream(src=2)

# Preallocate buffers
frame_rs           = np.empty((FRAME_H, FRAME_W, 3), np.uint8)
input_tensor_yolo  = np.empty((1, in_h_yolo,  in_w_yolo,  3), np.int8)
input_tensor_midas = np.empty((1, in_h_midas, in_w_midas, 3), np.uint8)

# -------------------- Stats --------------------
frame_cnt    = 0
t_total_inf  = 0.0
t_start_loop = time.time()

# -------------------- Paramètres tracker --------------------
MAX_DIST_PIXELS  = 300   # distance max entre deux frames pour même objet
MAX_LOST_FRAMES  = 60    # frames avant suppression du track

# -------------------- Tracker centroïde --------------------
tracked_objects = {}  # {track_id: {'cx': , 'cy': , 'bbox': , 'last_seen': }}
next_id = 0

def update_tracks(current_boxes, frame_cnt):
    """
    Tracker basé sur distance euclidienne entre centroïdes.
    Beaucoup plus robuste que IoU quand les boîtes sont instables.
    """
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
        best_dist = MAX_DIST_PIXELS  # seuil en pixels
       
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
    to_delete = [
        tid for tid, tobj in tracked_objects.items()
        if frame_cnt - tobj['last_seen'] > MAX_LOST_FRAMES
    ]
    for tid in to_delete:
        del tracked_objects[tid]

    return assignments

# -------------------- UDP Socket --------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -------------------- Main Loop --------------------
while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        continue
    frame_cnt += 1

    # Center of frame as new origin
    no_x = FRAME_W / 2
    no_y = FRAME_H / 2

    # ---------- Preprocessing ----------
    cv2.resize(frame, (FRAME_W, FRAME_H), dst=frame_rs)

    # YOLO — BGR
    resized_yolo = cv2.resize(frame_rs, (in_w_yolo, in_h_yolo))
    input_tensor_yolo[0] = (
        resized_yolo.astype(np.float32) /