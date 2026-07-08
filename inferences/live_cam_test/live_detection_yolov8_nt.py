# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

#!/usr/bin/env python3

import os

# Doit être set AVANT tout import tflite/delegate
os.environ["ADSP_LIBRARY_PATH"] = (
    "/opt/qcom/qirp-sdk/lib/aarch64-oe-linux-gcc11.2;"
    "/opt/qcom/qirp-sdk/lib/hexagon-v73/unsigned;"
    + os.environ.get("ADSP_LIBRARY_PATH", "")
)

import time
import cv2
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import ai_edge_litert.interpreter as tflite

# Initialize GStreamer
Gst.init(None)
import threading

class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.frame  = None
        self.ok     = False
        self.lock   = threading.Lock()
        self.running = True

        # Lire une première frame
        self.ok, self.frame = self.cap.read()

        # Lancer le thread
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
            return self.ok, self.frame.copy() if self.frame is not None else (False, None)

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
# -------------------- Parameters --------------------
MODEL_PATH    = "deepbox-tflite-float/yolov8n_saved_model/int8/yolov8n_full_integer_quant.tflite"
VIDEO_IN      = "13759780_2160_3840_60fps.mp4"
DELEGATE_PATH = "/opt/qcom/qirp-sdk/lib/aarch64-oe-linux-gcc11.2/libQnnTFLiteDelegate.so"
MAC_IP        = "192.168.5.41"

FRAME_W       = 1600
FRAME_H       = 900
FPS_OUT       = 30
CONF_THRES    = 0.3
NMS_IOU_THRES = 0.50

# -------------------- Load Model --------------------
delegate_options = {'backend_type': 'htp'}
delegate = tflite.load_delegate(DELEGATE_PATH, options=delegate_options)

interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=[delegate]
)
interpreter.allocate_tensors()

in_det  = interpreter.get_input_details()
out_det = interpreter.get_output_details()

print("Entrée :", in_det)
print("Sortie :", out_det)
print("dtype entrée :", in_det[0]["dtype"])
print("dtype sortie :", out_det[0]["dtype"])
print("quantization :", in_det[0]["quantization"])

in_h, in_w = in_det[0]["shape"][1:3]

# Quantization params
in_scale, in_zp   = in_det[0]["quantization"]
out_scale, out_zp = out_det[0]["quantization"]

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
cap = cv2.VideoCapture(2)  # 0 = première caméra détectée
# Forcer la résolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
# Preallocate frame buffers
frame_rs     = np.empty((FRAME_H, FRAME_W, 3), np.uint8)
input_tensor = np.empty((1, in_h, in_w, 3), np.int8)

# -------------------- Stats --------------------
frame_cnt    = 0
t_total_inf  = 0.0
t_start_loop = time.time()

# -------------------- Main Loop --------------------
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_cnt += 1

    # ---------- Preprocessing ----------
    cv2.resize(frame, (FRAME_W, FRAME_H), dst=frame_rs)
    resized = cv2.resize(frame_rs, (in_w, in_h))

    # float32 [0,1] → int8 [-128, 127]
    input_tensor[0] = (
        resized.astype(np.float32) / 255.0 / in_scale + in_zp
    ).clip(-128, 127).astype(np.int8)

    # ---------- Inference ----------
    interpreter.set_tensor(in_det[0]['index'], input_tensor)

    t0 = time.time()
    interpreter.invoke()
    t_inf = time.time() - t0
    t_total_inf += t_inf

    # ---------- Postprocessing ----------
    output_raw = interpreter.get_tensor(out_det[0]['index'])[0]

    # Dequantize → float32
    output = (output_raw.astype(np.float32) - out_zp) * out_scale

    # [84, 8400] → [8400, 84]
    output = output.transpose()

    # Boxes (cx cy w h) normalisées [0,1]
    boxes        = output[:, :4]
    class_scores = output[:, 4:]

    scores  = np.max(class_scores, axis=1)
    classes = np.argmax(class_scores, axis=1)

    

    # ---------- Stats toutes les 30 frames ----------
    if frame_cnt % 30 == 0:
        elapsed_loop = time.time() - t_start_loop
        fps_global   = frame_cnt / elapsed_loop
        avg_inf_ms   = (t_total_inf / frame_cnt) * 1000
        fps_inf      = 1.0 / (t_total_inf / frame_cnt)

        print(f"\n{'='*45}")
        print(f"  Stats @ frame {frame_cnt}")
        print(f"  FPS global (loop)   : {fps_global:.1f}")
        print(f"  FPS inference only  : {fps_inf:.1f}")
        print(f"  Latence moy. inf.   : {avg_inf_ms:.2f} ms")
        print(f"  Latence last frame  : {t_inf*1000:.2f} ms")
        print(f"{'='*45}\n")

    # ---------- Confidence filtering ----------
    mask    = scores > CONF_THRES
    boxes   = boxes[mask]
    scores  = scores[mask]
    classes = classes[mask]

    if len(boxes):
        # YOLO format: cx cy w h normalisés → pixels display
        cx = boxes[:, 0] * FRAME_W
        cy = boxes[:, 1] * FRAME_H
        w  = boxes[:, 2] * FRAME_W
        h  = boxes[:, 3] * FRAME_H

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.int32)

        # OpenCV NMS format [x, y, w, h]
        boxes_cv2 = np.column_stack((
            boxes_xyxy[:, 0],
            boxes_xyxy[:, 1],
            boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
            boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        ))

        idx_cv2 = cv2.dnn.NMSBoxes(
            boxes_cv2.tolist(),
            scores.tolist(),
            CONF_THRES,
            NMS_IOU_THRES
        )

        if len(idx_cv2):
            for i in idx_cv2.flatten():
                x1i, y1i, x2i, y2i = boxes_xyxy[i]
                sc = scores[i]

                cv2.rectangle(
                    frame_rs,
                    (x1i, y1i),
                    (x2i, y2i),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame_rs,
                    f"{sc:.2f}",
                    (x1i, max(10, y1i - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    # ---------- Video Output ----------
    data = frame_rs.tobytes()
    buf  = Gst.Buffer.new_allocate(None, len(data), None)
    buf.fill(0, data)
    buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, FPS_OUT)
    buf.pts = buf.dts = frame_cnt * buf.duration
    appsrc.emit('push-buffer', buf)

# -------------------- Finish --------------------
# Stats finales
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
cap.release()
print("Done – video streamed through GStreamer")