# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

#!/usr/bin/env python3

import os
import time
import cv2
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import ai_edge_litert.interpreter as tflite

# Initialize GStreamer
Gst.init(None)

# -------------------- Parameters --------------------
MODEL_PATH    = "midas-tflite-w8a8/midas.tflite"
VIDEO_IN      = "13759780_2160_3840_60fps.mp4"
DELEGATE_PATH = "/opt/qcom/qirp-sdk/lib/aarch64-oe-linux-gcc11.2/libQnnTFLiteDelegate.so"
MAC_IP        = "192.168.5.144"

FRAME_W       = 1600
FRAME_H       = 900
FPS_OUT       = 30

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
print("quantization entrée :", in_det[0]["quantization"])
print("quantization sortie :", out_det[0]["quantization"])

# MiDaS v2 attend 256x256x3
in_h, in_w = in_det[0]["shape"][1:3]
print(f"Résolution modèle : {in_w}x{in_h}")

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
cap = cv2.VideoCapture(VIDEO_IN)

# Preallocate buffers
frame_rs     = np.empty((FRAME_H, FRAME_W, 3), np.uint8)
input_tensor = np.empty((1, in_h, in_w, 3), np.uint8)

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

    # MiDaS attend RGB
    resized = cv2.resize(frame_rs, (in_w, in_h))
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # float32 [0,1] → int8 [-128, 127]
    input_tensor[0] = (
        resized_rgb.astype(np.float32) / 255.0 / in_scale + in_zp
    ).clip(0, 255).astype(np.uint8)

    # ---------- Inference ----------
    interpreter.set_tensor(in_det[0]['index'], input_tensor)

    t0 = time.time()
    interpreter.invoke()
    t_inf = time.time() - t0
    t_total_inf += t_inf

    # ---------- Postprocessing ----------
    output_raw = interpreter.get_tensor(out_det[0]['index'])[0]

    # Dequantize → float32
    depth_map = (output_raw.astype(np.float32) - out_zp) * out_scale
    # depth_map shape : (256, 256) — valeurs relatives (pas en mètres)

    # ---------- Visualisation ----------
    # Normaliser [0, 255] pour affichage
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)

    # Colormap MAGMA (plus lisible que GRAY pour la profondeur)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

    # Redimensionner à la taille d'affichage
    depth_display = cv2.resize(depth_color, (FRAME_W, FRAME_H))

    # Overlay : depth colorée en transparence sur la frame originale
    output_frame = cv2.addWeighted(frame_rs, 0.5, depth_display, 0.5, 0)

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
        print(f"  Depth min/max       : {depth_map.min():.3f} / {depth_map.max():.3f}")
        print(f"{'='*45}\n")

    # ---------- Video Output ----------
    data = output_frame.tobytes()
    buf  = Gst.Buffer.new_allocate(None, len(data), None)
    buf.fill(0, data)
    buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, FPS_OUT)
    buf.pts = buf.dts = frame_cnt * buf.duration
    appsrc.emit('push-buffer', buf)

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
cap.release()
print("Done – video streamed through GStreamer")