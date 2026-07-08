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

IOU_TRACK_THRESH = 0.35 # minimum overlap to consider same object
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

# -------------------- IoU tracking --------------------
tracked_objects  = {}   # {track_id: {'bbox': [...], 'last_seen': frame_cnt}}
next_id          = 0


def compute_iou(box_1, box_2):
    # box = [x1, y1, x2, y2]
    # Intersection top-left
    ix1 = max(box_1[0], box_2[0])
    iy1 = max(box_1[1], box_2[1])
    # Intersection bottom-right
    ix2 = min(box_1[2], box_2[2])
    iy2 = min(box_1[3], box_2[3])
    # Intersection area (0 if no overlap)
    ia = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    # Union
    area1 = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    area2 = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])
    union = area1 + area2 - ia
    return ia / union if union > 0 else 0.0

def update_tracks(current_boxes, frame_cnt):
    # Associates each current box to a tracked object, returns {idx: track_id}
    global tracked_objects, next_id
    assignments = {}    # {idx_current: track_id}
    used_tracks = set()

    for idx, box in enumerate(current_boxes):
        best_id  = None
        best_iou = IOU_TRACK_THRESH

        for tid, tobj in tracked_objects.items():
            if tid in used_tracks:
                continue
            iou = compute_iou(box, tobj['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_id  = tid

        if best_id is not None:
            # Known object — update position
            assignments[idx]                      = best_id
            tracked_objects[best_id]['bbox']      = box
            tracked_objects[best_id]['last_seen'] = frame_cnt
            used_tracks.add(best_id)
        else:
            # New object — create track
            assignments[idx] = next_id
            tracked_objects[next_id] = {
                'bbox':      box,
                'last_seen': frame_cnt
            }
            next_id += 1

    # Clean up lost tracks (not seen for > 30 frames)
    to_delete = [
        tid for tid, tobj in tracked_objects.items()
        if frame_cnt - tobj['last_seen'] > 60
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
        resized_yolo.astype(np.float32) / 255.0 / in_scale_yolo + in_zp_yolo
    ).clip(-128, 127).astype(np.int8)

    # MiDaS — RGB
    resized_midas = cv2.resize(frame_rs, (in_w_midas, in_h_midas))
    resized_rgb   = cv2.cvtColor(resized_midas, cv2.COLOR_BGR2RGB)
    input_tensor_midas[0] = (
        resized_rgb.astype(np.float32) / 255.0 / in_scale_midas + in_zp_midas
    ).clip(0, 255).astype(np.uint8)

    # ---------- Inference ----------
    yolo_interpreter.set_tensor(in_det_yolo[0]['index'],  input_tensor_yolo)
    midas_interpreter.set_tensor(in_det_midas[0]['index'], input_tensor_midas)

    t0 = time.time()
    yolo_interpreter.invoke()
    midas_interpreter.invoke()
    t_inf = time.time() - t0
    t_total_inf += t_inf

    # ---------- Postprocessing ----------
    output_raw_yolo  = yolo_interpreter.get_tensor(out_det_yolo[0]['index'])[0]
    output_raw_midas = midas_interpreter.get_tensor(out_det_midas[0]['index'])[0]

    # Dequantize YOLO
    output = (output_raw_yolo.astype(np.float32) - out_zp_yolo) * out_scale_yolo
    output = output.transpose()   # [84, 8400] → [8400, 84]

    # Dequantize MiDaS → depth map
    depth_map = (output_raw_midas.astype(np.float32) - out_zp_midas) * out_scale_midas

    boxes        = output[:, :4]
    class_scores = output[:, 4:]
    scores       = np.max(class_scores, axis=1)
    classes      = np.argmax(class_scores, axis=1)

    # ---------- Depth visualisation ----------
    depth_norm    = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_color   = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)
    depth_display = cv2.resize(depth_color, (FRAME_W, FRAME_H))
    output_frame  = cv2.addWeighted(frame_rs, 0.5, depth_display, 0.5, 0)

    # ---------- Stats every 30 frames ----------
    if frame_cnt % 30 == 0:
        elapsed_loop = time.time() - t_start_loop
        print(f"\n{'='*45}")
        print(f"  Stats @ frame {frame_cnt}")
        print(f"  FPS global (loop)   : {frame_cnt / elapsed_loop:.1f}")
        print(f"  FPS inference only  : {1.0 / (t_total_inf / frame_cnt):.1f}")
        print(f"  Latence moy. inf.   : {(t_total_inf / frame_cnt)*1000:.2f} ms")
        print(f"  Latence last frame  : {t_inf*1000:.2f} ms")
        print(f"  Depth min/max       : {depth_map.min():.3f} / {depth_map.max():.3f}")
        print(f"{'='*45}\n")

    # ---------- Confidence filtering ----------
    mask    = scores > CONF_THRES
    boxes   = boxes[mask]
    scores  = scores[mask]
    classes = classes[mask]

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

        idx_cv2 = cv2.dnn.NMSBoxes(
            boxes_cv2.tolist(), scores.tolist(), CONF_THRES, NMS_IOU_THRES
        )

        cx_int = cx.astype(np.int32)
        cy_int = cy.astype(np.int32)

        if len(idx_cv2):
            indx = idx_cv2.flatten()

            # Build current boxes list and update tracker
            current_boxes_list = [boxes_xyxy[i].tolist() for i in indx]
            track_assignments  = update_tracks(current_boxes_list, frame_cnt)

            objects_data = []   # accumulate all objects for single UDP send

            for local_idx, i in enumerate(indx):
                cxi                  = cx_int[i]
                cyi                  = cy_int[i]
                x1i, y1i, x2i, y2i  = boxes_xyxy[i]
                sc                   = scores[i]
                track_id             = track_assignments[local_idx]

                # --- Z extraction (MiDaS native space) ---
                cx_m = np.clip(int(cxi * scale_x_midas), 0, in_w_midas - 1)
                cy_m = np.clip(int(cyi * scale_y_midas), 0, in_h_midas - 1)
                z_raw       = float(depth_map[cy_m, cx_m])
                z_corrected = Z_MAX - z_raw   # large = far, small = close

                # Accumulate for UDP
                objects_data.append(
                    f"{track_id},{int(cxi - no_x)},{int(no_y - cyi)},{z_corrected:.2f}"
                )

                # --- Drawing ---
                color = (0, 255, 0) if z_corrected >= 850 else (0, 0, 255)
                cv2.rectangle(output_frame, (x1i, y1i), (x2i, y2i), color, 2)
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

            # Single UDP send per frame — format: "id,x,y,z;id,x,y,z;..."
            if objects_data:
                message = ";".join(objects_data)
                sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
                #print("UDP:", message)

    # ---------- Video Output ----------
    cv2.circle(output_frame, (int(FRAME_W/2), int(FRAME_H/2)), radius=6, color=(255, 0, 0), thickness=-1)
    cv2.line(output_frame, (0, int(FRAME_H/2)), (FRAME_W, int(FRAME_H/2)), (230, 0, 0), 1)
    cv2.putText(
        output_frame,
        "(x: 0 , y : 0 , z : 0)",
        (int(FRAME_W/2) + 10, int(FRAME_H/2) + 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
    )
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
cap.stop()
print("Done – video streamed through GStreamer")