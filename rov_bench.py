# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

#!/usr/bin/env python3
# =============================================================================
#  BENCHMARK PIPELINE VISION ROV — mesure segment par segment
#  Tous les timers utilisent time.perf_counter() pour la précision.
#  Les segments VIDEO STREAM et UDP sont mesurés à chaque frame (sans condition).
# =============================================================================

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

# ============================================================
#  CameraStream — lecture non-bloquante dans un thread dédié
#  NB : t_cap_out sera ~0 ms (non-bloquant par construction).
#  Pour mesurer la latence réelle de la caméra, commenter
#  CameraStream et utiliser cap_direct (voir MODE DIAGNOSTIC).
# ============================================================
class CameraStream:
    def __init__(self, src=2):
        self.cap = cv2.VideoCapture(src)
        # MJPG en premier — obligatoire avant width/height/fps
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        fourcc_val = self.cap.get(cv2.CAP_PROP_FOURCC)
        fps_val    = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[CameraStream] FOURCC={int(fourcc_val)}  FPS demandé={fps_val}")
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
MAC_IP           = "192.168.5.197"
UDP_IP           = "127.0.0.1"
UDP_PORT         = 17002
SELECT_PORT      = 17001
LIST_PORT        = 17003

FRAME_W       = 1280
FRAME_H       = 720
FPS_OUT       = 80
CONF_THRES    = 0.20
NMS_IOU_THRES = 0.50

ALPHA_XY = 0.4
ALPHA_Z  = 0.2

# -------------------- Load Delegate --------------------
delegate_options = {'backend_type': 'htp'}
delegate = tflite.load_delegate(DELEGATE_PATH, options=delegate_options)

# -------------------- Load YOLO --------------------
yolo_interpreter = tflite.Interpreter(
    model_path=MODEL_PATH_YOLO,
    experimental_delegates=[delegate]
)
yolo_interpreter.allocate_tensors()
in_det_yolo   = yolo_interpreter.get_input_details()
out_det_yolo  = yolo_interpreter.get_output_details()
in_h_yolo, in_w_yolo         = in_det_yolo[0]["shape"][1:3]
in_scale_yolo, in_zp_yolo    = in_det_yolo[0]["quantization"]
out_scale_yolo, out_zp_yolo  = out_det_yolo[0]["quantization"]
print(f"YOLO  — entrée : {in_w_yolo}x{in_h_yolo}  dtype: {in_det_yolo[0]['dtype']}")

# -------------------- Load MiDaS --------------------
midas_interpreter = tflite.Interpreter(
    model_path=MODEL_PATH_MIDAS,
    experimental_delegates=[delegate]
)
midas_interpreter.allocate_tensors()
in_det_midas   = midas_interpreter.get_input_details()
out_det_midas  = midas_interpreter.get_output_details()
in_h_midas, in_w_midas           = in_det_midas[0]["shape"][1:3]
in_scale_midas, in_zp_midas      = in_det_midas[0]["quantization"]
out_scale_midas, out_zp_midas    = out_det_midas[0]["quantization"]
print(f"MiDaS — entrée : {in_w_midas}x{in_h_midas}  dtype: {in_det_midas[0]['dtype']}")

scale_x_midas = in_w_midas / FRAME_W
scale_y_midas = in_h_midas / FRAME_H
Z_MAX = 255.0 * out_scale_midas

# -------------------- GStreamer Pipeline --------------------
# block=false : appsrc.emit() ne bloque plus si le buffer interne est plein.
# Permet de mesurer le temps réel de l'appel sans stall GStreamer.
pipeline = Gst.parse_launch(
    f'appsrc name=src '
    f'is-live=true '
    f'block=false '
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
# ---- Buffers réutilisables pour la quantization (évite 5 allocations/frame) ----
YOLO_SCALE  = 1.0 / (255.0 * in_scale_yolo)
MIDAS_SCALE = 1.0 / (255.0 * in_scale_midas)
buf_yolo_f32  = np.empty((in_h_yolo,  in_w_yolo,  3), np.float32)
buf_midas_f32 = np.empty((in_h_midas, in_w_midas, 3), np.float32)

# -------------------- Stats --------------------
frame_cnt    = 0
t_total_inf  = 0.0
t_start_loop = time.perf_counter()

# Accumulateurs pour moyennes glissantes (fenêtre = 30 frames)
acc = {k: 0.0 for k in [
    'cap', 'prep', 'inf', 'pp', 'depth_vis',
    'conf_filter', 'nms', 'track', 'boxing',
    'udp', 'vid', 'tobytes'
]}

# -------------------- Tracker centroïde --------------------
MAX_DIST_PIXELS = 400
MAX_LOST_FRAMES = 60
tracked_objects = {}
next_id = 0

def update_tracks(current_boxes, frame_cnt):
    global tracked_objects, next_id
    assignments = {}
    used_tracks = set()
    centroids = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in current_boxes]

    for idx, (cx, cy) in enumerate(centroids):
        best_id   = None
        best_dist = MAX_DIST_PIXELS
        for tid, tobj in tracked_objects.items():
            if tid in used_tracks:
                continue
            dx = cx - tobj['cx']
            dy = cy - tobj['cy']
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < best_dist:
                best_dist = dist
                best_id   = tid

        if best_id is not None:
            assignments[idx]                      = best_id
            tracked_objects[best_id]['cx']        = cx
            tracked_objects[best_id]['cy']        = cy
            tracked_objects[best_id]['bbox']      = current_boxes[idx]
            tracked_objects[best_id]['last_seen'] = frame_cnt
            used_tracks.add(best_id)
            tracked_objects[best_id]['x_f'] = (
                ALPHA_XY * (cx - FRAME_W/2)
                + (1 - ALPHA_XY) * tracked_objects[best_id].get('x_f', cx - FRAME_W/2)
            )
            tracked_objects[best_id]['y_f'] = (
                ALPHA_XY * (FRAME_H/2 - cy)
                + (1 - ALPHA_XY) * tracked_objects[best_id].get('y_f', FRAME_H/2 - cy)
            )
        else:
            assignments[idx] = next_id
            tracked_objects[next_id] = {
                'cx': cx, 'cy': cy,
                'bbox': current_boxes[idx],
                'last_seen': frame_cnt,
                'x_f': cx - FRAME_W/2,
                'y_f': FRAME_H/2 - cy,
                'z_f': None,
            }
            next_id += 1

    to_delete = [
        tid for tid, tobj in tracked_objects.items()
        if frame_cnt - tobj['last_seen'] > MAX_LOST_FRAMES
    ]
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

def poll_selection():
    global selected_id
    while True:
        try:
            data, _ = sel_sock.recvfrom(64)
        except BlockingIOError:
            break
        msg = data.decode(errors="ignore").strip()
        if msg == "clear":
            selected_id = None
        elif msg.startswith("select,"):
            try:
                selected_id = int(msg.split(",")[1])
            except (ValueError, IndexError):
                pass

# ============================================================
#  Main Loop
# ============================================================
while True:
    # --- Initialisation des timers à chaque frame ---
    t_cap_out = t_prep_out = t_inf_out = t_pp_out = 0.0
    t_depth_vis_out = t_conf_filter_out = t_nms_out = 0.0
    t_track_out = t_boxing_out = t_udp_out = t_vid_out = t_tobytes_out = 0.0

    # ── 1. Lecture caméra ──────────────────────────────────
    t0 = time.perf_counter()
    ok, frame = cap.read()
    t_cap_out = time.perf_counter() - t0
    if not ok or frame is None:
        continue
    frame_cnt += 1

    poll_selection()
    list_data = []
    no_x = FRAME_W / 2
    no_y = FRAME_H / 2

    # ── 2. Preprocessing ──────────────────────────────────
    t0 = time.perf_counter()
    cv2.resize(frame, (FRAME_W, FRAME_H), dst=frame_rs)

    resized_yolo = cv2.resize(frame_rs, (in_w_yolo, in_h_yolo))
    buf_yolo_f32[:] = resized_yolo
    buf_yolo_f32   *= YOLO_SCALE
    buf_yolo_f32   += in_zp_yolo
    np.clip(buf_yolo_f32, -128, 127, out=buf_yolo_f32)
    input_tensor_yolo[0] = buf_yolo_f32.astype(np.int8)

    resized_midas = cv2.resize(frame_rs, (in_w_midas, in_h_midas))
    resized_rgb   = cv2.cvtColor(resized_midas, cv2.COLOR_BGR2RGB)
    buf_midas_f32[:] = resized_rgb
    buf_midas_f32   *= MIDAS_SCALE
    buf_midas_f32   += in_zp_midas
    np.clip(buf_midas_f32, 0, 255, out=buf_midas_f32)
    input_tensor_midas[0] = buf_midas_f32.astype(np.uint8)
    t_prep_out = time.perf_counter() - t0

    # ── 3. Inférence NPU ──────────────────────────────────
    yolo_interpreter.set_tensor(in_det_yolo[0]['index'],  input_tensor_yolo)
    midas_interpreter.set_tensor(in_det_midas[0]['index'], input_tensor_midas)
    t0 = time.perf_counter()
    yolo_interpreter.invoke()
    midas_interpreter.invoke()
    t_inf_out = time.perf_counter() - t0
    t_total_inf += t_inf_out

    # ── 4. Postprocessing (déquantisation) ────────────────
    t0 = time.perf_counter()
    output_raw_yolo  = yolo_interpreter.get_tensor(out_det_yolo[0]['index'])[0]
    output_raw_midas = midas_interpreter.get_tensor(out_det_midas[0]['index'])[0]
    output   = (output_raw_yolo.astype(np.float32) - out_zp_yolo) * out_scale_yolo
    output   = output.transpose()
    depth_map = (output_raw_midas.astype(np.float32) - out_zp_midas) * out_scale_midas
    boxes        = output[:, :4]
    class_scores = output[:, 4:]
    scores       = np.max(class_scores, axis=1)
    classes      = np.argmax(class_scores, axis=1)
    t_pp_out = time.perf_counter() - t0

    # ── 5. Depth visualisation (debug) ────────────────────
    t0 = time.perf_counter()
    depth_norm    = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_color   = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)
    depth_display = cv2.resize(depth_color, (FRAME_W, FRAME_H))
    output_frame  = cv2.addWeighted(frame_rs, 0.5, depth_display, 0.5, 0)
    t_depth_vis_out = time.perf_counter() - t0

    # ── 6. Filtrage par confiance ──────────────────────────
    t0 = time.perf_counter()
    mask    = scores > CONF_THRES
    boxes   = boxes[mask]
    scores  = scores[mask]
    classes = classes[mask]
    t_conf_filter_out = time.perf_counter() - t0

    # ── 7. NMS ────────────────────────────────────────────
    t_nms_out   = 0.0
    t_track_out = 0.0
    t_boxing_out = 0.0

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
        boxes_cv2  = np.column_stack((x1, y1, x2-x1, y2-y1))

        t0 = time.perf_counter()
        idx_cv2 = cv2.dnn.NMSBoxes(
            boxes_cv2.tolist(), scores.tolist(), CONF_THRES, NMS_IOU_THRES
        )
        t_nms_out = time.perf_counter() - t0

        cx_int = cx.astype(np.int32)
        cy_int = cy.astype(np.int32)

        if len(idx_cv2):
            # ── 8. Tracking ───────────────────────────────
            t0 = time.perf_counter()
            indx = idx_cv2.flatten()
            current_boxes_list = [boxes_xyxy[i].tolist() for i in indx]
            track_assignments  = update_tracks(current_boxes_list, frame_cnt)
            t_track_out = time.perf_counter() - t0

            # ── 9. Z + drawing + liste Mac ────────────────
            t0 = time.perf_counter()
            for local_idx, i in enumerate(indx):
                cxi  = cx_int[i];  cyi = cy_int[i]
                x1i, y1i, x2i, y2i = boxes_xyxy[i]
                sc       = scores[i]
                track_id = track_assignments[local_idx]

                cx_m = np.clip(int(cxi * scale_x_midas), 0, in_w_midas - 1)
                cy_m = np.clip(int(cyi * scale_y_midas), 0, in_h_midas - 1)
                z_raw       = float(depth_map[cy_m, cx_m])
                z_corrected = Z_MAX - z_raw
                tid = track_assignments[local_idx]
                prev_z = tracked_objects[tid].get('z_f')
                tracked_objects[tid]['z_f'] = (
                    z_corrected if prev_z is None
                    else ALPHA_Z * z_corrected + (1 - ALPHA_Z) * prev_z
                )

                list_data.append(
                    f"{track_id},{int(cxi-no_x)},{int(no_y-cyi)},"
                    f"{z_corrected:.2f},{float(sc):.2f}"
                )

                is_sel = (track_id == selected_id)
                color  = (0, 255, 0) if z_corrected >= 850 else (0, 0, 255)
                thick  = 4 if is_sel else 2
                if is_sel:
                    color = (0, 255, 255)
                cv2.rectangle(output_frame, (x1i, y1i), (x2i, y2i), color, thick)
                cv2.circle(output_frame, (cxi, cyi), 6, (0, 0, 255), -1)
                cv2.putText(output_frame,
                    f"x:{int(cxi-no_x)} y:{int(no_y-cyi)} z:{z_corrected:.2f}",
                    (x1i, max(10, y1i-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(output_frame, f"{sc:.2f}",
                    (x1i, max(10, y1i-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(output_frame, f"Object:{track_id}",
                    (x1i, max(10, y1i-50)), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,255,200), 2)
                if is_sel:
                    cv2.putText(output_frame, "CIBLE",
                        (x1i, max(10, y1i-80)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            t_boxing_out = time.perf_counter() - t0

    # ── 10. UDP ───────────────────────────────────────────
    t0 = time.perf_counter()
    sock.sendto(";".join(list_data).encode(), (MAC_IP, LIST_PORT))
    if selected_id is not None and selected_id in tracked_objects:
        t = tracked_objects[selected_id]
        if t.get('z_f') is not None:
            pid_msg = f"{selected_id},{t['x_f']:.1f},{t['y_f']:.1f},{t['z_f']:.1f}"
            sock.sendto(pid_msg.encode(), (UDP_IP, UDP_PORT))
    t_udp_out = time.perf_counter() - t0

    # ── 11. Overlay + tobytes + GStreamer ─────────────────
    cv2.circle(output_frame, (int(FRAME_W/2), int(FRAME_H/2)), 6, (255, 0, 0), -1)
    cv2.line(output_frame, (0, int(FRAME_H/2)), (FRAME_W, int(FRAME_H/2)), (230,0,0), 1)
    cv2.putText(output_frame, "(x:0 y:0 z:0)",
        (int(FRAME_W/2)+10, int(FRAME_H/2)+10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    t0 = time.perf_counter()
    raw_bytes = output_frame.tobytes()
    t_tobytes_out = time.perf_counter() - t0

    t0 = time.perf_counter()
    buf = Gst.Buffer.new_allocate(None, len(raw_bytes), None)
    buf.fill(0, raw_bytes)
    buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, FPS_OUT)
    buf.pts = buf.dts = frame_cnt * buf.duration
    appsrc.emit('push-buffer', buf)
    t_vid_out = time.perf_counter() - t0

    # ── Accumulateurs ─────────────────────────────────────
    acc['cap']         += t_cap_out
    acc['prep']        += t_prep_out
    acc['inf']         += t_inf_out
    acc['pp']          += t_pp_out
    acc['depth_vis']   += t_depth_vis_out
    acc['conf_filter'] += t_conf_filter_out
    acc['nms']         += t_nms_out
    acc['track']       += t_track_out
    acc['boxing']      += t_boxing_out
    acc['udp']         += t_udp_out
    acc['tobytes']     += t_tobytes_out
    acc['vid']         += t_vid_out

    # ── Stats every 30 frames ─────────────────────────────
    if frame_cnt % 30 == 0:
        elapsed = time.perf_counter() - t_start_loop
        fps_g   = frame_cnt / elapsed
        fps_inf = 1.0 / (t_total_inf / frame_cnt)
        n = 30  # fenêtre de moyennage

        def ms(key): return acc[key] / n * 1000

        # Total mesuré = somme de tous les segments
        total_measured = sum(acc[k] for k in acc) / n * 1000
        frame_budget   = 1000.0 / fps_g
        fantome        = frame_budget - total_measured

        print(f"\n{'='*52}")
        print(f"  BENCHMARK @ frame {frame_cnt}")
        print(f"{'='*52}")
        print(f"  FPS global (loop)   : {fps_g:.1f}  ({frame_budget:.1f} ms/frame)")
        print(f"  FPS inference only  : {fps_inf:.1f}")
        print(f"  Latence moy. inf.   : {t_total_inf/frame_cnt*1000:.2f} ms")
        print(f"{'─'*52}")
        print(f"  [1] Camera read     : {ms('cap'):.3f} ms")
        print(f"  [2] Preprocessing   : {ms('prep'):.3f} ms")
        print(f"  [3] Inference (NPU) : {ms('inf'):.3f} ms")
        print(f"  [4] Postprocess     : {ms('pp'):.3f} ms")
        print(f"  [5] Depth visual    : {ms('depth_vis'):.3f} ms")
        print(f"  [6] Conf filter     : {ms('conf_filter'):.3f} ms")
        print(f"  [7] NMS             : {ms('nms'):.3f} ms  (0 si aucune det.)")
        print(f"  [8] Tracking        : {ms('track'):.3f} ms  (0 si aucune det.)")
        print(f"  [9] Boxing+Z+draw   : {ms('boxing'):.3f} ms  (0 si aucune det.)")
        print(f"  [10] UDP send       : {ms('udp'):.3f} ms")
        print(f"  [11] tobytes()      : {ms('tobytes'):.3f} ms")
        print(f"  [12] GStreamer push : {ms('vid'):.3f} ms")
        print(f"{'─'*52}")
        print(f"  Total mesuré        : {total_measured:.2f} ms")
        print(f"  Budget frame        : {frame_budget:.2f} ms")
        print(f"  Temps fantôme       : {fantome:.2f} ms  ← à identifier")
        print(f"{'─'*52}")
        print(f"  Depth min/max       : {depth_map.min():.2f} / {depth_map.max():.2f}")
        print(f"  Cible verrouillee   : {selected_id}")
        print(f"{'='*52}\n")

        # Remise à zéro des accumulateurs
        for k in acc:
            acc[k] = 0.0

# -------------------- Finish --------------------
elapsed_total = time.perf_counter() - t_start_loop
print(f"\n{'='*52}")
print(f"  RÉSUMÉ FINAL")
print(f"  Frames totales      : {frame_cnt}")
print(f"  Durée totale        : {elapsed_total:.1f} s")
print(f"  FPS moyen global    : {frame_cnt / elapsed_total:.1f}")
print(f"  FPS moyen inference : {1.0 / (t_total_inf / frame_cnt):.1f}")
print(f"  Latence moy. inf.   : {t_total_inf / frame_cnt * 1000:.2f} ms")
print(f"{'='*52}")

appsrc.emit('end-of-stream')
pipeline.set_state(Gst.State.NULL)
cap.stop()
print("Done.")