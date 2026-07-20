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
import json
import socket
import time
import cv2
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import ai_edge_litert.interpreter as tflite
import threading
Z_DIAG = False
if Z_DIAG:

    from z_diagnostic import sample_depth, ZDiagnosticLogger
Gst.init(None)

class CameraStream:
    def __init__(self, src=2):
        self.seq = 0
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        #self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
         # --- Verification : on refuse de demarrer si le mode negocie est lent ---
        real_w   = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        real_h   = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        real_fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc   = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fcc_str  = "".join(chr((fourcc >> 8*i) & 0xFF) for i in range(4))
        print(f"[CAM] negocie : {real_w:.0f}x{real_h:.0f} @ {real_fps:.0f} fps  ({fcc_str})")
        if real_fps < 25:
            raise RuntimeError(
                f"[CAM] FPS negocie trop bas ({real_fps:.0f}). "
                f"Mode lent detecte — verifier resolution/cable USB."
            )
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
            return self.ok, self.frame.copy(), self.seq


    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# -------------------- Parameters --------------------
MODEL_PATH_YOLO  = "models/deepbox-tflite-float/yolov8n_saved_model/int8/yolov8n_full_integer_quant.tflite"
MODEL_PATH_MIDAS = "/root/ai-rov/src/models/midas-tflite-w8a8/midas.tflite"
DELEGATE_PATH    = "/opt/qcom/qirp-sdk/lib/aarch64-oe-linux-gcc11.2/libQnnTFLiteDelegate.so"
MAC_IP           = "192.168.5.65"
UDP_IP           = "127.0.0.1"
UDP_PORT         = 17002
SELECT_PORT      = 17001         # écoute des sélections (Mac -> ROV)
LIST_PORT        = 17003          # diffusion de la liste d'objets (ROV -> Mac)
VIDEO_IN      = "13759780_2160_3840_60fps.mp4"

TEST_VIDEO = False

FRAME_W       = 640 #correction 
FRAME_H       = 480
FPS_OUT       = 30
CONF_THRES    = 0.25

MIN_CY_VALID  = 60      # NOUVEAU : rejette les détections trop hautes dans l'image (fantômes de surface)

NMS_IOU_THRES = 0.50

ALPHA_XY =  0.15     # réactivité position (0 = figé, 1 = brut)
ALPHA_Z  = 0.05   # réactivité profondeur (plus faible car MiDaS est plus bruité)
ALPHA_CXY_MIDAS = 0.25   # NOUVEAU : lissage du centroïde dans l'espace MiDaS

DEBUG_DEPTH = False # Midas Dbug mode

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
    f'block=false ' #correction 
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
if not TEST_VIDEO:
    cap = CameraStream(src=2)
else:
    cap= cv2.VideoCapture(VIDEO_IN)
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
t_start_loop = time.time()

# -------------------- Paramètres tracker --------------------
MAX_DIST_PIXELS  = 450 # distance max entre deux frames pour même objet
MAX_LOST_FRAMES  = 120    # frames avant suppression du track

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

# -------------------- UDP Sockets --------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   # envois PID + liste Mac

# Socket de réception des sélections (non bloquante : ne ralentit pas la boucle)
sel_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sel_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sel_sock.bind(("0.0.0.0", SELECT_PORT))
sel_sock.setblocking(False)

selected_id = None   # track_id verrouillé pour le PID (None = aucune cible)

z_desired = None # lock the detected z
def poll_selection():
    """Vide la socket de sélection et met à jour selected_id (sélection par ID)."""
    global selected_id, z_desired
    while True:
        try:
            data, _ = sel_sock.recvfrom(64)
        except BlockingIOError:
            break                                   # plus rien à lire
        msg = data.decode(errors="ignore").strip()
        if msg == "clear":
            selected_id = None
            z_desired = None
        elif msg.startswith("select,"):
            try:
                new_id = int(msg.split(",")[1])
                selected_id = new_id
                if new_id in tracked_objects and tracked_objects[new_id].get('z_f'):
                        z_desired = tracked_objects[new_id]['z_f']
            except (ValueError, IndexError):
                pass
if Z_DIAG:
    zlog = ZDiagnosticLogger("z_diag.csv")
# -------------------- Main Loop --------------------
last_seq = -1 
loop_cnt =0 
while True:
    loop_cnt+=1
    if cap.seq == last_seq:
        time.sleep(0.002)
        continue
    ok, frame, seq = cap.read()
    if not ok or frame is None:
        continue
    """ if seq == last_seq  : 
        time.sleep(0.0001)
        continue """
    last_seq = seq 
    frame_cnt += 1
    

    poll_selection()          # met à jour selected_id selon les clics du Mac
    list_data = []            # liste complète des objets -> interface Mac

    # Center of frame as new origin
    no_x = FRAME_W / 2
    no_y = FRAME_H / 2

    # ---------- Preprocessing ----------
    cv2.resize(frame, (FRAME_W, FRAME_H), dst=frame_rs)

    # YOLO — BGR
    resized_yolo = cv2.resize(frame_rs, (in_w_yolo, in_h_yolo))
    buf_yolo_f32[:] = resized_yolo
    buf_yolo_f32   *= YOLO_SCALE
    buf_yolo_f32   += in_zp_yolo
    np.clip(buf_yolo_f32, -128, 127, out=buf_yolo_f32)
    input_tensor_yolo[0] = buf_yolo_f32.astype(np.int8)

    # MiDaS — RGB
    resized_midas = cv2.resize(frame_rs, (in_w_midas, in_h_midas))
    resized_rgb   = cv2.cvtColor(resized_midas, cv2.COLOR_BGR2RGB)
    buf_midas_f32[:] = resized_rgb
    buf_midas_f32   *= MIDAS_SCALE
    buf_midas_f32   += in_zp_midas
    np.clip(buf_midas_f32, 0, 255, out=buf_midas_f32)
    input_tensor_midas[0] = buf_midas_f32.astype(np.uint8)

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
        print(f"  FPS/s : {frame_cnt / elapsed_loop:.1f}")  # ~7-9

        print(f"  FPS inference only  : {1.0 / (t_total_inf / frame_cnt):.1f}")
        print(f"  Latence moy. inf.   : {(t_total_inf / frame_cnt)*1000:.2f} ms")
        print(f"  Latence last frame  : {t_inf*1000:.2f} ms")
        print(f"  Depth min/max       : {depth_map.min():.3f} / {depth_map.max():.3f}")
        print(f"  Cible verrouillee   : {selected_id}")
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
        cx_int = cx.astype(np.int32)
        cy_int = cy.astype(np.int32)

        # NOUVEAU : rejet des fantômes en haut de frame
        valid_zone = cy_int >= MIN_CY_VALID
        boxes_xyxy  = boxes_xyxy[valid_zone]
        boxes_cv2   = boxes_cv2[valid_zone]
        scores      = scores[valid_zone]
        cx_int      = cx_int[valid_zone]
        cy_int      = cy_int[valid_zone]
        cx          = cx[valid_zone]
        cy          = cy[valid_zone]
        idx_cv2 = cv2.dnn.NMSBoxes(
            boxes_cv2.tolist(), scores.tolist(), CONF_THRES, NMS_IOU_THRES
        )

        

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
              
                # --- Z extraction (MiDaS native space) ---
                cx_m_raw = np.clip(cxi * scale_x_midas, 0, in_w_midas - 1)
                cy_m_raw = np.clip(cyi * scale_y_midas, 0, in_h_midas - 1)

                tid_for_smoothing = track_assignments[local_idx]
                prev_cx_m = tracked_objects[tid_for_smoothing].get('cx_m_f')
                prev_cy_m = tracked_objects[tid_for_smoothing].get('cy_m_f')

                cx_m_f = cx_m_raw if prev_cx_m is None else (
                    ALPHA_CXY_MIDAS * cx_m_raw + (1 - ALPHA_CXY_MIDAS) * prev_cx_m
                )
                cy_m_f = cy_m_raw if prev_cy_m is None else (
                    ALPHA_CXY_MIDAS * cy_m_raw + (1 - ALPHA_CXY_MIDAS) * prev_cy_m
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
                else:
                    z_raw = float(depth_map[cy_m, cx_m])
                z_corrected = Z_MAX - z_raw   # large = far, small = close
                tid = track_assignments[local_idx]
                prev_z = tracked_objects[tid].get('z_f')
                tracked_objects[tid]['z_f'] = (
                        z_corrected if prev_z is None
                        else ALPHA_Z * z_corrected + (1 - ALPHA_Z) * prev_z
                )

                # --- Liste pour l'interface Mac : id,x,y,z,conf ---
                list_data.append(
                    f"{track_id},{int(cxi - no_x)},{int(no_y - cyi)},"
                    f"{z_corrected:.2f},{float(sc):.2f}"
                )

                # --- Drawing ---
                is_sel = (track_id == selected_id)
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
                if is_sel:
                    cv2.putText(
                        output_frame, "CIBLE",
                        (x1i, max(10, y1i - 80)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
                    )
      

    # ---------- Envois UDP ----------
    
    # Liste complète des objets -> interface Mac (toujours, même vide pour vider la liste)
    sock.sendto(";".join(list_data).encode(), (MAC_IP, LIST_PORT))

    # Cible verrouillée uniquement -> contrôleur PID (valeurs filtrées)
    if selected_id is not None and selected_id in tracked_objects:
        t = tracked_objects[selected_id]
        if t.get('z_f') is not None:
           # Normalization 
           dx = float(np.clip(t['x_f']/(FRAME_W/2),-1.0,1.0))
           dy = float(np.clip(t['y_f']/(FRAME_H/2),-1.0,1.0))
           #depth error
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
    t_vid_out = time.time() - t_vid 


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