#!/usr/bin/env python3
"""
Test isolé : Depth Anything V2 (TFLite) sur HTP via un QAIRT plus récent,
téléchargé séparément dans /opt/qcom/qairt-new (à adapter au chemin réel).

Objectif : confirmer que le GELU du backbone ViT est bien supporté par cette
version de QAIRT sur HTP, AVANT de toucher au pipeline principal.

A lancer isolément : python test_depth_htp_new_qairt.py
"""

import os

# --- Chemin réel trouvé sur rov-test après extraction (QAIRT 2.48.0) ---
NEW_QAIRT_ROOT = "/opt/qcom/qairt-new/qairt/2.48.0.260626"

# IMPORTANT : bien mettre le NOUVEAU chemin en premier, avant l'ancien
# (sinon le système pourrait charger les .so de /opt/qcom/qirp-sdk en premier)
os.environ["ADSP_LIBRARY_PATH"] = (
    f"{NEW_QAIRT_ROOT}/lib/hexagon-v73/unsigned;"
    "/opt/qcom/qirp-sdk/lib/hexagon-v73/unsigned;"
    + os.environ.get("ADSP_LIBRARY_PATH", "")
)

import time
import numpy as np
import ai_edge_litert.interpreter as tflite

MODEL_PATH_DEPTH = "models/depth_anything_v2-tflite-float/depth_anything_v2.tflite"  # ajuste si besoin

# --- AJUSTE ce chemin vers le libQnnTFLiteDelegate.so du NOUVEAU SDK ---
NEW_DELEGATE_PATH = f"{NEW_QAIRT_ROOT}/lib/aarch64-oe-linux-gcc11.2/libQnnTFLiteDelegate.so"

print(f"Chargement du delegate depuis : {NEW_DELEGATE_PATH}")
delegate_options = {
    'backend_type': 'htp',
    'htp_performance_mode': '1',  # ou 'sustained_high_performance'
}
try:
    delegate = tflite.load_delegate(NEW_DELEGATE_PATH, options=delegate_options)
except Exception as e:
    print(f"❌ Échec du chargement du delegate : {e}")
    raise SystemExit(1)

print("Chargement du modèle Depth Anything V2 sur HTP...")
t0 = time.time()
try:
    depth_interpreter = tflite.Interpreter(
        model_path=MODEL_PATH_DEPTH,
        experimental_delegates=[delegate]
    )
    depth_interpreter.allocate_tensors()
except Exception as e:
    print(f"❌ Échec — GELU probablement toujours pas supporté sur HTP avec ce SDK : {e}")
    raise SystemExit(1)

print(f"✅ Modèle chargé sur HTP en {(time.time()-t0)*1000:.1f} ms")

in_det = depth_interpreter.get_input_details()
out_det = depth_interpreter.get_output_details()
in_h, in_w = in_det[0]["shape"][1:3]
print(f"Entrée : {in_w}x{in_h}  dtype={in_det[0]['dtype']}")

# ---- Image de test (bruit aléatoire — juste pour valider la mécanique) ----
dummy = np.random.randint(0, 255, (1, in_h, in_w, 3)).astype(in_det[0]['dtype'])

for _ in range(3):
    depth_interpreter.set_tensor(in_det[0]['index'], dummy)
    depth_interpreter.invoke()

N = 20
t0 = time.time()
for _ in range(N):
    depth_interpreter.set_tensor(in_det[0]['index'], dummy)
    depth_interpreter.invoke()
elapsed = (time.time() - t0) / N * 1000
print(f"Latence moyenne sur {N} runs (HTP) : {elapsed:.2f} ms")