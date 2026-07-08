import os, sys, time, urllib.request
import numpy as np
from PIL import Image
import onnxruntime_qnn as ort



 # -- Options  de  session------------------------------------------------
so = ort.SessionOptions() 
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
def curr_ms():
    return round(time.time() * 1000)

use_npu = True if len(sys.argv) >= 2 and sys.argv[1] == '--use-npu' else False
MODEL_PATH = os.path.expanduser("~/ai-rov/deepbox-onnx-w8a16/vgg_3d_detection.onnx")

MODEL_DATA_PATH = "vgg_3d_detection.data"
# Options du provider
providers = []

if use_npu:
    _C.register_execution_provider_library(
        "QNNExecutionProvider",
        onnxruntime_qnn.get_library_path()
    )
    qnn_options = {
        "backend_path": onnxruntime_qnn.get_qnn_htp_path(),
        "htp_performance_mode": "burst",
    }
    providers.append(("QNNExecutionProvider", qnn_options))
else:
    providers.append("CPUExecutionProvider")
# -- Chargement
sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=providers)
# Verification HTP actif
actual_providers = sess.get_providers()
print(f"Using providers: {actual_providers}") # Show which providers are actually loaded
# -- Inference ---------------------------------------------------------
input_name = sess.get_inputs()[0].name
x = np.random.randn(1, 3, 224, 224).astype(np.uint16)
outputs = sess.run(None, {input_name: x})
print(f"Output shape : {outputs[0].shape}")

inputs  = sess.get_inputs()
outputs = sess.get_outputs()

for i in inputs:
    print(i.name, i.shape,i.type)

for o in outputs:
    print(o.name, o.shape, o.type)
