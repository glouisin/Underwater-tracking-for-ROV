import torch
import torchvision.models as models
import numpy as np
import os








# ?? Etape 3bis : SavedModel -> TFLite INT8 (avec calibration) ?????????
def savedmodel_to_tflite_int8(saved_model_dir: "ai-rov/src",
                               output: str = "3Dbox_int8.tflite",
                               n_calib: int = 100):
    import tensorflow as tf

    # Generateur de donnees de calibration
    # IMPORTANT : doit representer la distribution reelle des entrees
    def representative_dataset():
        for _ in range(n_calib):
            # Remplacer par de vraies images pour un modele de production
            sample = np.random.randint(0, 256,
                                       (1, 224, 224, 3),
                                       dtype=np.uint8).astype(np.float32)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # Activer la quantification INT8 complete (poids + activations)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type  = tf.uint8   # entree uint8
    converter.inference_output_type = tf.uint8   # sortie uint8

    tflite_model = converter.convert()
    with open(output, "wb") as f:
        f.write(tflite_model)

    size_int8   = os.path.getsize(output) / 1e6
    size_float  = os.path.getsize("ai-rov/src/deepbox-tflite-float/vgg_3d_detection.tflite") / 1e6
    print(f"[OK] TFLite INT8 : {output} ({size_int8:.1f} MB)")
    print(f"     Reduction    : {100*(1-size_int8/size_float):.0f}%")


if __name__ == "__main__":
    #onnx_to_savedmodel("mobilenet.onnx")
    #savedmodel_to_tflite_float32("saved_model")
    savedmodel_to_tflite_int8("saved_model")