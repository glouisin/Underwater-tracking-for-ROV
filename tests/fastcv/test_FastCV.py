#!/usr/bin/env python3
"""
Wrapper ctypes pour fcvScaleDownMNInterleaveu8 (FastCV).
Remplace cv2.resize() pour les downscales BGR/RGB du pipeline vision.

ATTENTION : la signature exacte n'est pas confirmée (pas de header sur la board).
Ce script teste l'appel et vérifie visuellement le résultat avant intégration.
"""

import ctypes
import numpy as np
import cv2
import time

# -------------------- Chargement de la lib --------------------
fastcv = ctypes.CDLL("/usr/lib/libfastcvopt.so")

# -------------------- Signature confirmée via fastcv.h --------------------
# FASTCV_API void
# fcvScaleDownMNu8( const uint8_t* __restrict src,
#                   uint32_t srcWidth, uint32_t srcHeight, uint32_t srcStride,
#                   uint8_t* __restrict dst,
#                   uint32_t dstWidth, uint32_t dstHeight, uint32_t dstStride );
# NOTE: single-channel uniquement (un plan 8-bit) — pas de paramètre "channels".
fastcv.fcvScaleDownMNu8.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # src
    ctypes.c_uint32,                  # srcWidth
    ctypes.c_uint32,                  # srcHeight
    ctypes.c_uint32,                  # srcStride
    ctypes.POINTER(ctypes.c_uint8),  # dst
    ctypes.c_uint32,                  # dstWidth
    ctypes.c_uint32,                  # dstHeight
    ctypes.c_uint32,                  # dstStride
]
fastcv.fcvScaleDownMNu8.restype = None


def fastcv_resize_channel(src_plane: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    """Resize un seul plan 8-bit (grayscale) via FastCV."""
    src_plane = np.ascontiguousarray(src_plane, dtype=np.uint8)
    src_h, src_w = src_plane.shape

    dst_plane = np.empty((dst_h, dst_w), dtype=np.uint8)

    src_ptr = src_plane.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    dst_ptr = dst_plane.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    fastcv.fcvScaleDownMNu8(
        src_ptr, src_w, src_h, src_w,
        dst_ptr, dst_w, dst_h, dst_w,
    )
    return dst_plane


def fastcv_resize(src: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    """
    Resize BGR 3 canaux en traitant chaque plan séparément via FastCV,
    puis refusion avec cv2.merge.
    """
    b, g, r = cv2.split(src)
    b_rs = fastcv_resize_channel(b, dst_w, dst_h)
    g_rs = fastcv_resize_channel(g, dst_w, dst_h)
    r_rs = fastcv_resize_channel(r, dst_w, dst_h)
    return cv2.merge([b_rs, g_rs, r_rs])


# -------------------- Test de validation --------------------
if __name__ == "__main__":
    # Image de test synthétique : damier pour vérifier visuellement la déformation
    test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    test_img[::40, :, :] = 255
    test_img[:, ::40, :] = 255

    print("Test 1 : appel FastCV ScaleDownMNInterleave")
    try:
        result_fastcv = fastcv_resize(test_img, 640, 640)
        print(f"  OK — shape sortie : {result_fastcv.shape}")
        cv2.imwrite("/root/ai-rov/src/tmp/test_fastcv_resize.png", result_fastcv)
        print("  Image sauvegardée : /tmp/test_fastcv_resize.png")
    except Exception as e:
        print(f"  ÉCHEC : {e}")
        print("  → La signature ne correspond probablement pas.")
        print("  → Essaie le script alternatif avec ordre de params différent.")
        raise SystemExit(1)

    print("\nTest 2 : comparaison avec cv2.resize (référence)")
    result_cv2 = cv2.resize(test_img, (640, 640))
    cv2.imwrite("/root/ai-rov/src/tmp/test_cv2_resize.png", result_cv2)
    print("  Image sauvegardée : /tmp/test_cv2_resize.png")

    print("\nTest 3 : benchmark de vitesse (100 itérations)")
    N = 100

    t0 = time.perf_counter()
    for _ in range(N):
        _ = fastcv_resize(test_img, 640, 640)
    t_fastcv = (time.perf_counter() - t0) / N * 1000

    t0 = time.perf_counter()
    for _ in range(N):
        _ = cv2.resize(test_img, (640, 640))
    t_cv2 = (time.perf_counter() - t0) / N * 1000

    print(f"  FastCV : {t_fastcv:.3f} ms/appel")
    print(f"  OpenCV : {t_cv2:.3f} ms/appel")
    print(f"  Ratio  : {t_cv2/t_fastcv:.2f}x" if t_fastcv > 0 else "  N/A")

    print("\n=> Compare visuellement /tmp/test_fastcv_resize.png et /tmp/test_cv2_resize.png")
    print("=> Si l'image FastCV est corrompue/déformée, la signature est fausse.")
