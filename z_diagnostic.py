"""
z_diagnostic_logger.py
----------------------
Logger léger pour isoler la source des sauts de profondeur (Z) MiDaS.
À insérer dans la boucle vision sur l'ROV. Aucune dépendance hors stdlib + numpy.

Capture par frame :
  - z_px  : disparité brute au pixel central du centroïde (ce que tu fais déjà)
  - z_med : médiane d'une fenêtre autour du centroïde (= test du fix spatial)
  - z_std : écart-type local (rugosité de la depth map à cet endroit)
  - cx_m, cy_m : centroïde dans le repère de la depth map (dérive tracker)
  - conf  : confiance YOLO (détection faible -> Z instable ?)
  - bw,bh : taille bbox (proxy de distance, optionnel)

Le tout permet de distinguer les 3 causes en UNE seule capture :
  tracker (centroïde) vs MiDaS temporel vs quantification uint8.
"""

import csv
import time
import numpy as np


def sample_depth(depth_map, cx, cy, win=5):
    """
    Échantillonne la depth map MiDaS autour du centroïde.
    Retourne (z_pixel_central, z_median_fenetre, z_std_fenetre) en disparité brute.

    - depth_map : np.ndarray 2D (disparité brute uint8/uint16, PAS une distance)
    - cx, cy    : centroïde DANS le repère de la depth map
    - win       : côté de la fenêtre carrée (impair conseillé : 3, 5, 7)

    NB : utilise .item() au lieu de float() -> corrige la deprecation NumPy (ligne ~402).
    """
    h, w = depth_map.shape[:2]
    cx = int(round(cx))
    cy = int(round(cy))

    if not (0 <= cy < h and 0 <= cx < w):
        return float("nan"), float("nan"), float("nan")

    z_px = depth_map[cy, cx].item()          # pixel central (fix deprecation)

    r = win // 2
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    patch = depth_map[y0:y1, x0:x1].astype(np.float32)

    return float(z_px), float(np.median(patch)), float(np.std(patch))


class ZDiagnosticLogger:
    """
    Bufferise via le buffer OS, flush périodique pour ne pas staller la boucle
    temps réel. Appelle .stop() à l'arrêt pour fermer proprement.
    """

    HEADER = ["frame", "t", "valid", "conf",
              "cx_m", "cy_m", "bw", "bh",
              "z_px", "z_med", "z_std"]

    def __init__(self, path="z_diag.csv", flush_every=30):
        self.path = path
        self.flush_every = flush_every
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self.HEADER)
        self._n = 0
        self._t0 = time.perf_counter()

    def log(self, frame, valid, conf, cx_m, cy_m,
            z_px, z_med, z_std,
            bw=float("nan"), bh=float("nan")):
        """Une ligne par frame. Sur frame sans détection : valid=0 et z_* en NaN."""
        t = time.perf_counter() - self._t0
        self._w.writerow([
            frame, f"{t:.4f}", int(valid), f"{conf:.4f}",
            f"{cx_m:.2f}", f"{cy_m:.2f}", f"{bw:.2f}", f"{bh:.2f}",
            f"{z_px:.3f}", f"{z_med:.3f}", f"{z_std:.3f}",
        ])
        self._n += 1
        if self._n % self.flush_every == 0:
            self._f.flush()

    def stop(self):
        self._f.flush()
        self._f.close()
        print(f"[z-diag] {self._n} frames -> {self.path}")