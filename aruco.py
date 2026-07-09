#!/usr/bin/env python3
"""
Génère les images de marqueurs ArUco à imprimer pour le protocole de
calibration (un ID par palier de distance).

Usage:
    python3 generate_aruco_markers.py --n-markers 7 --size-px 600 --out-dir markers/

Imprime chaque image sur une feuille rigide (carton plastifié conseillé
pour tenue en piscine), en gardant une bordure blanche autour du marqueur
(le contour du damier doit rester bien contrasté et net, pas de reliure au
ras du bord). Associe ensuite chaque ID à sa distance dans
MARKER_TO_DISTANCE (fichier detect_paliers.py) selon l'ordre où tu les
utilises pendant le protocole.

Note : DICT_4X4_50 = 50 IDs possibles (0 à 49), largement suffisant pour
quelques paliers de distance. Un marqueur physique par ID -- pas besoin de
réimprimer entre les sessions, la même planche de marqueurs sert à tous
les runs (répétitions incluses).
"""

import argparse
import os

import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-markers", type=int, default=7, help="nombre de marqueurs à générer (= nombre de paliers)")
    p.add_argument("--size-px", type=int, default=600, help="taille de l'image en pixels (carré)")
    p.add_argument("--dict", default="DICT_4X4_50", help="dictionnaire ArUco (garder DICT_4X4_50 sauf besoin spécifique)")
    p.add_argument("--out-dir", default="markers", help="dossier de sortie")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dict_id = getattr(cv2.aruco, args.dict)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    for marker_id in range(args.n_markers):
        img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, args.size_px)

        # Ajoute une bordure blanche (marge) autour du marqueur pour faciliter
        # la détection quand la plaque physique est vue avec un léger angle
        # ou depuis une distance importante.
        border = args.size_px // 6
        img_with_border = cv2.copyMakeBorder(
            img, border, border, border, border,
            cv2.BORDER_CONSTANT, value=255
        )

        # Label texte sous le marqueur pour identification visuelle facile
        # à l'impression (utile pour ne pas confondre les plaques sur le
        # bord de la piscine)
        label_h = 60
        img_final = cv2.copyMakeBorder(
            img_with_border, 0, label_h, 0, 0,
            cv2.BORDER_CONSTANT, value=255
        )
        cv2.putText(
            img_final, f"ID {marker_id}",
            (10, img_final.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2
        )

        out_path = os.path.join(args.out_dir, f"marker_{marker_id:02d}.png")
        cv2.imwrite(out_path, img_final)
        print(f"  {out_path}")

    print(f"\n{args.n_markers} marqueurs générés dans {args.out_dir}/")
    print("Rappel : associe chaque ID à une distance dans MARKER_TO_DISTANCE")
    print("(dans detect_paliers.py) selon l'ordre d'utilisation prévu.")


if __name__ == "__main__":
    main()