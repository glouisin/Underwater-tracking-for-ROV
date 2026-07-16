#!/usr/bin/env python3
"""
Enregistre une vidéo depuis la caméra pour le protocole de calibration
(paliers de distance MiDaS/Depth Anything).

Version HEADLESS -- pas d'affichage graphique requis (fonctionne en SSH
sans forwarding X11, ce qui est le cas typique sur rov2). Le marquage de
palier se fait en tapant Entrée dans le terminal plutôt qu'en appuyant sur
une touche dans une fenêtre.

Usage:
    python3 record_calibration_video.py --out calib_test.mp4
    python3 record_calibration_video.py --out calib_test.mp4 --device 2 --duration 120

Pendant l'enregistrement :
    Tape Entrée (juste valider une ligne vide) pour marquer un palier
    -> log la frame courante dans la console, utile en backup meme avec
       la detection ArUco automatique en post-traitement.
    Tape 'q' puis Entrée pour arreter et sauvegarder.

Si tu as bien un serveur X disponible (ex: connecté en local ou via
`ssh -X`) et que tu veux quand meme l'aperçu visuel avec overlay,
utilise --show-preview (nécessite alors un DISPLAY valide).
"""

import argparse
import threading
import time

import cv2


def stdin_listener(stop_event, mark_event_list):
    """Lit stdin ligne par ligne dans un thread séparé : Entrée seule = marquer
    un palier, 'q' = arrêter. Ne bloque pas la boucle de capture principale."""
    while not stop_event.is_set():
        try:
            line = input()
        except EOFError:
            break
        if line.strip().lower() == 'q':
            stop_event.set()
            break
        else:
            mark_event_list.append(time.time())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="fichier de sortie (.mp4)")
    p.add_argument("--device", type=int, default=2, help="index /dev/videoN (defaut: 2, comme CameraStream)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--duration", type=float, default=None, help="duree max en secondes (defaut: illimite, arreter en tapant q + Entrée)")
    p.add_argument("--show-preview", action="store_true", help="active l'aperçu graphique (nécessite un DISPLAY valide, ex: ssh -X)")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print(f"Impossible d'ouvrir /dev/video{args.device}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (args.width, args.height))

    print(f"Enregistrement -> {args.out}  ({args.width}x{args.height} @ {args.fps}fps)")
    if args.show_preview:
        print("Mode aperçu graphique actif : 'q' dans la fenêtre pour arrêter, espace pour marquer.")
    else:
        print("Mode headless (pas de fenêtre) : tape Entrée pour marquer un palier, 'q' + Entrée pour arrêter.")

    frame_idx = 0
    t_start = time.time()
    markers_log = []

    stop_event = threading.Event()
    mark_events = []  # timestamps (time.time()) des marquages, remplis par le thread stdin

    if not args.show_preview:
        listener_thread = threading.Thread(target=stdin_listener, args=(stop_event, mark_events), daemon=True)
        listener_thread.start()

    last_mark_count_seen = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Erreur lecture caméra, arrêt.")
            break

        writer.write(frame)
        elapsed = time.time() - t_start

        if args.show_preview:
            display = frame.copy()
            cv2.putText(display, f"Frame: {frame_idx}  t={elapsed:.1f}s",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Enregistrement calibration (q=quitter, espace=marquer palier)", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                markers_log.append((frame_idx, elapsed))
                print(f"  [PALIER MARQUÉ] frame={frame_idx}  t={elapsed:.1f}s")
        else:
            # Consomme les marquages arrivés depuis le thread stdin et les
            # associe à la frame courante (au moment où Entrée a été détecté)
            if len(mark_events) > last_mark_count_seen:
                new_marks = len(mark_events) - last_mark_count_seen
                for _ in range(new_marks):
                    markers_log.append((frame_idx, elapsed))
                    print(f"  [PALIER MARQUÉ] frame={frame_idx}  t={elapsed:.1f}s")
                last_mark_count_seen = len(mark_events)

            # Affichage périodique de la progression (toutes les ~2s) puisqu'il
            # n'y a pas d'aperçu visuel pour se repérer en mode headless
            if frame_idx % (args.fps * 2) == 0:
                print(f"  ... frame {frame_idx}  t={elapsed:.1f}s")

        frame_idx += 1
        if args.duration is not None and elapsed >= args.duration:
            print("Durée max atteinte, arrêt.")
            break

    stop_event.set()
    cap.release()
    writer.release()
    if args.show_preview:
        cv2.destroyAllWindows()

    print(f"\nTerminé. {frame_idx} frames écrites dans {args.out}")
    if markers_log:
        print("\nPaliers marqués manuellement (backup) :")
        for f_idx, t in markers_log:
            print(f"  frame {f_idx}  (t={t:.1f}s)")


if __name__ == "__main__":
    main()