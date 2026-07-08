#!/usr/bin/env python3
# Reproduit l'ANCIEN comportement (lecture chaque tour, pas de gating)
# et separe : tours de boucle/s   vs   frames NEUVES/s
import cv2, time, threading

class Grab:
    def __init__(self, src=2):
        self.seq = 0
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.ok, self.frame = self.cap.read()
        self.run = True
        threading.Thread(target=self._u, daemon=True).start()
    def _u(self):
        while self.run:
            ok, f = self.cap.read()
            self.ok, self.frame, self.seq = ok, f, self.seq + 1

g = Grab(2)
time.sleep(0.5)

loops = 0          # tours de boucle (= ton ancien "FPS")
last_seq = -1
seen = set()       # seq uniques = frames reellement neuves
t0 = time.time()
while time.time() - t0 < 5.0:
    s = g.seq
    seen.add(s)
    loops += 1
    # on simule un peu de charge par tour, comme l'inference
    time.sleep(0.018)
dt = time.time() - t0
g.run = False

print(f"Tours de boucle/s  (ancien compteur) : {loops/dt:.1f}")
print(f"Frames NEUVES/s    (camera reelle)   : {(len(seen)-1)/dt:.1f}")
print(f"Ratio retraitement                   : {loops/max(1,len(seen)-1):.1f}x")