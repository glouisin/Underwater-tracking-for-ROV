#!/usr/bin/env python3
import cv2, time


cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Ce que le driver a REELLEMENT negocie
print("FOURCC :", int(cap.get(cv2.CAP_PROP_FOURCC)))
print("W x H  :", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS    :", cap.get(cv2.CAP_PROP_FPS))

n, t0 = 0, time.time()
while n < 150:
    ok, f = cap.read()
    if not ok:
        continue
    n += 1
    if n % 30 == 0:
        dt = time.time() - t0
        print(f"  {n} frames  ->  {n/dt:.1f} FPS")
cap.release()