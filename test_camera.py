import cv2

# Caméra locale (USB ou intégrée)
cap = cv2.VideoCapture(0)  # 0 = première caméra

# Ou via un flux réseau (RTSP, typique sur ROV)
#cap = cv2.VideoCapture("rtsp://192.168.1.x:8554/stream")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("ROV Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()