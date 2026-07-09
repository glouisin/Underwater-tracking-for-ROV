import socket
import json

UDP_PORT = 17002
UDP_IP = "127.0.0.1"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

print(f"Listening on {UDP_IP}:{UDP_PORT}...")

frame = 0

while True:
    try:
        data, _ = sock.recvfrom(1024)
        msg = data.decode()
        frame += 1

        packet = json.loads(msg)

        valid  = packet.get("vision_valid", False)
        conf   = packet.get("confidence", 0.0)
        dx     = packet.get("dx_normalized", 0.0)
        dy     = packet.get("dy_normalized", 0.0)
        dz     = packet.get("scale_error_normalized", 0.0)

        status = "TRACKING" if valid else "NO TARGET"

        print(f"[f{frame:04d}] {status} | "
              f"conf={conf:.2f} | "
              f"dx={dx:+.3f} | "
              f"dy={dy:+.3f} | "
              f"dz={dz:+.3f}")

    except BlockingIOError:
        pass
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON invalide : {e} — reçu : {msg[:80]}")