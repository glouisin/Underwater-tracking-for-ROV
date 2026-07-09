from __future__ import annotations

import argparse
import json
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

from tracker_core import OpticalFlowTracker


@dataclass
class SharedState:
    jpeg_bytes: Optional[bytes] = None
    status: Dict[str, Any] = field(default_factory=dict)
    width: int = 0
    height: int = 0
    frame_seq: int = 0       # increments for every processed vision frame
    jpeg_seq: int = 0        # increments only when a new UI JPEG is encoded
    last_command_id: int = 0


class SharedMjpegCameraClient:
    def __init__(
        self,
        stream_url: str,
        connect_timeout: float = 2.0,
        read_timeout: float = 5.0,
        reconnect_delay: float = 1.0,
        chunk_size: int = 4096,
    ) -> None:
        self.stream_url = stream_url
        self.connect_timeout = float(connect_timeout)
        self.read_timeout = float(read_timeout)
        self.reconnect_delay = float(reconnect_delay)
        self.chunk_size = int(chunk_size)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cond = threading.Condition()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_seq: int = 0
        self._session = requests.Session()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                response = self._session.get(
                    self.stream_url,
                    stream=True,
                    timeout=(self.connect_timeout, self.read_timeout),
                )
                response.raise_for_status()

                buffer = bytearray()
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if self._stop_event.is_set():
                        break
                    if not chunk:
                        continue
                    buffer.extend(chunk)

                    while True:
                        start = buffer.find(b'\xff\xd8')
                        if start < 0:
                            if len(buffer) > 2 * self.chunk_size:
                                del buffer[:-2]
                            break

                        end = buffer.find(b'\xff\xd9', start + 2)
                        if end < 0:
                            if start > 0:
                                del buffer[:start]
                            break

                        jpg = bytes(buffer[start:end + 2])
                        del buffer[:end + 2]

                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
                            continue

                        with self._cond:
                            self._latest_frame = frame
                            self._frame_seq += 1
                            self._cond.notify_all()
            except requests.RequestException as exc:
                print(f'[WARN] MJPEG stream from mjpeg_server.py error: {exc}')
            except Exception as exc:
                print(f'[WARN] Shared camera decode error: {exc}')

            if not self._stop_event.is_set():
                time.sleep(self.reconnect_delay)

    def read(self, timeout: float = 1.0, last_seq: Optional[int] = None) -> Tuple[bool, Optional[np.ndarray], int]:
        deadline = time.time() + max(0.0, timeout)
        with self._cond:
            while not self._stop_event.is_set():
                seq_ready = self._frame_seq > 0 and (last_seq is None or self._frame_seq != last_seq)
                if seq_ready and self._latest_frame is not None:
                    return True, self._latest_frame.copy(), self._frame_seq

                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)

        return False, None, self._frame_seq

    def close(self) -> None:
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._session.close()


class EdgeTrackerAgent:
    POINT_BOX_FRACTION = 0.12
    MIN_POINT_BOX_PX = 48.0
    MAX_POINT_BOX_PX = 120.0

    def __init__(
        self,
        server_url: str,
        agent_id: str = 'edge-1',
        camera_index: int = 2,
        width: int = 1280,
        height: int = 720,
        camera_fps: int = 30,
        upload_fps: float = 10.0,
        jpeg_quality: int = 80,
        controller_url: str = 'http://127.0.0.1:17001',
        controller_timeout: float = 0.25,
        controller_push_hz: float = 20.0,
        controller_transport: str = 'udp',
        controller_udp_host: str = '127.0.0.1',
        controller_udp_port: int = 17002,
        vision_min_conf: float = 0.50,
        vision_deadband: float = 0.04,
        vision_k_forward: float = 0.45,
        vision_k_lateral: float = 0.45,
        vision_k_vertical: float = 0.30,
        vision_k_range: float = 0.30,
        vision_range_deadband: float = 0.03,
        vision_range_axis: str = 'forward',
        vision_range_min_conf: float = 0.75,
        vision_range_max_abs: float = 0.40,
        vision_max_bbox_area_ratio: float = 0.35,
        vision_max_cmd: float = 0.35,
        camera_stream_url: Optional[str] = None,
        stream_connect_timeout: float = 2.0,
        stream_read_timeout: float = 5.0,
        stream_reconnect_delay: float = 1.0,
    ) -> None:
        self.server_url = server_url.rstrip('/')
        self.agent_id = agent_id
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.camera_fps = camera_fps
        self.upload_fps = max(1.0, upload_fps)
        self.jpeg_quality = max(30, min(95, jpeg_quality))

        self.camera_stream_url = camera_stream_url.strip() if camera_stream_url else None
        self.stream_connect_timeout = float(stream_connect_timeout)
        self.stream_read_timeout = float(stream_read_timeout)
        self.stream_reconnect_delay = float(stream_reconnect_delay)

        self.tracker = OpticalFlowTracker()
        self.shared = SharedState()
        self.shared_lock = threading.Lock()
        self.command_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self.stop_event = threading.Event()

        self.controller_url = controller_url.rstrip('/')
        self.controller_timeout = float(controller_timeout)
        self.controller_push_hz = max(1.0, float(controller_push_hz))
        self.controller_transport = str(controller_transport).strip().lower()
        if self.controller_transport not in ('udp', 'http'):
            self.controller_transport = 'udp'
        self.controller_udp_addr = (str(controller_udp_host), int(controller_udp_port))
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.vision_min_conf = float(vision_min_conf)
        self.vision_deadband = float(vision_deadband)
        self.vision_k_forward = float(vision_k_forward)
        self.vision_k_lateral = float(vision_k_lateral)
        self.vision_k_vertical = float(vision_k_vertical)
        self.vision_k_range = float(vision_k_range)
        self.vision_range_deadband = float(vision_range_deadband)
        self.vision_range_axis = str(vision_range_axis).strip().lower()
        if self.vision_range_axis not in ('forward', 'vertical', 'off'):
            self.vision_range_axis = 'forward'
        self.vision_range_min_conf = float(vision_range_min_conf)
        self.vision_range_max_abs = float(vision_range_max_abs)
        self.vision_max_bbox_area_ratio = float(vision_max_bbox_area_ratio)
        self.vision_max_cmd = float(vision_max_cmd)
        self._last_controller_error_log_ts = 0.0

    def run(self) -> None:
        worker = threading.Thread(target=self.network_worker, daemon=True)
        worker.start()
        self.capture_loop()

    def network_worker(self) -> None:
        session = requests.Session()
        last_uploaded_seq = -1
        last_pushed_seq = -1
        upload_interval = 1.0 / self.upload_fps
        push_interval = 1.0 / self.controller_push_hz
        next_upload_ts = 0.0
        next_push_ts = 0.0

        while not self.stop_event.is_set():
            now = time.time()
            if now >= next_upload_ts:
                self.upload_latest(session, last_uploaded_seq)
                with self.shared_lock:
                    last_uploaded_seq = self.shared.jpeg_seq
                next_upload_ts = now + upload_interval

            if now >= next_push_ts:
                self.push_latest_vision(session, last_pushed_seq)
                with self.shared_lock:
                    last_pushed_seq = self.shared.frame_seq
                next_push_ts = now + push_interval

            self.poll_commands(session)
            time.sleep(0.03)

    def upload_latest(self, session: requests.Session, last_uploaded_seq: int) -> None:
        with self.shared_lock:
            seq = self.shared.jpeg_seq
            jpeg_bytes = self.shared.jpeg_bytes
            tracker_status = dict(self.shared.status)
            width = self.shared.width
            height = self.shared.height

        # Upload only when a new preview JPEG was actually encoded.
        # Control/status can still run at full frame rate through push_latest_vision().
        if jpeg_bytes is None or seq == last_uploaded_seq:
            return

        ui_status = self.build_ui_status(tracker_status)

        try:
            response = session.post(
                f'{self.server_url}/api/agent/upload',
                data={
                    'agent_id': self.agent_id,
                    'status_json': json.dumps(ui_status),
                    'width': str(width),
                    'height': str(height),
                },
                files={
                    'frame': ('frame.jpg', jpeg_bytes, 'image/jpeg'),
                },
                timeout=(0.5, 1.2),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f'[WARN] Upload failed: {exc}')

    def poll_commands(self, session: requests.Session) -> None:
        with self.shared_lock:
            after_id = self.shared.last_command_id
        try:
            response = session.get(
                f'{self.server_url}/api/agent/commands',
                params={'agent_id': self.agent_id, 'after_id': after_id},
                timeout=(0.5, 1.0),
            )
            response.raise_for_status()
            payload = response.json()
            commands: List[Dict[str, Any]] = payload.get('commands', [])
            if not commands:
                return
            max_id = after_id
            for cmd in commands:
                self.command_queue.put(cmd)
                max_id = max(max_id, int(cmd.get('id', 0)))
            with self.shared_lock:
                self.shared.last_command_id = max_id
        except requests.RequestException:
            return
        except ValueError:
            return

    def apply_pending_commands(self, latest_frame) -> None:
        while True:
            try:
                cmd = self.command_queue.get_nowait()
            except queue.Empty:
                break

            cmd_type = cmd.get('type')
            payload = cmd.get('payload', {}) or {}
            if cmd_type == 'select_point':
                ok = self.select_point(
                    latest_frame,
                    float(payload.get('x', 0.0)),
                    float(payload.get('y', 0.0)),
                )
                if not ok:
                    self.tracker.state.status_text = 'Invalid point selection or no good frame yet'
                    self.tracker.state.mode = 'IDLE'
            elif cmd_type == 'select_bbox':
                ok = self.tracker.select_bbox(
                    latest_frame,
                    float(payload.get('x', 0.0)),
                    float(payload.get('y', 0.0)),
                    float(payload.get('w', 0.0)),
                    float(payload.get('h', 0.0)),
                )
                if not ok:
                    self.tracker.state.status_text = 'Invalid bbox selection or no good frame yet'
                    self.tracker.state.mode = 'IDLE'
            elif cmd_type == 'reset':
                self.tracker.reset()
            elif cmd_type == 'toggle_pause':
                self.tracker.toggle_pause()
            elif cmd_type == 'toggle_points':
                self.tracker.toggle_show_points()

    @staticmethod
    def clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(v)))

    def point_to_bbox(self, latest_frame: np.ndarray, x: float, y: float) -> Tuple[float, float, float, float]:
        height, width = latest_frame.shape[:2]
        size = self.clamp(
            min(float(width), float(height)) * self.POINT_BOX_FRACTION,
            self.MIN_POINT_BOX_PX,
            self.MAX_POINT_BOX_PX,
        )
        size = min(size, float(width), float(height))
        cx = self.clamp(x, 0.0, max(0.0, float(width) - 1.0))
        cy = self.clamp(y, 0.0, max(0.0, float(height) - 1.0))
        x0 = self.clamp(cx - 0.5 * size, 0.0, max(0.0, float(width) - size))
        y0 = self.clamp(cy - 0.5 * size, 0.0, max(0.0, float(height) - size))
        return x0, y0, size, size

    def select_point(self, latest_frame: np.ndarray, x: float, y: float) -> bool:
        bbox = self.point_to_bbox(latest_frame, x, y)
        ok = self.tracker.select_bbox(latest_frame, *bbox)
        if ok:
            self.tracker.state.status_text = 'Point selected'
        return ok

    def build_ui_status(self, status: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'mode': str(status.get('mode', 'IDLE')),
            'confidence': float(status.get('confidence', 0.0)),
            'stream_fps': float(status.get('stream_fps', 0.0)),
            'status_text': str(status.get('status_text', 'No target selected')),
            'track_ok': bool(status.get('track_ok', False)),
            'target_x': float(status.get('target_cx', 0.0)),
            'target_y': float(status.get('target_cy', 0.0)),
        }

    def build_preview_frame(self, frame: np.ndarray, status: Dict[str, Any]) -> np.ndarray:
        preview = frame.copy()
        bbox = status.get('bbox')
        if bbox is None:
            return preview

        mode = str(status.get('mode', 'IDLE'))
        if mode == 'TRACKING':
            color = (0, 220, 0)
        elif mode == 'LOST':
            color = (0, 0, 255)
        else:
            color = (0, 200, 255)

        x, y, w, h = [float(v) for v in bbox]
        cx = int(round(float(status.get('target_cx', x + 0.5 * w))))
        cy = int(round(float(status.get('target_cy', y + 0.5 * h))))
        cv2.drawMarker(preview, (cx, cy), color, cv2.MARKER_CROSS, 24, 2)
        cv2.circle(preview, (cx, cy), 9, color, 2)
        return preview

    def build_visual_packet(self, status: Dict[str, Any]) -> Dict[str, Any]:
        mode = str(status.get('mode', 'IDLE'))
        bbox = status.get('bbox')
        conf = float(status.get('confidence', 0.0))

        frame_cx = float(status.get('frame_cx', 0.0))
        frame_cy = float(status.get('frame_cy', 0.0))
        frame_width = max(1.0, 2.0 * frame_cx)
        frame_height = max(1.0, 2.0 * frame_cy)

        valid = (mode == 'TRACKING') and (bbox is not None) and (conf >= self.vision_min_conf)

        if valid:
            dx_norm = self.clamp(
                float(status.get('move_dx_px', 0.0)) / max(1.0, 0.5 * frame_width),
                -1.0,
                1.0,
            )
            dy_norm = self.clamp(
                float(status.get('move_dy_px', 0.0)) / max(1.0, 0.5 * frame_height),
                -1.0,
                1.0,
            )
            dz_norm = self.clamp(
                float(status.get('dz_rel', 0.0)),
                -1.0,
                1.0,
            )
        else:
            dx_norm = 0.0
            dy_norm = 0.0
            dz_norm = 0.0

        return {
            'vision_valid': bool(valid),
            'confidence': conf,
            'dx_normalized': float(dx_norm),
            'dy_normalized': float(dy_norm),
            'scale_error_normalized': float(dz_norm),
        }

    def push_latest_vision(self, session: requests.Session, last_pushed_seq: int) -> None:
        with self.shared_lock:
            seq = self.shared.frame_seq
            status = dict(self.shared.status)

        if not status or seq == last_pushed_seq:
            return

        packet = self.build_visual_packet(status)
        if self.controller_transport == 'udp':
            self.push_latest_vision_udp(packet)
        else:
            self.push_latest_vision_http(session, packet)

    def push_latest_vision_udp(self, packet: Dict[str, Any]) -> None:
        try:
            raw = json.dumps(packet).encode('utf-8')
            self.udp_sock.sendto(raw, self.controller_udp_addr)
        except OSError as exc:
            now = time.time()
            if (now - self._last_controller_error_log_ts) >= 1.0:
                print(f'[WARN] UDP controller vision push failed: {exc}')
                self._last_controller_error_log_ts = now

    def push_latest_vision_http(self, session: requests.Session, packet: Dict[str, Any]) -> None:
        try:
            response = session.post(
                f'{self.controller_url}/internal/vision',
                json=packet,
                timeout=self.controller_timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            now = time.time()
            if (now - self._last_controller_error_log_ts) >= 1.0:
                print(f'[WARN] HTTP controller vision push failed: {exc}')
                self._last_controller_error_log_ts = now

    def _open_direct_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.camera_fps)

        if not cap.isOpened():
            raise RuntimeError(f'Cannot open camera index {self.camera_index}')
        return cap

    def capture_loop(self) -> None:
        print(f'[INFO] Edge agent {self.agent_id} connected to {self.server_url}')

        shared_client: Optional[SharedMjpegCameraClient] = None
        cap: Optional[cv2.VideoCapture] = None
        last_seq = -1
        preview_interval = 1.0 / self.upload_fps
        next_preview_ts = 0.0

        if self.camera_stream_url:
            print(f'[INFO] Using shared camera stream: {self.camera_stream_url}')
            shared_client = SharedMjpegCameraClient(
                self.camera_stream_url,
                connect_timeout=self.stream_connect_timeout,
                read_timeout=self.stream_read_timeout,
                reconnect_delay=self.stream_reconnect_delay,
            )
            shared_client.start()
        else:
            print(f'[INFO] Using direct camera index={self.camera_index}, target={self.width}x{self.height}@{self.camera_fps}')
            cap = self._open_direct_camera()

        try:
            while not self.stop_event.is_set():
                if shared_client is not None:
                    ok, frame, last_seq = shared_client.read(timeout=1.0, last_seq=last_seq)
                else:
                    assert cap is not None
                    ok, frame = cap.read()

                if (not ok) or frame is None:
                    time.sleep(0.01)
                    continue

                self.apply_pending_commands(frame)

                now = time.time()
                need_preview = now >= next_preview_ts
                _, status = self.tracker.process_frame(frame, annotate=False)

                jpeg_bytes = None
                if need_preview:
                    preview_frame = self.build_preview_frame(frame, status)
                    encode_ok, buffer = cv2.imencode(
                        '.jpg',
                        preview_frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                    )
                    if encode_ok:
                        jpeg_bytes = buffer.tobytes()
                        next_preview_ts = now + preview_interval

                with self.shared_lock:
                    self.shared.frame_seq += 1
                    if jpeg_bytes is not None:
                        self.shared.jpeg_bytes = jpeg_bytes
                        self.shared.jpeg_seq = self.shared.frame_seq
                    self.shared.status = status
                    self.shared.width = int(frame.shape[1])
                    self.shared.height = int(frame.shape[0])
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            if shared_client is not None:
                shared_client.close()
                print('[INFO] Shared camera client closed.')
            if cap is not None:
                cap.release()
                print('[INFO] Camera released.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Edge dual-box optical-flow tracker agent using MJPEG from mjpeg_server.py and HTTP push to local controller')
    parser.add_argument('--server', default='http://127.0.0.1:5000', help='Host web server URL')
    parser.add_argument('--agent-id', default='edge-1', help='Agent identifier')
    parser.add_argument('--camera', type=int, default=2, help='Direct camera index')
    parser.add_argument('--camera-stream-url', default='http://127.0.0.1:8080', help='MJPEG HTTP URL exposed by mjpeg_server.py; empty string falls back to direct camera')
    parser.add_argument('--width', type=int, default=1280, help='Capture width for direct mode')
    parser.add_argument('--height', type=int, default=720, help='Capture height for direct mode')
    parser.add_argument('--camera-fps', type=int, default=30, help='Camera FPS hint for direct mode')
    parser.add_argument('--stream-connect-timeout', type=float, default=2.0, help='Shared camera connect timeout')
    parser.add_argument('--stream-read-timeout', type=float, default=5.0, help='Shared camera read timeout')
    parser.add_argument('--stream-reconnect-delay', type=float, default=1.0, help='Delay before reconnecting to shared stream')
    parser.add_argument('--upload-fps', type=float, default=10.0, help='JPEG upload FPS')
    parser.add_argument('--jpeg-quality', type=int, default=80, help='JPEG quality 30-95')
    parser.add_argument('--controller-url', default='http://127.0.0.1:17001', help='Local controller HTTP API base URL')
    parser.add_argument('--controller-timeout', type=float, default=0.25, help='Timeout when pushing local vision packets to controller')
    parser.add_argument('--controller-push-hz', type=float, default=20.0, help='Rate for pushing vision packets to local controller')
    parser.add_argument('--controller-transport', choices=['udp', 'http'], default='udp', help='Use UDP for low-latency local control, or HTTP for debugging/backward compatibility')
    parser.add_argument('--controller-udp-host', default='127.0.0.1', help='UDP host for local controller vision packets')
    parser.add_argument('--controller-udp-port', type=int, default=17002, help='UDP port for local controller vision packets')
    parser.add_argument('--vision-min-conf', type=float, default=0.50, help='Min tracker confidence for auto control')
    parser.add_argument('--vision-deadband', type=float, default=0.04, help='Deadband for auto command')
    parser.add_argument('--vision-k-forward', type=float, default=0.45, help='Gain from ey_norm -> forward')
    parser.add_argument('--vision-k-lateral', type=float, default=0.45, help='Gain from ex_norm -> lateral')
    parser.add_argument('--vision-k-vertical', type=float, default=0.30, help='Gain from ey_norm -> vertical, mainly for forward-facing camera pitch/height correction')
    parser.add_argument('--vision-k-range', type=float, default=0.30, help='Gain from scale-based ez_norm -> range hold command')
    parser.add_argument('--vision-range-deadband', type=float, default=0.03, help='Deadband for scale/range command')
    parser.add_argument('--vision-range-axis', choices=['forward', 'vertical', 'off'], default='forward', help='Which ROV axis receives optical-axis range correction from scale')
    parser.add_argument('--vision-range-min-conf', type=float, default=0.75, help='Min confidence before allowing scale/range command')
    parser.add_argument('--vision-range-max-abs', type=float, default=0.40, help='Reject range command if abs(ez_norm) is larger than this')
    parser.add_argument('--vision-max-bbox-area-ratio', type=float, default=0.35, help='Reject range/forward command if bbox occupies too much of the frame')
    parser.add_argument('--vision-max-cmd', type=float, default=0.35, help='Clamp auto command')
    args = parser.parse_args()

    stream_url = args.camera_stream_url.strip() or None

    agent = EdgeTrackerAgent(
        server_url=args.server,
        agent_id=args.agent_id,
        camera_index=args.camera,
        camera_stream_url=stream_url,
        width=args.width,
        height=args.height,
        camera_fps=args.camera_fps,
        stream_connect_timeout=args.stream_connect_timeout,
        stream_read_timeout=args.stream_read_timeout,
        stream_reconnect_delay=args.stream_reconnect_delay,
        upload_fps=args.upload_fps,
        jpeg_quality=args.jpeg_quality,
        controller_url=args.controller_url,
        controller_timeout=args.controller_timeout,
        controller_push_hz=args.controller_push_hz,
        controller_transport=args.controller_transport,
        controller_udp_host=args.controller_udp_host,
        controller_udp_port=args.controller_udp_port,
        vision_min_conf=args.vision_min_conf,
        vision_deadband=args.vision_deadband,
        vision_k_forward=args.vision_k_forward,
        vision_k_lateral=args.vision_k_lateral,
        vision_k_vertical=args.vision_k_vertical,
        vision_k_range=args.vision_k_range,
        vision_range_deadband=args.vision_range_deadband,
        vision_range_axis=args.vision_range_axis,
        vision_range_min_conf=args.vision_range_min_conf,
        vision_range_max_abs=args.vision_range_max_abs,
        vision_max_bbox_area_ratio=args.vision_max_bbox_area_ratio,
        vision_max_cmd=args.vision_max_cmd,
    )
    agent.run()


if __name__ == '__main__':
    main()
    