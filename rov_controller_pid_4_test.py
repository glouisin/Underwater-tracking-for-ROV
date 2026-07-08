from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink2

import time
import random
import socket
import threading
import serial
import numpy as np
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import sys
import termios
import tty
import select
import math
import re
import ms5837
from pca9685 import PCA9685
from MMC5983 import MMC5983
from ICM20602 import ICM20602
from collections import deque
from ADC import ADS1115
from EKF_ICM import EKF_ICM

from Si7021 import Si7021
#control codes sent from GCS via Manual Control Packet
LIGHT_SWITCH = 0x01
GAIN_INCREASE = 0x02
GAIN_DECREASE = 0x04
TILT_INCREASE = 0x08
TILT_DECREASE = 0x10
LIGHT_INCREASE = 0x20
LIGHT_DECREASE = 0x40
TILT_CENTER = 0x80
GRIPPER_OPEN = 0x100
GRIPPER_CLOSE = 0x200

Light_Status = 1
Camera_Tilt = 0
Thruster_Arm = 1

LIGHT_I2C_ADDR = 15
LIGHT_MAX_POWER = 1900
LIGHT_OFF_POWER = 1100
CAMERA_TILT_I2C_ADDR = 2
LightPower = 0

GRIPPER_ADDR = 14
GRIPPER_PWM_CLOSE = 1200
GRIPPER_PWM_OPEN = 1700

Kp, Ki, Kd = 2.5, 1, 0.7
Kpz, Kiz, Kdz = 2.0, 1.0, 0.5

hold_heading = False
hold_depth = False

exit_flag = False
hold_vision_x = False
hold_vision = False

latest_pitch = 0.0
latest_yaw = 0.0
latest_roll = 0.0
latest_depth = 0.0

current_mode = 2

#Nhiet do  , do am
temp = 0.0
humid = 0.0

#Nhiet do nuoc
WaterTemp = 0.0
#IMU data
xacc, yacc, zacc = 0.0, 0.0, 0.0
xgyro, ygyro, zgyro = 0.0, 0.0, 0.0
xmag, ymag, zmag = 0.0, 0.0, 0.0

desired_yaw = 0.0
desired_depth = 0.0

integral = 0.0
last_error = 0.0
integral_z = 0.0
last_error_z = 0.0

prev_x = 0
prev_y = 0
prev_z = 0
prev_r = 0

manual_vector = np.array([0.0, 0.0, 0.0, 0.0])
joystick_forward = 0.0
joystick_lateral = 0.0
joystick_ascend = 0.0
joystick_yaw = 0.0

VISION_API_HOST = '127.0.0.1'
VISION_API_PORT = 17001
VISION_UDP_HOST = '127.0.0.1'
VISION_UDP_PORT = 17002
VISION_TIMEOUT = 0.50
MANUAL_DEADBAND = 0.08
VISION_MANUAL_HOLDOFF = 0.25
AUTO_CONTROL_HZ = 20.0
AUTO_VISION_ENABLED = True

vision_valid = False
vision_forward = 0.0
vision_lateral = 0.0
vision_ex = 0.0
vision_ey = 0.0
vision_confidence = 0.0
vision_vertical = 0.0
vision_ez = 0.0
vision_dz_rel = 0.0
vision_raw_dz_rel = 0.0
vision_pair_dz_rel = 0.0
vision_range_cmd = 0.0
vision_range_axis = 'forward'
vision_mode = 'IDLE'
vision_bbox = None
vision_frame_width = 0.0
vision_frame_height = 0.0
vision_bbox_area_ratio = 0.0
last_vision_time = 0.0
last_manual_input_time = 0.0
last_vision_debug_log = 0.0
vision_lock = threading.Lock()

vision_move_x_norm = 0.0
vision_move_y_norm = 0.0

VISION_USE_CONTROLLER_PID = True

# Dấu mặc định dựa trên mapping hiện tại:
# current lateral_cmd = +K * ex_norm
# current range_cmd   = -K * ez_norm
VISION_LATERAL_SIGN = -1.0
VISION_RANGE_SIGN = -1.0
VISION_VERTICAL_SIGN = +1.0

Kpvx, Kivx, Kdvx = 0.35, 0.0, 0.02
Kpvz, Kivz, Kdvz = 0.08, 0.0, 0.0

VISION_PID_DEADBAND_X = 0.035
VISION_PID_DEADBAND_Z = 0.060
VISION_PID_MAX_LATERAL = 0.30
VISION_MIN_LATERAL_CMD = 0.12
VISION_PID_MAX_FORWARD = 0.10
VISION_MAX_YAW_CMD = 0.25
VISION_PID_MAX_LATERAL_STEP = 0.04
VISION_PID_MAX_FORWARD_STEP = 0.025
VISION_PID_I_LIMIT = 0.20
VISION_PID_D_ALPHA = 0.25          
VISION_PID_MAX_DT = 0.20

vision_pid_last_time = 0.0
vision_ix = 0.0
vision_iz = 0.0
vision_last_x = 0.0
vision_last_z = 0.0
vision_dx_filt = 0.0
vision_dz_filt = 0.0
vision_last_forward_cmd = 0.0
vision_last_lateral_cmd = 0.0

VISION_ENABLE_ASCEND = False
# Safety gate: if the selected object fills too much of the image, never allow
# positive forward thrust from vision. This helps prevent slow visual-scale drift
# from driving the ROV into the target.
VISION_MAX_BBOX_AREA_RATIO = 0.35

increment = 0.1

tune_heading = False
tune_depth = False
tune_log = []
tune_start_time = 0
tune_output_axis = 3

control_lock = threading.RLock()

#BEGIN THRUSTER CONTROLLER
thruster_controller = PCA9685(2, use_extclk=True)
# Mixing matrix 6x4
SQRT2_2 = 0.7071
mixing_matrix = np.array([
    [  1, -1,  0, -SQRT2_2],   # T1: forward-right (CW)
    [ -1, -1,  0, -SQRT2_2],   # T2: forward-left  (CW)
    [  1,  1,  0, -SQRT2_2],   # T3: rear-right    (CCW)
    [  1, -1,  0, +SQRT2_2],   # T4: rear-left     (CCW)
    [  0,  0,  1,        0],   # T5: vertical forward-right (CW)
    [  0,  0, -1,        0],   # T6: vertical forward-left  (CCW)
    [  0,  0, -1,        0],   # T7: vertical rear-right  (CCW)
    [  0,  0,  1,        0],   # T8: vertical forward-left  (CCW)
])
# THRUSTER_MAP = [9, 10, 4, 6, 7, 8, 5, 11]
THRUSTER_MAP = [0]*8

ESC_GAIN = 0.5
PowerLevel = (ESC_GAIN*100)
ESC_MAX_POWER = 400
ESC_NEUTRAL = 1500

CONFIG_PATH = "thruster_config.json"

THRUSTER_INDEX_MAP = {
    "forward-right": 0,
    "forward-left": 1,
    "rear-right": 2,
    "rear-left": 3,
    "vertical-forward-right": 4,
    "vertical-forward-left": 5,
    "vertical-rear-right": 6,
    "vertical-rear-left": 7
}

def load_thrust_config():
    global THRUSTER_MAP
    if not os.path.exists(CONFIG_PATH):
        print(f"[WARN] Config not found: {CONFIG_PATH}, using default THRUSTER_MAP")
        return

    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)

        new_map = list(THRUSTER_MAP)  
        for name, idx in THRUSTER_INDEX_MAP.items():
            if name in config:
                new_map[idx] = config[name]
            else:
                print(f"[WARN] Missing key '{name}' in config, keeping default channel {THRUSTER_MAP[idx]}")

        THRUSTER_MAP = new_map
        print("Loaded thruster config:", THRUSTER_MAP)

    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {CONFIG_PATH}: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to load thruster config: {e}")

def udp_config_listener():

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    server_sock.bind(("0.0.0.0", 12345))  # Bind to all interfaces on port 12345
    print("UDP config listener started on port 12345")
    while True:
        try:
            data, addr = server_sock.recvfrom(1024)
            # Process the received configuration data
            message = data.decode('utf-8').strip()

            if message.startswith("PING:"):
                channel_name = int(message.split(":", 1)[1])
                print(f"Received PING from {addr} for channel {channel_name}")

                thruster_controller.channel_set_pwm(channel_name, 1650)
                time.sleep(1.0)
                thruster_controller.channel_set_pwm(channel_name, ESC_NEUTRAL)

            elif message.startswith("SAVE_CONFIG:"):
                json_string = message.split(":", 1)[1]
                print(" [UDP-CONFIG] Nhận cấu hình mới từ GCS. Tiến hành giải mã...")

                parsed_json = json.loads(json_string)
                with open(CONFIG_PATH, "w") as f:
                    json.dump(parsed_json, f, indent=4)
                print(f"[UDP-CONFIG] Đã lưu cấu hình mới vào file: {CONFIG_PATH}")    

                load_thrust_config()    
        except Exception as e:
            print(f"[UDP-CONFIG] Lỗi xử lý cấu hình: {e}")        

# Gọi ngay khi khởi động
load_thrust_config()
#END THRUSTER CONTROLLER

masterin = None
addr = None
masterout_hbt = None
masterout_sta = None
masterout_pos = None
masterout_imu = None
masterout_att = None

#sensors for heading calculation
#for old heading
#imu = ICM20602(spi_bus=0, spi_dev=3)
#imu.initialize()
#mmc = MMC5983(spi_bus=0, spi_dev=0)
#mmc.reset()

#EKF heading (Cuong)
icm = EKF_ICM(spi_bus=0, spi_dev=3)
icm.initialize()
icm.calibrate_gyro()
#icm.calibrate_mag()
icm.ekf_filter.mag = True
icm.ekf_filter.frequency = 1000
print("✅ IMU initialized")
print("WHO_AM_I:", hex(icm.whoami()))

icm_q = np.array([1.0, 0.0, 0.0, 0.0])
last_icm_time = time.perf_counter()

#ICM2
icm2 = EKF_ICM(spi_bus=0, spi_dev=3)
icm2.initialize()
icm2.calibrate_gyro()
icm2.ekf_filter.frequency = 1000

icm_q2 = np.array([1.0, 0.0, 0.0, 0.0])
#Si7021 Init:
temp_humid = Si7021(port='/dev/ttyHS3')

batt = ADS1115()

#running average buffer for heading filtering
window_size = 4   # số mẫu để tính trung bình trượt
heading_buffer = deque(maxlen=window_size)

last_time = time.time()

def apply_min_command(u: float, min_cmd: float) -> float:
    if abs(u) < 1e-6:
        return 0.0
    return math.copysign(max(abs(u), min_cmd), u)

# ===================== PID AUTOTUNE (Relay Åström–Hägglund) =====================
def _wrap_angle(rad):
    return math.atan2(math.sin(rad), math.cos(rad))

def auto_tune_pid(
    axis: str,                      # 'heading' hoặc 'depth'
    *,
    relay_amplitude: float = 0.20,  # biên độ on/off cho lệnh điều khiển
    hysteresis: float = 0.02,       # deadband quanh setpoint để giảm chatter
    warmup_time: float = 2.0,       # thời gian làm nóng (s)
    tune_time: float = 12.0,        # thời gian thu thập dao động (s)
    sample_time: float = 0.02,      # chu kỳ mẫu (s)
    min_period: float = 0.3,        # bỏ dao động quá nhanh
    max_period: float = 15.0,       # bỏ dao động quá chậm / outlier
    rule: str = "zn_pid",           # 'zn_p' | 'zn_pi' | 'zn_pid'
    clamp: float = 0.5,             # giới hạn biên đầu ra khi tune (đơn vị lệnh [-1..1])
    apply_result: bool = True       # True -> gán vào (Kp,Ki,Kd) hoặc (Kpz,Kiz,Kdz)
) -> dict:
    """
    Tự động tune PID cho 'heading' (yaw) hoặc 'depth' bằng relay on/off.
    Trả về dict: {ok, Ku, Tu, A, Kp, Ki, Kd, samples}. Nếu apply_result=True, gán vào hệ số PID toàn cục.
    """
    global latest_yaw, latest_depth
    global Kp, Ki, Kd, Kpz, Kiz, Kdz

    # ==== Chọn sensor, trục viết lệnh, và cách tính error ====
    if axis.lower() == "heading":
        is_angle = True
        read_measurement = lambda: latest_yaw
        write_axis_index = 3   # control_vector[3] = yaw
    elif axis.lower() == "depth":
        is_angle = False
        read_measurement = lambda: latest_depth
        write_axis_index = 2   # control_vector[2] = ascend
    else:
        return {"ok": False, "reason": "axis must be 'heading' or 'depth'"}

    # ==== Setpoint mặc định = giá trị hiện tại ====
    setpoint = read_measurement()

    def compute_error(y):
        e = setpoint - y
        return _wrap_angle(e) if is_angle else e

    # ==== Hàm ghi lệnh chỉ trên 1 trục (các trục khác = 0) ====
    def write_axis_output(val):
        vec = manual_vector.copy()
        vec[:] = 0.0
        # clamp biên để an toàn
        u = max(-clamp, min(clamp, val))
        vec[write_axis_index] = u
        thrusts = mixing_matrix @ vec
        # chuẩn hóa nếu cần
        #m = float(np.max(np.abs(thrusts))) if np.size(thrusts) else 1.0
        #if m > 1.0:
        #    thrusts = thrusts / m
        send_thrust_pwm(thrusts)

    # ==== Khởi động ====
    dt = float(sample_time)
    u_hi = +abs(relay_amplitude)
    u_lo = -abs(relay_amplitude)
    u = u_hi

    # Warmup để về gần limit cycle
    t0 = time.time()
    while time.time() - t0 < warmup_time:
        y = read_measurement()
        e = compute_error(y)
        if u == u_hi and e > hysteresis:
            u = u_lo
        elif u == u_lo and e < -hysteresis:
            u = u_hi
        write_axis_output(u)
        time.sleep(dt)

    # ==== Đo dao động ====
    zero_cross_times = []
    peak_vals = []
    last_peak_val = None
    last_peak_time = None
    prev_sign = None
    samples = 0

    t_start = time.time()
    while time.time() - t_start < tune_time:
        y = read_measurement()
        e = compute_error(y)

        # Relay switching
        if u == u_hi and e > hysteresis:
            u = u_lo
        elif u == u_lo and e < -hysteresis:
            u = u_hi
        write_axis_output(u)

        # Zero-crossing của error
        sign = 1 if e >= 0 else -1
        if prev_sign is not None and sign != prev_sign:
            zero_cross_times.append(time.time())
        prev_sign = sign

        # Peak detection (trên |e|)
        ae = abs(e)
        if (last_peak_val is None) or (ae > last_peak_val):
            last_peak_val = ae
            last_peak_time = time.time()
        # “đóng” peak sau 5 mẫu kể từ khi đạt đỉnh
        if last_peak_time is not None and (time.time() - last_peak_time) > (5 * dt):
            peak_vals.append(last_peak_val)
            last_peak_val = None
            last_peak_time = None

        samples += 1
        time.sleep(dt)

    # ==== Ước lượng chu kỳ Tu từ zero-crossing ====
    periods = []
    for i in range(2, len(zero_cross_times)):
        # hai lần zero-cross cách nhau ~1 chu kỳ
        period = zero_cross_times[i] - zero_cross_times[i - 2]
        if min_period <= period <= max_period:
            periods.append(period)

    Tu = (sum(periods) / len(periods)) if periods else None
    A = (sum(peak_vals) / len(peak_vals)) if peak_vals else None

    if Tu is None or A is None or A <= 1e-9:
        # Trả về 0 lệnh
        send_thrust_pwm([0]*6)
        return {"ok": False, "reason": "Insufficient oscillation data", "Tu": Tu, "A": A, "samples": samples}

    # ==== Tính Ku từ relay: Ku = 4*d / (pi*A) ====
    d = abs(relay_amplitude)
    Ku = 4.0 * d / (math.pi * A)

    # ==== Quy tắc Ziegler–Nichols ====
    if rule == "zn_p":
        Kp_new, Ki_new, Kd_new = 0.50 * Ku, 0.0, 0.0
    elif rule == "zn_pi":
        Kp_new = 0.45 * Ku
        Ti = Tu / 1.2
        Ki_new = Kp_new / Ti
        Kd_new = 0.0
    else:  # "zn_pid"
        Kp_new = 0.60 * Ku
        Ti = Tu / 2.0
        Td = Tu / 8.0
        Ki_new = Kp_new / Ti
        Kd_new = Kp_new * Td

    # Ngừng motor sau khi tune
    send_thrust_pwm([0]*6)

    # ==== Áp dụng vào biến global nếu muốn ====
    if apply_result:
        if axis.lower() == "heading":
            Kp, Ki, Kd = float(Kp_new), float(Ki_new), float(Kd_new)
            globals()["Kp"], globals()["Ki"], globals()["Kd"] = Kp, Ki, Kd
            print(f"✅ Tuned HEADING PID -> Kp={Kp:.3f}, Ki={Ki:.3f}, Kd={Kd:.3f} | Ku={Ku:.3f}, Tu={Tu:.3f}")
        else:
            Kpz, Kiz, Kdz = float(Kp_new), float(Ki_new), float(Kd_new)
            globals()["Kpz"], globals()["Kiz"], globals()["Kdz"] = Kpz, Kiz, Kdz
            print(f"✅ Tuned DEPTH PID -> Kp={Kpz:.3f}, Ki={Kiz:.3f}, Kd={Kdz:.3f} | Ku={Ku:.3f}, Tu={Tu:.3f}")

    return {
        "ok": True, "axis": axis,
        "Ku": Ku, "Tu": Tu, "A": A,
        "Kp": float(Kp_new), "Ki": float(Ki_new), "Kd": float(Kd_new),
        "samples": samples
    }
# =================== END PID AUTOTUNE ===================

def gripper_open():
    thruster_controller.channel_set_pwm(GRIPPER_ADDR, GRIPPER_PWM_OPEN)

def gripper_close():
    thruster_controller.channel_set_pwm(GRIPPER_ADDR, GRIPPER_PWM_CLOSE)

def set_camera_tilt():
    global thruster_controller
    tiltpwm = int(1500 + Camera_Tilt * 400 / 45.0)
    thruster_controller.channel_set_pwm(CAMERA_TILT_I2C_ADDR, tiltpwm)

def set_light_status():
    global Light_Status
    lightpwm = int(LightPower * (LIGHT_MAX_POWER - LIGHT_OFF_POWER) / 100 + LIGHT_OFF_POWER)
    if Light_Status == 1:
        thruster_controller.channel_set_pwm(LIGHT_I2C_ADDR, lightpwm)
    else:
        thruster_controller.channel_set_pwm(LIGHT_I2C_ADDR, LIGHT_OFF_POWER)

def icm_normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v
#Ham doc Temp, humid uart
def temp_humid_reader():
    global temp, humid , temp_humid

    try:
        while True:
            current_data = temp_humid.get_latest_data()
            if current_data:
                temp, humid = current_data
                # print(f"Temp: {temp} , Humid: {humid}")
            time.sleep(0.01)
    except KeyboardInterrupt:
        temp_humid.close()


def heading_reader_ekf():
    global latest_yaw, exit_flag, latest_pitch, latest_roll, last_icm_time, icm_q , icm_q2, xacc, yacc, zacc, xgyro, ygyro, zgyro, xmag, ymag, zmag

    accel, gyro = icm.read_accel_gyro()
    mag = icm.read_mag()
    ax, ay, az = accel
    gx, gy, gz = gyro
    mx, my, mz = mag

    acc_data = np.array([ay, ax, -az], dtype=np.float64)
    gyr_data = np.array([gy, gx, -gz], dtype=np.float64)
    mag_data = np.array([my, mx, -mz], dtype=np.float64)

    acc_data = icm_normalize(acc_data)
    mag_data = icm_normalize(mag_data)

    now = time.perf_counter()
    dt = now - last_icm_time
    last_icm_time = now
    icm.ekf_filter.Dt = dt

    try:
        icm_q = icm.ekf_filter.update(
            q=icm_q,
            gyr=gyr_data,
            acc=acc_data,
            mag=mag_data   #mag có vấn đè??
        )
    except Exception as e:
        print(f"⚠️ EKF update error: {e}")

    if icm_q is not None:
        pitch, roll, yaw = icm.quaternion_to_euler_NED(icm_q)

    pitch0, roll0 = icm.compute_pitch_roll(ax, ay, ax)

    try:
        while not exit_flag:
            accel, gyro = icm.read_accel_gyro()
            mag = icm.read_mag()
            ax, ay, az = accel
            gx, gy, gz = gyro
            mx, my, mz = mag

            xacc, yacc, zacc = ax, ay, az
            xgyro, ygyro, zgyro = gx, gy, gz
            xmag, ymag, zmag = mx, my, mz


            acc_data = np.array([ay, ax, -az], dtype=np.float64)
            gyr_data = np.array([gy, gx, -gz], dtype=np.float64)
            mag_data = np.array([my, mx, -mz], dtype=np.float64)

            acc_data = icm_normalize(acc_data)
            mag_data = icm_normalize(mag_data)

            now = time.perf_counter()
            dt = now - last_icm_time
            last_icm_time = now
            icm.ekf_filter.Dt = dt

            try:
                icm_q = icm.ekf_filter.update(
                    q=icm_q,
                    gyr=gyr_data,
                    acc=acc_data,
                    mag=mag_data   #mag có vấn đè??
                )
            except Exception as e:
                print(f"⚠️ EKF update error: {e}")
                continue

            try:
                icm_q2 = icm2.ekf_filter.update(
                    q=icm_q2,
                    gyr=gyr_data,
                    acc=acc_data
                 )
            except Exception as e:
                print(f"flase icm2: {e}")
                continue

            if icm_q is not None:
                pitch, roll, yaw = icm.quaternion_to_euler_NED(icm_q)
                #pitch, roll = icm.compute_pitch_roll(ax, ay, ax)
                pitch2, roll2, yaw_temp = icm2.quaternion_to_euler_NED(icm_q2)
                # print(f"Pitch: {math.degrees(pitch2):6.2f}° | "
                #     f"Roll: {math.degrees(roll2):6.2f}° | "
                #     f"Yaw: {math.degrees(yaw):6.2f}°")
            else:
                print("⚠️ EKF update failed")

            #latest_pitch = (pitch - pitch0)
            #latest_roll = (roll - roll0)
            latest_pitch = pitch2
            latest_roll = roll2
            latest_yaw = (-yaw)
    except KeyboardInterrupt:
        print("⛔ Lỗi đọc heading hoặc Kết thúc.")

""" def heading_reader():
    global exit_flag, latest_yaw, latest_pitch, latest_roll
    try:
        while not exit_flag:
            accel, gyro = imu.read_accel_gyro()
            ax, ay, az = [round(v, 3) for v in accel]
            gx, gy, gz = [round(v, 2) for v in gyro]
            pitch, roll = imu.compute_pitch_roll(ax, ay, -az)

            #print(f"Accel [g]: x={ax}, y={ay}, z={az} | Pitch: {round(pitch,2)}°, Roll: {round(roll,2)}°")
            #print(f"Gyro  [dps]: x={gx}, y={gy}, z={gz}")

            mag = mmc.measure()
            if mag:
                #print("Magnetometer [Gauss]:", mag)
                heading = mmc.calculate_heading(mag, pitch=pitch, roll=roll)  # Thêm pitch/roll nếu có IMU
                heading_buffer.append(heading)

                # Tính trung bình trượt
                avg_heading = sum(heading_buffer) / len(heading_buffer)
                #print(f"Heading [deg]: {heading:.2f} | Running Avg [{len(heading_buffer)}]: {avg_heading:.2f}")
            else:
                print("⚠️ Không có dữ liệu")

            latest_pitch = math.radians(pitch)
            latest_roll = math.radians(roll)
            latest_yaw = math.radians(avg_heading)
            time.sleep(0.05)  # đọc 20 Hz
    except KeyboardInterrupt:
        print("⛔ Lỗi đọc heading hoặc Kết thúc.")
    finally:
        mmc.close()
        imu.close() """

def depth_reader():
    global latest_depth, exit_flag, WaterTemp
    try:
        #Cảm biến độ sâu tại I2C3
        sensor = ms5837.MS5837_30BA(bus=3)
        if not sensor.init():
            print("❌ Failed to initialize MS5837 sensor")
           # exit_flag = True
            return
        sensor.setFluidDensity(ms5837.DENSITY_SALTWATER)  # seawater
        print("✅ Depth sensor initialized (MS5837-30BA)")
    except Exception as e:
        print("❌ Depth sensor error:", e)
        #exit_flag = True
        return

    while not exit_flag:
        try:
            if sensor.read():
                latest_depth = sensor.depth()
                WaterTemp = sensor.temperature()  # Cập nhật nhiệt độ nước
                # print("Depth: ", latest_depth)
                # print("Water Temperature: ", WaterTemp)
            time.sleep(0.05)
        except:
            time.sleep(1)
            continue

def send_thrust_pwm(thrusts):
    #print("Send thrust" );
    for i, t in enumerate(thrusts):
        power = int(np.clip(t, -1, 1) * ESC_GAIN * ESC_MAX_POWER) + ESC_NEUTRAL
        thruster_controller.channel_set_pwm(THRUSTER_MAP[i], power)
        #print(THRUSTER_MAP[i], power)

HEATBEAT_TIMOUT = 5.0
lasttime_heartbeat = time.time()
is_lost_connection = False

#===================== ROV HOLD VISION =====================#
def reset_vision_pid_memory():
    global vision_pid_last_time
    global vision_ix, vision_iz
    global vision_last_x, vision_last_z
    global vision_dx_filt, vision_dz_filt
    global vision_last_forward_cmd, vision_last_lateral_cmd

    vision_pid_last_time = 0.0
    vision_ix = 0.0
    vision_iz = 0.0
    vision_last_x = 0.0
    vision_last_z = 0.0
    vision_dx_filt = 0.0
    vision_dz_filt = 0.0
    vision_last_forward_cmd = 0.0
    vision_last_lateral_cmd = 0.0

def reset_pid_memory():
    global integral, last_error, integral_z, last_error_z, last_time
    integral = 0.0
    last_error = 0.0
    integral_z = 0.0
    last_error_z = 0.0
    last_time = time.time()
    reset_vision_pid_memory()

def clamp_unit(v: float, limit: float = 1.0) -> float:
    return max(-limit, min(limit, float(v)))


def slew_limit(current: float, previous: float, max_step: float) -> float:
    delta = _clamp_abs(float(current) - float(previous), max(0.0, float(max_step)))
    return float(previous) + delta


def update_vision_from_packet(msg: dict):
    global vision_valid, vision_confidence, last_vision_time
    global vision_move_x_norm, vision_move_y_norm, vision_ez

    with vision_lock:
        now = time.time()

        vision_valid = bool(msg.get('vision_valid', False))
        vision_confidence = float(msg.get('confidence', 0.0))

        if vision_valid:
            vision_move_x_norm = clamp_unit(msg.get('dx_normalized', 0.0), 1.0)
            vision_move_y_norm = clamp_unit(msg.get('dy_normalized', 0.0), 1.0)
            vision_ez = clamp_unit(msg.get('scale_error_normalized', 0.0), 1.0)
        else:
            vision_move_x_norm = 0.0
            vision_move_y_norm = 0.0
            vision_ez = 0.0
        print(msg.get('dx_normalized', 0.0))
        last_vision_time = now


def get_vision_snapshot() -> dict:
    with vision_lock:
        return {
            'vision_valid': vision_valid,
            'vision_confidence': vision_confidence,
            'last_vision_time': last_vision_time,
            'vision_move_x_norm': vision_move_x_norm,
            'vision_move_y_norm': vision_move_y_norm,
            'vision_ez': vision_ez,
        }


class InternalVisionApiHandler(BaseHTTPRequestHandler):
    server_version = 'ROVVisionAPI/1.0'

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/internal/health':
            self._send_json({'ok': True, 'service': 'rov_controller_pid_dual_internal_api'})
            return
        if self.path == '/internal/vision':
            payload = get_vision_snapshot()
            payload['ok'] = True
            self._send_json(payload)
            return
        self._send_json({'ok': False, 'error': 'not found'}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path != '/internal/vision':
            self._send_json({'ok': False, 'error': 'not found'}, HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            content_length = 0

        if content_length <= 0:
            self._send_json({'ok': False, 'error': 'empty body'}, HTTPStatus.BAD_REQUEST)
            return

        try:
            raw = self.rfile.read(content_length)
            msg = json.loads(raw.decode('utf-8'))
        except Exception as exc:
            self._send_json({'ok': False, 'error': f'invalid json: {exc}'}, HTTPStatus.BAD_REQUEST)
            return

        update_vision_from_packet(msg)
        self._send_json({'ok': True, 'ts': time.time()})

    def log_message(self, format, *args):
        return


def internal_vision_api_server():
    server = ThreadingHTTPServer((VISION_API_HOST, VISION_API_PORT), InternalVisionApiHandler)
    server.timeout = 0.5
    print(f"🎯 Internal vision API listening on http://{VISION_API_HOST}:{VISION_API_PORT}")
    try:
        while not exit_flag:
            server.handle_request()
    finally:
        server.server_close()



def udp_vision_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((VISION_UDP_HOST, VISION_UDP_PORT))
    sock.settimeout(0.5)
    print(f"🎯 UDP vision receiver listening on {VISION_UDP_HOST}:{VISION_UDP_PORT}")

    try:
        while not exit_flag:
            try:
                raw, _ = sock.recvfrom(4096)
                msg = json.loads(raw.decode('utf-8'))
                update_vision_from_packet(msg)
            except socket.timeout:
                continue
            except Exception as exc:
                print(f"⚠️ UDP vision packet error: {exc}")
    finally:
        sock.close()


def auto_control_loop():
    global exit_flag
    global joystick_forward, joystick_lateral, joystick_ascend, joystick_yaw

    period = 1.0 / AUTO_CONTROL_HZ
    while not exit_flag:
        control_thruster(joystick_forward, joystick_lateral, joystick_ascend, joystick_yaw)
        time.sleep(period)  

def control_loop():
    global prev_x, prev_y, prev_z, prev_r, Light_Status, Camera_Tilt, ESC_GAIN, LightPower, PowerLevel
    global joystick_forward, joystick_lateral, joystick_ascend, joystick_yaw
    global lasttime_heartbeat, is_lost_connection

    #IP v4 only
    masterin = mavutil.mavlink_connection('udpin:0.0.0.0:16001')

    #IP v6 and v4 (dual-stack)
    #masterin = mavutil.mavlink_connection('udpin:[::]:16001')

    while True:
        msg = masterin.recv_match(blocking=False)

        current_time = time.time()
        if msg is not None and msg.get_type() != 'BAD_DATA':
            is_lost_connection = False
            if msg.get_type() == 'HEARTBEAT':
                lasttime_heartbeat = current_time
                print("Received heartbeat from drone")
            elif msg.get_type() == 'MANUAL_CONTROL':
                joystick_forward = msg.y/1000.0
                joystick_lateral = msg.x/1000.0
                joystick_ascend = msg.z/1000.0
                joystick_yaw = msg.r/1000.0
                control_thruster(joystick_forward, joystick_lateral, joystick_ascend, joystick_yaw)

                if msg.buttons & GAIN_INCREASE != 0:
                    ESC_GAIN = ESC_GAIN + 0.1
                    PowerLevel = ESC_GAIN * 100
                    if ESC_GAIN > 0.95:
                        ESC_GAIN = 0.95
                    PowerLevel = ESC_GAIN * 100
                if msg.buttons & GAIN_DECREASE != 0:
                    ESC_GAIN = ESC_GAIN - 0.1
                    if ESC_GAIN < 0:
                        ESC_GAIN = 0
                    PowerLevel = ESC_GAIN * 100
                if msg.buttons & LIGHT_SWITCH != 0:
                    Light_Status = 1- Light_Status
                    set_light_status()
                if msg.buttons & TILT_INCREASE != 0:
                    Camera_Tilt = Camera_Tilt + 1
                    if Camera_Tilt > 45:
                        Camera_Tilt = 45
                    set_camera_tilt()
                if msg.buttons & TILT_DECREASE != 0:
                    Camera_Tilt = Camera_Tilt - 1
                    if Camera_Tilt < -45:
                        Camera_Tilt = -45
                    set_camera_tilt()
                if msg.buttons & TILT_CENTER != 0:
                    Camera_Tilt = 0
                    set_camera_tilt()
                if msg.buttons & LIGHT_INCREASE != 0:
                    LightPower = LightPower + 10
                    if LightPower > 100:
                        LightPower = 100
                if msg.buttons & LIGHT_DECREASE != 0:
                    LightPower = LightPower - 10
                    if LightPower < 0:
                        LightPower = 0
                if msg.buttons & GRIPPER_OPEN != 0:
                    gripper_open()
                if msg.buttons & GRIPPER_CLOSE != 0:
                    gripper_close()
                set_light_status()
            elif msg.get_type() == 'SET_MODE':
                process_command(msg.custom_mode)
            else:
                print("Unknown message type: " + msg.get_type())
        elif msg is None:
            time.sleep(0.2)

        if (current_time - lasttime_heartbeat) > HEATBEAT_TIMOUT and not is_lost_connection:
            is_lost_connection = True
            joystick_forward = 0.0
            joystick_lateral = 0.0
            joystick_ascend = 0.0
            joystick_yaw = 0.0
            print("Khong nhan duoc heartbeat tu GCS trong 5 giay, chuyen ve manual mode de an toan")
            process_command(2)
            send_thrust_pwm([0]*8)
            lasttime_heartbeat = current_time

def process_command(button_code):
    global Thruster_Arm, hold_heading, hold_depth, hold_vision, exit_flag
    global latest_yaw, latest_depth, desired_yaw, desired_depth, current_mode

    if (button_code == 4):      # X - depth hold
        if hold_vision:
            hold_heading = False
        hold_vision = False
        reset_vision_pid_memory()
        if not hold_depth:
            Thruster_Arm = 1
            desired_depth = latest_depth
            hold_depth = True
            current_mode = 4
        elif hold_depth:
            Thruster_Arm = 1
            hold_depth = False
            current_mode = 2
        print(f"Hold depth at {desired_depth:.2f} m")
        # print(f"🔒 Hold depth at {desired_depth:.2f} m")

    elif (button_code == 2):    # B - manual
        Thruster_Arm = 1
        hold_heading = False
        hold_depth = False
        hold_vision = False
        current_mode = 2
        reset_vision_pid_memory()
        print("🔓 Manual mode")

    elif (button_code == 1):    # Y - heading hold
        hold_vision = False
        reset_vision_pid_memory()
        if not hold_heading:
            Thruster_Arm = 1
            desired_yaw = latest_yaw
            hold_heading = True
            current_mode = 1
        elif hold_heading:
            Thruster_Arm = 1
            hold_heading = False
            current_mode = 2
        print(f"Hold heading at {math.degrees(desired_yaw):.1f} deg")
        # print(f"🔒 Hold heading at {math.degrees(desired_yaw):.1f}°")
    elif (button_code == 3):
        if not hold_vision:
            Thruster_Arm = 1
            desired_yaw = latest_yaw
            hold_heading = True
            hold_vision = True
            current_mode = 3
            reset_pid_memory()
            print("Hold vision")
        elif hold_vision:
            Thruster_Arm = 1
            hold_heading = False
            hold_vision = False
            current_mode = 2
            reset_pid_memory()
            print("Vision hold off")
        return

def init_thruster():
    thruster_controller.set_pwm_frequency(50)
    thruster_controller.output_enable()

    #arm all ESCs
    send_thrust_pwm([0]*8)

def control_thruster(forward, lateral, ascend, yaw, treat_input_as_manual=True):
    with control_lock:
        return _control_thruster_impl(
            forward, lateral, ascend, yaw,
            treat_input_as_manual=treat_input_as_manual
        )

def _apply_deadband(e: float, db: float) -> float:
    return 0.0 if abs(e) < db else float(e)


def _clamp_abs(v: float, limit: float) -> float:
    return max(-limit, min(limit, float(v)))


def compute_vision_hold_commands(snapshot: dict, now: float):
    global vision_pid_last_time
    global vision_ix, vision_iz
    global vision_last_x, vision_last_z
    global vision_dx_filt, vision_dz_filt
    global vision_last_forward_cmd, vision_last_lateral_cmd

    dx_normalized = float(snapshot.get('vision_move_x_norm', 0.0))
    scale_error_normalized = float(snapshot.get('vision_ez', 0.0))

    if vision_pid_last_time <= 0.0:
        dt = 1.0 / AUTO_CONTROL_HZ
    else:
        dt = now - vision_pid_last_time
    vision_pid_last_time = now

    if dt <= 0.0 or dt > VISION_PID_MAX_DT:
        dt = 1.0 / AUTO_CONTROL_HZ
        vision_dx_filt = 0.0
        vision_dz_filt = 0.0

    # dx > 0 means the target moved to the right in the image. The default
    # sign keeps the current mapping: lateral command must drive it back to 0.
    error_x = _apply_deadband(VISION_LATERAL_SIGN * dx_normalized, VISION_PID_DEADBAND_X)
    derivative_x = (error_x - vision_last_x) / dt
    vision_dx_filt = (1.0 - VISION_PID_D_ALPHA) * vision_dx_filt + VISION_PID_D_ALPHA * derivative_x

    vision_ix = _clamp_abs(vision_ix + error_x * dt, VISION_PID_I_LIMIT)
    vision_last_x = error_x
    u_lateral_unsat = Kpvx * error_x + Kivx * vision_ix + Kdvx * vision_dx_filt
    u_lateral = _clamp_abs(u_lateral_unsat, VISION_PID_MAX_LATERAL)
    if abs(u_lateral_unsat) > VISION_PID_MAX_LATERAL and Kivx > 0.0:
        vision_ix = _clamp_abs(vision_ix - error_x * dt, VISION_PID_I_LIMIT)
    u_lateral = apply_min_command(u_lateral, VISION_MIN_LATERAL_CMD)
    u_lateral = slew_limit(u_lateral, vision_last_lateral_cmd, VISION_PID_MAX_LATERAL_STEP)
    vision_last_lateral_cmd = u_lateral

    # scale_error > 0 means the target looks larger than at selection time,
    # so the default sign commands backward motion.
    error_scale = _apply_deadband(VISION_RANGE_SIGN * scale_error_normalized, VISION_PID_DEADBAND_Z)
    derivative_scale = (error_scale - vision_last_z) / dt
    vision_dz_filt = (1.0 - VISION_PID_D_ALPHA) * vision_dz_filt + VISION_PID_D_ALPHA * derivative_scale

    vision_iz = _clamp_abs(vision_iz + error_scale * dt, VISION_PID_I_LIMIT)
    vision_last_z = error_scale
    u_forward_unsat = Kpvz * error_scale + Kivz * vision_iz + Kdvz * vision_dz_filt
    u_forward = _clamp_abs(u_forward_unsat, VISION_PID_MAX_FORWARD)
    if abs(u_forward_unsat) > VISION_PID_MAX_FORWARD and Kivz > 0.0:
        vision_iz = _clamp_abs(vision_iz - error_scale * dt, VISION_PID_I_LIMIT)
    u_forward = slew_limit(u_forward, vision_last_forward_cmd, VISION_PID_MAX_FORWARD_STEP)
    vision_last_forward_cmd = u_forward

    return u_forward, u_lateral

def _control_thruster_impl(forward, lateral, ascend, yaw, treat_input_as_manual=True):
    global Thruster_Arm, hold_heading, hold_depth, exit_flag, hold_vision
    global latest_yaw, latest_depth, desired_yaw, desired_depth
    global integral, last_error, integral_z, last_error_z, last_time
    global last_manual_input_time

    if Thruster_Arm == 0:
        print("--- DISARM")
        send_thrust_pwm([0] * 8)
        return

    now = time.time()
    dt = now - last_time
    last_time = now

    manual_vector[0] = forward
    manual_vector[1] = lateral
    manual_vector[2] = ascend
    manual_vector[3] = yaw

    if treat_input_as_manual:
        manual_active = (
            abs(forward) > MANUAL_DEADBAND or
            abs(lateral) > MANUAL_DEADBAND or
            abs(ascend) > MANUAL_DEADBAND or
            abs(yaw) > MANUAL_DEADBAND
        )
        if manual_active:
            last_manual_input_time = now
    else:
        manual_active = False

    control_vector = manual_vector.copy()

    snapshot = get_vision_snapshot()
    vision_fresh = snapshot['vision_valid'] and ((now - snapshot['last_vision_time']) <= VISION_TIMEOUT)
    manual_recent = (now - last_manual_input_time) <= VISION_MANUAL_HOLDOFF
    vision_ready = AUTO_VISION_ENABLED and (not manual_active) and (not manual_recent) and vision_fresh
    vision_active = vision_ready and hold_vision

    # Vision only takes over axes when the pilot is not actively moving the sticks.
    # The agent already decides whether scale/range correction is mixed into
    # forward_cmd or vertical_cmd using --vision-range-axis.
    if manual_active or manual_recent or (not vision_fresh) or (not hold_vision):
        reset_vision_pid_memory()

    if vision_active:
        if VISION_USE_CONTROLLER_PID:
            if not hold_heading:
                desired_yaw = latest_yaw
                hold_heading = True
                integral = 0.0
                last_error = 0.0

            vision_forward_cmd, vision_lateral_cmd = compute_vision_hold_commands(snapshot, now)
            control_vector[0] = vision_forward_cmd
            control_vector[1] = vision_lateral_cmd
        # Safety: nếu object quá lớn trong ảnh, không cho tiến tới nữa

        if VISION_ENABLE_ASCEND and (not hold_depth):
            # Chưa khuyến nghị bật ngay. Nếu bật, cũng nên viết PD riêng cho move_y_norm.
            control_vector[2] = 0.0

        # Safety: if the object is already very large in the frame, do not allow
        # vision to command positive forward thrust. Negative/backward thrust is
        # still allowed so the ROV can move away.

    if hold_heading:
        error = -math.atan2(math.sin(desired_yaw - latest_yaw), math.cos(desired_yaw - latest_yaw))
        if dt > 0:
            integral += error * dt
            derivative = (error - last_error) / dt
        else:
            derivative = 0.0
        last_error = error
        control_vector[3] = Kp * error + Ki * integral + Kd * derivative
        if vision_active:
            control_vector[3] = _clamp_abs(control_vector[3], VISION_MAX_YAW_CMD)

    if hold_depth:
        error_z = desired_depth - latest_depth
        if dt > 0:
            integral_z += error_z * dt
            derivative_z = (error_z - last_error_z) / dt
        else:
            derivative_z = 0.0
        last_error_z = error_z
        control_vector[2] = -(Kpz * error_z + Kiz * integral_z + Kdz * derivative_z)

    thrusts = mixing_matrix @ control_vector

    #debug
    global last_vision_debug_log

    if vision_active and (now - last_vision_debug_log) > 0.3:
        last_vision_debug_log = now
        print(
            f"[VISION] dx={snapshot['vision_move_x_norm']:+.3f}, "
            f"scale={snapshot['vision_ez']:+.3f}, "
            f"u_fwd={control_vector[0]:+.3f}, "
            f"u_lat={control_vector[1]:+.3f}, "
            f"conf={snapshot['vision_confidence']:.2f}"
        )

    # max_abs = float(np.max(np.abs(thrusts))) if np.size(thrusts) else 0.0
    # if max_abs > 1.0:
    #     thrusts = thrusts / max_abs
    send_thrust_pwm(thrusts)

    mode_name = 'VISION' if vision_active else (
        'HOLD' if (hold_heading or hold_depth) else 'MANUAL'
    )

    # print(
    #     f"Yaw: {math.degrees(latest_yaw):.1f}°/{math.degrees(desired_yaw):.1f}°, "
    #     f"Depth: {latest_depth:.2f}/{desired_depth:.2f} m | "
    #     f"Vision: valid={vision_fresh}, ex={snapshot['vision_ex']:.2f}, ey={snapshot['vision_ey']:.2f}, ez={snapshot['vision_ez']:.2f}, dz={snapshot['vision_dz_rel']:.3f}, raw={snapshot['vision_raw_dz_rel']:.3f}, range={snapshot['vision_range_cmd']:.2f}/{snapshot['vision_range_axis']}, conf={snapshot['vision_confidence']:.2f} | "
    #     f"Mode: {mode_name} | Vector: {control_vector.round(2)}"
    # )

#===================== END ROV HOLD VISION =====================#

# ===================== AJOUT : ARMEMENT VISION SANS GCS =====================
# Problème identifié : hold_vision (activé par process_command(3)) n'était
# jusqu'ici déclenché QUE par un bouton MANUAL_CONTROL reçu via MAVLink
# depuis une station sol (GCS). Sans GCS connectée pendant les tests du
# pipeline vision seul, hold_vision restait à False en permanence, et
# control_thruster() ignorait donc systématiquement le PID vision, même si
# des paquets dx/dy/dz valides arrivaient sur le port UDP 17002.
# Ce nouveau listener UDP permet d'armer/désarmer le mode vision (et de
# forcer un retour en manuel) sans dépendre de MAVLink, en envoyant un
# simple message texte sur VISION_ARM_PORT :
#   echo -n "ARM_VISION"    | nc -u -w1 127.0.0.1 12346
#   echo -n "DISARM_VISION" | nc -u -w1 127.0.0.1 12346
#   echo -n "MANUAL"        | nc -u -w1 127.0.0.1 12346
VISION_ARM_PORT = 12346  # AJOUT : port dédié, distinct de udp_config_listener (12345)

def vision_arm_listener():
    # AJOUT : socket dédiée, indépendante des autres listeners UDP du fichier.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", VISION_ARM_PORT))
    sock.settimeout(0.5)  # AJOUT : timeout pour pouvoir sortir proprement sur exit_flag
    print(f"🎯 Vision arm listener started on port {VISION_ARM_PORT}")
    while not exit_flag:
        try:
            data, sender = sock.recvfrom(64)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[VISION-ARM] Erreur de réception: {e}")
            continue

        msg = data.decode(errors="ignore").strip().upper()

        if msg == "ARM_VISION":
            # AJOUT : réutilise process_command(3), qui gère déjà hold_vision,
            # hold_heading et reset_pid_memory() de façon cohérente avec le
            # reste du contrôleur — mêmes effets que si un bouton GCS était pressé.
            if not hold_vision:
                process_command(3)
            print(f"[VISION-ARM] Vision mode armé (requête de {sender})")
        elif msg == "DISARM_VISION":
            # AJOUT : process_command(3) est un toggle ; le rappeler désarme si déjà armé.
            if hold_vision:
                process_command(3)
            print(f"[VISION-ARM] Vision mode désarmé (requête de {sender})")
        elif msg == "MANUAL":
            # AJOUT : retour forcé en mode manuel, utile comme filet de sécurité pendant les tests.
            process_command(2)
            print(f"[VISION-ARM] Retour en mode manuel (requête de {sender})")
        else:
            print(f"[VISION-ARM] Commande inconnue reçue: {msg!r} (de {sender})")
# ===================== FIN AJOUT ARMEMENT VISION =====================

def ping_server():
    global addr
    global masterout_hbt, masterout_sta, masterout_pos, masterout_imu, masterout_att

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0',15000))

    while True:
        data, client = sock.recvfrom(1024)
        new_addr = client[0]

        # CHỈ reconnect khi IP khác
        if new_addr != addr:
            print("New client:", client)

            addr = new_addr

            if masterout_hbt is not None:
                masterout_hbt.close()
                masterout_sta.close()
                masterout_pos.close()
                masterout_imu.close()
                masterout_att.close()

            masterout_hbt = mavutil.mavlink_connection(f'udpout:{addr}:16001')
            masterout_sta = mavutil.mavlink_connection(f'udpout:{addr}:16002')
            masterout_pos = mavutil.mavlink_connection(f'udpout:{addr}:16003')
            masterout_imu = mavutil.mavlink_connection(f'udpout:{addr}:16004')
            masterout_att = mavutil.mavlink_connection(f'udpout:{addr}:16005')

            print("Connected to client at:", addr)

def send_heartbeat():
    try:
        hbt_msg = mavlink2.MAVLink_heartbeat_message(
            type=mavlink2.MAV_TYPE_GENERIC,
            autopilot=mavlink2.MAV_AUTOPILOT_GENERIC,
            base_mode=0,
            custom_mode=0,
            system_status=mavlink2.MAV_STATE_ACTIVE,
            mavlink_version=3
        )
        masterout_hbt.mav.send(hbt_msg)
    except Exception as e:
        print("Error sending heartbeat:", e)

def send_named_value(name, value):
    try:
        msg = mavlink2.MAVLink_named_value_float_message(
            time_boot_ms=int((time.time() - start_time) * 1000),
            name=name.encode('utf-8'),
            value=float(value)
        )
    except Exception as e:
            print(f"Error sending named value '{name}':", e)
    masterout_sta.mav.send(msg)


def send_sys_status(): #rov health, pin voltage, current, temp, humid,... to GCS
    try:
        sys_status_msg = mavlink2.MAVLink_sys_status_message(
            onboard_control_sensors_present=1,
            onboard_control_sensors_enabled=1,
            onboard_control_sensors_health=1,
            load=500,
            voltage_battery=12000,  # 12.0V (in millivolts)
            current_battery=150,    # 150 mA (in 10 mA units)
            battery_remaining=90,   # 90%
            drop_rate_comm=0,
            errors_comm=0,
            errors_count1=int(humid),
            errors_count2=int(temp),
            errors_count3= int(current_mode),
            errors_count4=int(latest_depth)
        )
        if masterout_sta:
             masterout_sta.mav.send(sys_status_msg)
             send_named_value("Temp", temp)
             send_named_value("Humid", humid)
             send_named_value("Depth", latest_depth)
             send_named_value("CameraTilt", Camera_Tilt)
             send_named_value("LightLevel", LightPower)
             send_named_value("PowerLevel", PowerLevel)
             send_named_value("WaterTemp", WaterTemp)

             mode_heading= 1 if hold_heading else 0
             send_named_value("HoldHead", mode_heading)
             mode_depth= 4 if hold_depth else 0
             send_named_value("HoldDepth", mode_depth)
    except Exception as e:
        print("Error sending sys status:", e)

def send_global_position():
    try:
        global_position_msg = mavlink2.MAVLink_global_position_int_message(
            time_boot_ms=int((time.time() - start_time) * 1000),
            lat=int(21.0285 * 1e7),      # Replace with your lat
            lon=int(105.8542 * 1e7),     # Replace with your lon
            alt=int(5.0 * 1000),         # Altitude above sea level (mm)
            relative_alt=int(5.0 * 1000),# Altitude above ground (mm)
            vx=0,
            vy=0,
            vz=0,
            hdg=0
        )
        masterout_pos.mav.send(global_position_msg)
    except Exception as e:
        print("Error sending global position:", e)

def send_imu():  #
    try:
        imu_msg = mavlink2.MAVLink_raw_imu_message(
            time_usec=int((time.time() - start_time) * 1e6),
            xacc=int(xacc),  # Simulated accel in mg
            yacc=int(yacc),
            zacc=int(zacc),
            xgyro=int(xgyro),
            ygyro=int(ygyro),
            zgyro=int(zgyro),
            xmag=int(xmag),
            ymag=int(ymag),
            zmag=int(zmag)
        )
        masterout_imu.mav.send(imu_msg)
    except Exception as e:
        print("Error sending IMU data:", e)

def send_attitude():
    try:
        raw_i = batt.read_raw(ADS1115.MUX_AIN2)  # chưa hiệu chỉnh dòng
        raw_v = batt.read_raw(ADS1115.MUX_AIN3)  # điện áp pin (đã calibrate)

        v_pack = batt.raw_to_pack_voltage(raw_v)
        v_cell = v_pack / 4.0
        soc = batt.interp_percent(v_cell)

        # Send an attitude message to the drone
        attitude_msg = mavlink2.MAVLink_attitude_message(
            time_boot_ms=int((time.time() - start_time) * 1000),
            roll=latest_roll,
            pitch=latest_pitch,
            yaw=latest_yaw,
            rollspeed=raw_i,
            pitchspeed=soc,
            yawspeed=latest_depth
        )
        masterout_att.mav.send(attitude_msg)
        print("Attitude sended")
    except Exception as e:
        print("Error sending attitude data:", e)

start_time = time.time()

threading.Thread(target=depth_reader, daemon=True).start()
threading.Thread(target=heading_reader_ekf, daemon=True).start()
threading.Thread(target=temp_humid_reader, daemon=True).start()
threading.Thread(target=internal_vision_api_server, daemon=True).start()
threading.Thread(target=udp_vision_receiver, daemon=True).start()
#Receive new config via UDP and save to file
threading.Thread(target=udp_config_listener, daemon=True).start()

#init the thruster controller
init_thruster()

ping_thread = threading.Thread(target=ping_server)
ping_thread.start()

control_thread = threading.Thread(target=control_loop)
control_thread.start()

# AJOUT : démarre réellement la boucle de contrôle autonome. auto_control_loop()
# existait déjà dans le fichier mais n'était jamais lancée comme thread : en
# conséquence, control_thruster() n'était appelée que sur réception d'un
# paquet MAVLink MANUAL_CONTROL (voir control_loop), jamais en pilotage vision
# autonome sans GCS. Démarré ici (après init_thruster() et l'armement des ESC
# dans init_thruster(), pour éviter d'envoyer des commandes PWM avant que le
# contrôleur PCA9685 soit prêt).
threading.Thread(target=auto_control_loop, daemon=True).start()

# AJOUT : démarre le listener permettant d'armer/désarmer le mode vision (hold_vision)
# par un simple message UDP, sans dépendre d'un bouton MANUAL_CONTROL envoyé par une GCS.
# Voir la définition de vision_arm_listener() ci-dessus pour le protocole de commandes.
threading.Thread(target=vision_arm_listener, daemon=True).start()

# # Tune heading (yaw) – giữ tại góc hiện tại
# res_h = auto_tune_pid("heading",
#                        relay_amplitude=0.95,
#                        hysteresis=0.03,
#                        tune_time=20.0,
#                        clamp=0.95,
#                        rule="zn_pid",
#                        apply_result=True)

try:
    while True:
        if (masterout_hbt != None):
            send_heartbeat()
            send_sys_status()
            send_global_position()
            send_attitude()
            send_imu()
            

        time.sleep(1)  # Send messages every 1000 millisecond
except:
    exit_flag = True
    print("⛔ Kết thúc")
    send_thrust_pwm([0]*6)