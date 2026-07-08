from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink2

import time
import random
import socket
import threading
import serial
import numpy as np

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

latest_pitch = 0.0
latest_yaw = 0.0
latest_roll = 0.0
latest_depth = 0.0

current_mode = 2    #Đồng bộ với ROV Controller GUI, có thể là "MANUAL", "HOLD", hoặc "TUNE" để hiển thị trạng thái hiện tại trên GUI

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

increment = 0.1

tune_heading = False
tune_depth = False
tune_log = []
tune_start_time = 0
tune_output_axis = 3

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
THRUSTER_MAP = [9, 10, 4, 6, 7, 8, 5, 11]
ESC_GAIN = 0.5
PowerLevel = (ESC_GAIN*100)
ESC_MAX_POWER = 400
ESC_NEUTRAL = 1500
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
                print(f"Temp: {temp} , Humid: {humid}")
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
            exit_flag = True
            return
        sensor.setFluidDensity(ms5837.DENSITY_SALTWATER)  # seawater
        print("✅ Depth sensor initialized (MS5837-30BA)")
    except Exception as e:
        print("❌ Depth sensor error:", e)
        exit_flag = True
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

def control_loop():
    global prev_x, prev_y, prev_z, prev_r, Light_Status, Camera_Tilt, ESC_GAIN, LightPower, PowerLevel

    #IP v4 only
    masterin = mavutil.mavlink_connection('udpin:0.0.0.0:16001')

    #IP v6 and v4 (dual-stack)
    #masterin = mavutil.mavlink_connection('udpin:[::]:16001')

    while True:
        msg = masterin.recv_match(blocking=True)
        if msg is not None and msg.get_type() != 'BAD_DATA':
            if msg.get_type() == 'HEARTBEAT':
                print("Received heartbeat from drone")
            elif msg.get_type() == 'MANUAL_CONTROL':
                control_thruster(msg.y/1000.0, msg.x/1000.0, msg.z/1000.0, msg.r/1000.0)
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

def process_command(button_code):
    global Thruster_Arm, hold_heading, hold_depth, exit_flag, latest_yaw, latest_depth, desired_yaw, desired_depth, integral, last_error, integral_z, last_error_z, last_time, current_mode
    if (button_code == 4):      #X - depth hold
        if not hold_depth:
            Thruster_Arm = 1
            desired_depth = latest_depth
            hold_depth = True
            current_mode = 4
        elif hold_depth:
            Thruster_Arm = 1
            hold_depth = False
            current_mode = 2    
        print(f"🔒 Hold depth at {desired_depth:.2f} m")
    elif (button_code == 2):    #B - manual
        Thruster_Arm = 1
        hold_heading = False
        hold_depth = False
        current_mode = 2
        print(f"🔓 Manual mode")
    elif (button_code == 1):    #Y - stabilize or heading hold
        if not hold_heading:
            Thruster_Arm = 1
            desired_yaw = latest_yaw
            hold_heading = True
            current_mode = 1
        elif hold_heading:
            Thruster_Arm = 1
            hold_heading = False
            current_mode = 2
        print(f"🔒 Hold heading at {math.degrees(desired_yaw):.1f}°")
#    elif (button_code == 0):    # DISARM (controller moi)
#        Thruster_Arm = 0
#        hold_heading = False
#        hold_depth = False

def init_thruster():
    thruster_controller.set_pwm_frequency(50)
    thruster_controller.output_enable()

    #arm all ESCs
    send_thrust_pwm([0]*8)

def control_thruster(forward, lateral, ascend, yaw):
    global Thruster_Arm, hold_heading, hold_depth, exit_flag, latest_yaw, latest_depth, desired_yaw, desired_depth, integral, last_error, integral_z, last_error_z, last_time
    if Thruster_Arm == 0:
        print("--- DISARM")
        send_thrust_pwm([0]*8)
    else:
        now = time.time()
        dt = now - last_time
        last_time = now
        manual_vector[0] = forward
        manual_vector[1] = lateral
        manual_vector[2] = ascend
        manual_vector[3] = yaw

        control_vector = manual_vector.copy()
        if hold_heading:
            error = -math.atan2(math.sin(desired_yaw - latest_yaw), math.cos(desired_yaw - latest_yaw))
            if dt > 0:
                integral += error * dt
                derivative = (error - last_error) / dt
            else:
                derivative = 0.0
            last_error = error
            control_vector[3] = Kp * error + Ki * integral + Kd * derivative

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
        send_thrust_pwm(thrusts)

        print(f"Yaw: {math.degrees(latest_yaw):.1f}°/{math.degrees(desired_yaw):.1f}°, Depth: {latest_depth:.2f}/{desired_depth:.2f} m | Mode: {'TUNE' if tune_heading or tune_depth else ('HOLD' if hold_heading or hold_depth else 'MANUAL')} | Vector: {control_vector.round(2)}")

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
            xacc=xacc,  # Simulated accel in mg
            yacc=yacc,
            zacc=zacc,
            xgyro=xgyro,
            ygyro=ygyro,
            zgyro=zgyro,
            xmag=xmag,
            ymag=ymag,
            zmag=zmag
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

#init the thruster controller
init_thruster()

ping_thread = threading.Thread(target=ping_server)
ping_thread.start()

control_thread = threading.Thread(target=control_loop)
control_thread.start()

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
