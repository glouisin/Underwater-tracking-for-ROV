import spidev
import time
import math
from ahrs.filters import EKF
import numpy as np
from MMC5983 import MMC5983

class EKF_ICM:
    def __init__(self, spi_bus=0, spi_dev=3):
        self.spi_bus = spi_bus
        self.spi_dev = spi_dev

        self.spi = spidev.SpiDev()
        self.spi.open(spi_bus, spi_dev)
        self.spi.max_speed_hz = 1000000
        self.spi.mode = 0b11  # Mode 3

        self.mag_sensor = MMC5983(spi_bus=0, spi_dev=0)

        self.accel_scale = 16384.0  # Will be set properly later
        self.gyro_scale = 131.0     # Will be set properly later
              
        self.ekf_filter = EKF()

        self.q = np.array([1, 0, 0, 0])  # Initial quaternion (identity)

    def read_register(self, reg_addr):
        result = self.spi.xfer2([reg_addr | 0x80, 0x00])[1]
        return result

    def write_register(self, reg_addr, value):
        self.spi.xfer2([reg_addr & 0x7F, value])

    def whoami(self):
        return self.read_register(0x75)  # WHO_AM_I

    def set_accel_range(self, range_code):
        """
        Set accelerometer range.
        range_code: 0x00=±2g, 0x08=±4g, 0x10=±8g, 0x18=±16g
        """
        self.write_register(0x1C, range_code)
        time.sleep(0.01)
        actual = self.read_register(0x1C) & 0x18
        if actual == 0x00:
            self.accel_scale = 16384.0
        elif actual == 0x08:
            self.accel_scale = 8192.0
        elif actual == 0x10:
            self.accel_scale = 4096.0
        elif actual == 0x18:
            self.accel_scale = 2048.0
        else:
            self.accel_scale = 16384.0
        print("ACCEL_CONFIG:", hex(actual), ", scale:", self.accel_scale)

    def set_gyro_range(self, range_code):
        """
        Set gyroscope range.
        range_code: 0x00=±250dps, 0x08=±500dps, 0x10=±1000dps, 0x18=±2000dps
        """
        self.write_register(0x1B, range_code)
        time.sleep(0.01)
        actual = self.read_register(0x1B) & 0x18
        if actual == 0x00:
            self.gyro_scale = 131.0
        elif actual == 0x08:
            self.gyro_scale = 65.5
        elif actual == 0x10:
            self.gyro_scale = 32.8
        elif actual == 0x18:
            self.gyro_scale = 16.4
        else:
            self.gyro_scale = 131.0
        print("GYRO_CONFIG:", hex(actual), ", scale:", self.gyro_scale)

    def initialize(self):
        self.write_register(0x6B, 0x80)  # Reset device
        time.sleep(0.1)
        self.write_register(0x6B, 0x00)  # Wake up device
        time.sleep(0.01)
        self.write_register(0x6C, 0x00)  # Enable accel and gyro
        time.sleep(0.01)
        self.set_accel_range(0x00)  # Default to ±2g
        self.set_gyro_range(0x00)   # Default to ±250dps

    def read_word(self, reg):
        high = self.read_register(reg)
        low = self.read_register(reg + 1)
        val = (high << 8) | low
        return val - 65536 if val > 32767 else val

    def read_accel_gyro(self):
        raw_data = self.spi.xfer2([0x3B | 0x80] + [0x00] * 14)[1:15]
        if len(raw_data) != 14:
            print("❗ Lỗi: Không đủ dữ liệu")
            return (0, 0, 0), (0, 0, 0)
        
        def to_signed(val):
            return val - 65536 if val > 32767 else val
        
        ax = to_signed((raw_data[0] << 8) | raw_data[1]) / self.accel_scale * 9.81
        ay = to_signed((raw_data[2] << 8) | raw_data[3]) / self.accel_scale * 9.81
        az = to_signed((raw_data[4] << 8) | raw_data[5]) / self.accel_scale * 9.81
        
        gx_raw = to_signed((raw_data[8] << 8) | raw_data[9]) / self.gyro_scale
        gy_raw = to_signed((raw_data[10] << 8) | raw_data[11]) / self.gyro_scale
        gz_raw = to_signed((raw_data[12] << 8) | raw_data[13]) / self.gyro_scale

        gx = (gx_raw - self.gyro_bias[0]) * (math.pi/180)
        gy = (gy_raw - self.gyro_bias[1]) * (math.pi/180)
        gz = (gz_raw - self.gyro_bias[2]) * (math.pi/180)

        return (ax, ay, az), (gx, gy, gz)

    def read_mag(self):
        mag = self.mag_sensor.measure()
        if mag:
            mx, my, mz = [v * 1e5 for v in mag]  # chuyển sang nanoTesla
            mag_raw = np.array([mx, my, mz])

            # Áp dụng hiệu chỉnh nếu có
            if hasattr(self, 'mag_offset') and hasattr(self, 'mag_scale'):
                mag_corrected = (mag_raw - self.mag_offset) / self.mag_scale
            else:
                mag_corrected = mag_raw

            # # Chuẩn hóa vector từ trường
            # mag_norm = mag_corrected / np.linalg.norm(mag_corrected)
            # return mag_norm
            return mag_raw
        else:
            return np.array([0.0, 0.0, 0.0])

    
    def quaternion_to_euler_ENU(self, q):
        w, x, y, z = q
        roll = math.atan2(2*(w*x+y*z), 1 - 2*(x*x + y*y))
        pitch = math.asin(2*(w*y - z*x))
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return roll, pitch, yaw
    
    def quat_ENU_to_NED(self,q):
        w, x, y, z = q
        return np.array([w, y, x, -z])

    def quaternion_to_euler_NED(self, q):
       w, x, y, z = self.quat_ENU_to_NED(q)
    # Roll (x-axis)
       roll = math.atan2(2*(w*x + y*z), w*w - x*x - y*y + z*z)
    # Pitch (y-axis)
       pitch = -math.asin(2*(x*z - w*y))
    # Yaw (z-axis)
       yaw = math.atan2(2*(x*y + w*z), w*w + x*x - y*y - z*z)

    # Swap lại để khớp với thứ tự muốn in
       return pitch, roll, yaw

    def calibrate_gyro(self, samples=500, delay=0.01):
       print("Calibrating gyroscope... keep IMU still.")
       gx_sum = gy_sum = gz_sum = 0.0

       for _ in range(samples):
            raw_data = self.spi.xfer2([0x43 | 0x80] + [0x00] * 6)[1:7]

            def to_signed(v):
                return v - 65536 if v > 32767 else v

            raw_gx = to_signed((raw_data[0] << 8) | raw_data[1]) / self.gyro_scale
            raw_gy = to_signed((raw_data[2] << 8) | raw_data[3]) / self.gyro_scale
            raw_gz = to_signed((raw_data[4] << 8) | raw_data[5]) / self.gyro_scale

            gx_sum += raw_gx
            gy_sum += raw_gy
            gz_sum += raw_gz

            time.sleep(delay)

       self.gyro_bias = (
           gx_sum/samples,
           gy_sum/samples,
           gz_sum/samples)
    
       return self.gyro_bias

    def compute_pitch_roll(self, ax, ay, az):
        pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))
        roll = math.atan2(ay, az)
        return pitch, roll
        
    def calibrate_mag(self, samples=500, delay=0.01):
        data = []

        for _ in range(samples):
            mag = self.read_mag()
            data.append(mag)
            time.sleep(delay)

        data = np.array(data)
        mag_min = data.min(axis=0)
        mag_max = data.max(axis=0)
        self.mag_offset = (mag_max + mag_min) / 2
        self.mag_scale = (mag_max - mag_min) / 2

    
    def open(self):
        self.spi.open(self.spi_bus, self.spi_dev)
        self.spi.max_speed_hz = 1000000
        self.spi.mode = 0b11

    def close(self):
        self.spi.close()
