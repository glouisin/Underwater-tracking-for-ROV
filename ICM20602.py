import spidev
import time
import math

class ICM20602:
    def __init__(self, spi_bus=1, spi_dev=3):
        self.spi_bus = spi_bus
        self.spi_dev = spi_dev

        self.spi = spidev.SpiDev()
        self.spi.open(spi_bus, spi_dev)
        self.spi.max_speed_hz = 1000000
        self.spi.mode = 0b11  # Mode 3

        self.accel_scale = 16384.0  # Will be set properly later
        self.gyro_scale = 131.0     # Will be set properly later


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
        raw_data = self.spi.xfer2([0x3B | 0x80] + [0x00] * 12)[1:]
        if len(raw_data) != 12:
            print("Lỗi: Không đủ dữ liệu")
            return (0, 0, 0), (0, 0, 0)
        
        def to_signed(val):
            return val - 65536 if val > 32767 else val
        
        ax = to_signed((raw_data[0] << 8) | raw_data[1]) / self.accel_scale
        ay = to_signed((raw_data[2] << 8) | raw_data[3]) / self.accel_scale
        az = to_signed((raw_data[4] << 8) | raw_data[5]) / self.accel_scale
        
        gx = (to_signed((raw_data[6] << 8) | raw_data[7]) / self.gyro_scale) - getattr(self, 'gyro_bias', (0, 0, 0))[0]
        gy = (to_signed((raw_data[8] << 8) | raw_data[9]) / self.gyro_scale) - getattr(self, 'gyro_bias', (0, 0, 0))[1]
        gz = (to_signed((raw_data[10] << 8) | raw_data[11]) / self.gyro_scale) - getattr(self, 'gyro_bias', (0, 0, 0))[2]

        #print(f"ax: {ax}, ay: {ay}, az: {az}, gx: {gx}, gy: {gy}, gz: {gz}")

        return (ax, ay, az), (gx, gy, gz)

    def compute_pitch_roll(self, ax, ay, az):
        pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2)) * 180.0 / math.pi
        roll = math.atan2(ay, az) * 180.0 / math.pi
        return pitch, roll
    
    def calibrate_gyro(self, samples=100):
        print("Calibrating gyroscope, keep device still...")
        gx_sum = gy_sum = gz_sum = 0
        for _ in range(samples):
            _, (gx, gy, gz) = self.read_accel_gyro()
            gx_sum += gx
            gy_sum += gy
            gz_sum += gz
            time.sleep(0.01)
        self.gyro_bias = (gx_sum / samples, gy_sum / samples, gz_sum / samples)
        print(f"Gyro bias: {self.gyro_bias}")
        return self.gyro_bias
    
    def open(self):
        self.spi.open(self.spi_bus, self.spi_dev)
        self.spi.max_speed_hz = 1000000
        self.spi.mode = 0b00

    def close(self):
        self.spi.close()

if __name__ == "__main__":
    icm = ICM20602()
    icm.initialize()
    print("WHO_AM_I:", hex(icm.whoami()))
    accel, gyro = icm.read_accel_gyro()
    print("Accel [g]:", accel)
    print("Gyro  [dps]:", gyro)
    pitch, roll = icm.compute_pitch_roll(*accel)
    print(f"Pitch: {round(pitch, 2)}°, Roll: {round(roll, 2)}°")
    icm.close()