import spidev
import time
import math

REG_XOUT_L = 0x00
REG_TOUT = 0x07
REG_STATUS = 0x08
REG_CONTROL0 = 0x09
REG_CONTROL1 = 0x0A
REG_CONTROL2 = 0x0B
REG_CONTROL3 = 0x0C
REG_PRODUCT_ID = 0x2F

# bits in REG_CONTROL0
REG_CONTROL0_TMM = (1<<0)
REG_CONTROL0_TMT = (1<<1)
REG_CONTROL0_SET = (1<<3)
REG_CONTROL0_RESET = (1<<4)
REG_CONTROL0_AUTOSREN = (1<<5)

# bits in REG_CONTROL1
REG_CONTROL1_BW0 = (1<<0)
REG_CONTROL1_BW1 = (1<<1)
REG_CONTROL1_SW_RST = (1<<7)
REG_CONTROL1_CONFIG = 0b11 # BW = 0b11 - 0.5ms per measurement

REG_CONTROL2_CMFREQ = 0b111
REG_CONTROL2_CMEN = (1<<4)
REG_CONTROL2_ENPRDSET = (1<<7)
REG_CONTROL2_CONFIG = 0b1111 # enable continuous measurement at 1kHz

MIN_DELAY_SET_RESET = 1e-3
MIN_DELAY_MEASURE = 1.1e-3

MMC5983_ID = 0x0C

class CompassData:
    def __init__(self, rawdata, caldata):
        self.x_raw = int.from_bytes(rawdata[0:2], 'big')
        self.y_raw = int.from_bytes(rawdata[2:4], 'big')
        self.z_raw = int.from_bytes(rawdata[4:6], 'big')

        xyz2 = rawdata[6]
        self.x_raw = (self.x_raw << 2) | (((xyz2 & 0xC0) >> 6) & 0x3)
        self.y_raw = (self.y_raw << 2) | (((xyz2 & 0x30) >> 4) & 0x3)
        self.z_raw = (self.z_raw << 2) | (((xyz2 & 0x03) >> 2) & 0x3)

        self.x_raw -= 0x20000
        self.y_raw -= 0x20000
        self.z_raw -= 0x20000

        # field strength in gauss
        self.x_norm = self.x_raw/16384
        self.y_norm = self.y_raw/16384
        self.z_norm = self.z_raw/16384

        self.x = self.x_norm - caldata[0]
        self.y = self.y_norm - caldata[1]
        self.z = self.z_norm - caldata[2]

        self.t_raw = rawdata[7]
        self.t = self.t_raw*200.0/256 - 75

class MMC5983:
    def __init__(self, spi_bus=1, spi_dev=0):
        self._bus = spidev.SpiDev()
        self._bus.open(spi_bus, spi_dev)
        self._bus.max_speed_hz = 10000000 # 10MHz

        self.caldata = [0, 0, 0]
        self.software_reset()
        self._id = self.read_id()
        self.config1()
        self.config2()
        self.calibrate()

    def config1(self):
        self.write(REG_CONTROL1, [REG_CONTROL1_CONFIG])

    def config2(self):
        self.write(REG_CONTROL2, [REG_CONTROL2_CONFIG])

    def software_reset(self):
        self.write(REG_CONTROL1, [REG_CONTROL1_SW_RST])
        time.sleep(0.015)

    def reset(self):
        self.write(REG_CONTROL0, [REG_CONTROL0_RESET])
        time.sleep(MIN_DELAY_SET_RESET)

    def set(self):
        self.write(REG_CONTROL0, [REG_CONTROL0_SET])
        time.sleep(MIN_DELAY_SET_RESET)

    def calibrate(self):
        self.caldata = [0, 0, 0]

        self.set()
        time.sleep(MIN_DELAY_MEASURE)
        setdata = self.read_data()

        self.reset()
        time.sleep(MIN_DELAY_MEASURE)
        resetdata = self.read_data()

        self.caldata = [
            (setdata[0] + resetdata[0])/2,
            (setdata[1] + resetdata[1])/2,
            (setdata[2] + resetdata[2])/2
        ]

    def set_BW(self, BW=(REG_CONTROL1_BW0 | REG_CONTROL1_BW1)):
        self.write(REG_CONTROL1, [BW])

    def read_id(self):
        id = self.readByte(REG_PRODUCT_ID)
        return id

    def measure(self):
        self.write(REG_CONTROL0, [REG_CONTROL0_TMM])
        time.sleep(MIN_DELAY_MEASURE)
        status = self.readByte(REG_STATUS)
        while not ((status & 1) == 1):
            status = self.readByte(REG_STATUS)
            continue

        return self.read_data()

    def read_data(self):
        rawdata = self.read(REG_XOUT_L, 8)
        compass = CompassData(rawdata, self.caldata)
        return [compass.x, compass.y, compass.z]

    def read(self, reg, nbytes=1):
        xferdata = [0] * (nbytes+1)
        xferdata[0] = reg | 0x80 # read transaction
        return self._bus.xfer(xferdata)[1:]

    def readByte(self, reg):
        return self.read(reg)[0]

    def write(self, reg, data):
        data.insert(0, reg)
        return self._bus.xfer(data)

    def close(self):
        self._bus.close()

    def calculate_heading(self, mag, pitch=0.0, roll=0.0):
        x, y, z = mag

        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)

        xh = x * math.cos(pitch_rad) + z * math.sin(pitch_rad)
        yh = x * math.sin(roll_rad) * math.sin(pitch_rad) + y * math.cos(roll_rad) - z * math.sin(roll_rad) * math.cos(pitch_rad)

        heading_rad = math.atan2(yh, xh)
        heading_deg = math.degrees(heading_rad)
        if heading_deg < 0:
            heading_deg += 360.0
        return heading_deg
    
if __name__ == "__main__":
    # Khởi tạo cảm biến (trên Pi là bus 1, dev 0; trên QC là bus 0, dev 0)
    mmc = MMC5983(spi_bus=0, spi_dev=0) 

    # Đọc dữ liệu liên tục
    try:
        lastcal = time.time()
        while True:
            if time.time() > lastcal + 1:
                mmc.calibrate()
                lastcal = time.time() 

            mag = mmc.measure()
            if mag:
                print(f"Magnetometer (Gauss): {mag[0]:.2f}, {mag[1]:.2f}, {mag[2]:.2f}")
                heading = mmc.calculate_heading(mag, pitch=0.0, roll=0.0)  # Thêm pitch/roll nếu có IMU
                print("Heading [deg]:", round(heading, 2))
            else:
                print("⚠️ Không có dữ liệu")

            time.sleep(0.1)  # đọc 10 Hz
    except KeyboardInterrupt:
        print("⛔ Kết thúc.")
    finally:
        mmc.close()