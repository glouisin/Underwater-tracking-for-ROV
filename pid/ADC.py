#!/usr/bin/env python3
import time
import smbus2 as smbus

class ADS1115:
    ADDR = 0x48
    REG_CONV   = 0x00
    REG_CONFIG = 0x01

    OS_SINGLE  = 0x8000
    # single-ended mux
    MUX_AIN2   = 0x6000
    MUX_AIN3   = 0x7000

    PGA_4V096  = 0x0200
    MODE_SS   = 0x0100
    DR_128SPS = 0x0080
    COMP_OFF  = 0x0003

    def __init__(self, bus=4, address=ADDR):  # <-- bus 4 theo bạn
        self.addr = address
        self.bus = smbus.SMBus(bus)

    def _write16(self, reg, val):
        self.bus.write_i2c_block_data(self.addr, reg, [(val >> 8) & 0xFF, val & 0xFF])

    def _read16(self, reg):
        d = self.bus.read_i2c_block_data(self.addr, reg, 2)
        return (d[0] << 8) | d[1]

    @staticmethod
    def _signed(v):
        return v - 0x10000 if (v & 0x8000) else v

    def read_raw(self, mux, timeout_s=0.2):
        cfg = (self.OS_SINGLE | mux | self.PGA_4V096 | self.MODE_SS | self.DR_128SPS | self.COMP_OFF)
        self._write16(self.REG_CONFIG, cfg)

        t0 = time.time()
        while True:
            if self._read16(self.REG_CONFIG) & self.OS_SINGLE:
                break
            if time.time() - t0 > timeout_s:
                raise TimeoutError("ADS1115 timeout")
            time.sleep(0.001)

        return self._signed(self._read16(self.REG_CONV))

    def interp_percent(self, v_cell: float) -> float:
        # Xấp xỉ SOC theo điện áp nghỉ/nhẹ tải (LiPo)
        table = [
            (4.20, 100), (4.15, 95), (4.11, 90), (4.08, 85), (4.02, 80),
            (3.98, 75), (3.95, 70), (3.91, 65), (3.87, 60), (3.85, 55),
            (3.84, 50), (3.82, 45), (3.80, 40), (3.79, 35), (3.77, 30),
            (3.75, 25), (3.73, 20), (3.71, 15), (3.69, 10), (3.61, 5),
            (3.50, 0),
        ]

        if v_cell >= table[0][0]:
            return 100.0
        if v_cell <= table[-1][0]:
            return 0.0

        for (v_hi, p_hi), (v_lo, p_lo) in zip(table[:-1], table[1:]):
            if v_lo <= v_cell <= v_hi:
                t = (v_cell - v_lo) / (v_hi - v_lo)
                return p_lo + t * (p_hi - p_lo)

        return 0.0

    # ====== CALIBRATION mới (2 điểm bạn cung cấp) ======
    A_V = 0.00201448
    B_V = -0.31670

    def raw_to_pack_voltage(self, raw_ain3: int) -> float:
        return self.A_V * raw_ain3 + self.B_V

if __name__ == "__main__":
    adc = ADS1115(bus=4)

    while True:
        raw_i = adc.read_raw(adc.MUX_AIN2)  # chưa hiệu chỉnh dòng
        raw_v = adc.read_raw(adc.MUX_AIN3)  # điện áp pin (đã calibrate)

        v_pack = adc.raw_to_pack_voltage(raw_v)
        v_cell = v_pack / 4.0
        soc = adc.interp_percent(v_cell)

        print(f"AIN2={raw_i:6d} | AIN3={raw_v:6d} | Vbat={v_pack:5.2f} V | Cell={v_cell:4.2f} V | SOC~{soc:5.1f}%")
        time.sleep(0.2)