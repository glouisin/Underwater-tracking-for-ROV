# -*- coding: utf-8 -*-
import serial
import time

class Si7021:
    def __init__(self, port='/dev/ttyHS3', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.connect()

    def connect(self):
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            time.sleep(2)  
            print(f"[UART] connected: {self.port}")
        except Exception as e:
            print(f"[UART] Error: {e}")
            self.ser = None

    def get_latest_data(self):
        if self.ser and self.ser.is_open and self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    parts = [float(x) for x in line.split(',')]
                    if len(parts) == 2:
                        return parts
            except (ValueError, UnicodeDecodeError):
                pass 
        return None

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"[UAR] disconnected: {self.port}")

if __name__ == "__main__":
    sensor = Si7021(port='/dev/ttyHS3', baudrate=9600)

    try:
        while True:
            data = sensor.get_latest_data()
            
            if data:
                temp, humi = data
                print(f"Temp : {temp} | Humi: {humi}")

            time.sleep(0.01) 
    except KeyboardInterrupt:
        sensor.close()