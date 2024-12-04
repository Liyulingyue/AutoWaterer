from machine import Pin
import utime

class Water:
    def __init__(self, sig):
        self.relay = Pin(16, Pin.OUT)
    
    def run(self, seconds):
        self.relay(1)
        utime.sleep(seconds)
        self.relay(0)
        
# relay = Pin(16, Pin.OUT)
