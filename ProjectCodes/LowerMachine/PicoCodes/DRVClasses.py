from machine import Pin, PWM
from drv8833 import DRV8833
import time

class Wheel:
    def __init__(self, ha1, ha2, hb1, hb2, ta1, ta2, tb1, tb2):
        # 以车头朝向为基础，车头右侧，车头左侧，车位右侧，车位左侧
        frequency = 40_000

        # Make sure to set the correct pins!
        ain1 = PWM(Pin(ha1, Pin.OUT))
        ain2 = PWM(Pin(ha2, Pin.OUT))
        bin1 = PWM(Pin(hb1, Pin.OUT))
        bin2 = PWM(Pin(hb2, Pin.OUT))
        ain1.freq(frequency)
        ain2.freq(frequency)
        bin1.freq(frequency)
        bin2.freq(frequency)
        
        self.drv_head = DRV8833(ain1, ain2, bin1, bin2)
        
        # Make sure to set the correct pins!
        ain1 = PWM(Pin(ta1, Pin.OUT))
        ain2 = PWM(Pin(ta2, Pin.OUT))
        bin1 = PWM(Pin(tb1, Pin.OUT))
        bin2 = PWM(Pin(tb2, Pin.OUT))
        ain1.freq(frequency)
        ain2.freq(frequency)
        bin1.freq(frequency)
        bin2.freq(frequency)
        
        self.drv_tail = DRV8833(ain1, ain2, bin1, bin2)
        
    def move_line(self, seconds, throttle=1):
        # 设置输出
        self.drv_head.throttle_a(throttle)
        self.drv_head.throttle_b(throttle)
        self.drv_tail.throttle_a(throttle)
        self.drv_tail.throttle_b(throttle)
        
        time.sleep(seconds)
        
        self.drv_head.stop_a()
        self.drv_head.stop_b()
        self.drv_tail.stop_a()
        self.drv_tail.stop_b()
    
    def move_forward(self, seconds, throttle=1):
        self.move_line(seconds, throttle=throttle)
    
    def move_backward(self, seconds, throttle=1):
        self.move_line(seconds, throttle=-throttle)
        
    def rotate(self, direction, seconds, throttle=1):
        # direction
        # 1: right up
        # 2: left up
        # 3: right bottom
        # 4: left bottom
        if direction == 1:
            self.drv_head.throttle_b(throttle)
            self.drv_tail.throttle_b(throttle)
        elif direction == 2:
            self.drv_head.throttle_a(throttle)
            self.drv_tail.throttle_a(throttle)
        elif direction == 3:
            self.drv_head.throttle_b(-throttle)
            self.drv_tail.throttle_b(-throttle)
        elif direction == 4:
            self.drv_head.throttle_a(-throttle)
            self.drv_tail.throttle_a(-throttle)
        else:
            self.drv_head.throttle_b(throttle)
            self.drv_tail.throttle_b(throttle)
           
        time.sleep(seconds)
        
        self.drv_head.stop_a()
        self.drv_head.stop_b()
        self.drv_tail.stop_a()
        self.drv_tail.stop_b()
        
        
        
class Water:
    def __init__(self, ha1, ha2, hb1, hb2):
        # ha1, ha2为有效位置，hb1,hb2为占位，无效
        frequency = 40_000

        # Make sure to set the correct pins!
        ain1 = PWM(Pin(ha1, Pin.OUT))
        ain2 = PWM(Pin(ha2, Pin.OUT))
        bin1 = PWM(Pin(hb1, Pin.OUT))
        bin2 = PWM(Pin(hb2, Pin.OUT))
        ain1.freq(frequency)
        ain2.freq(frequency)
        bin1.freq(frequency)
        bin2.freq(frequency)
        
        self.drv = DRV8833(ain1, ain2, bin1, bin2)
    
    def run(self, seconds, throttle=1):
        self.drv.throttle_a(throttle)
        time.sleep(seconds)
        self.drv.stop_a()
        
# Example
# from DRVClasses import Wheel, Water
# wheel = Wheel(2,3,4,5,6,7,8,9)
# wheel.move_forward(2)
# time.sleep(2)
# wheel.move_backward(2)

# water = Water(10, 11, 12, 13)
# water.run(1)
        
