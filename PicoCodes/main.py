from DRVClasses import Wheel
from Relay import Water
from UDPHelper import UDPHelper

wheel = Wheel(2,3,4,5,6,7,8,9)
water = Water(16)
udp = UDPHelper()

while 1:
    data, data_flag = udp.receive()
    if data_flag:
        data_list = data.split(" ")
        action_type = data_list[0]
        action_subtype = data_list[1]
        action_time = float(data_list[2])
        
        if action_type == "0":
            if action_subtype == "0": # 向前
                wheel.move_forward(action_time)
            elif action_subtype == "1": # 向后
                wheel.move_backward(action_time)
            elif action_subtype == "2": # 右上角
                wheel.rotate(1, action_time)
            elif action_subtype == "3": # 左上角
                wheel.rotate(2, action_time)
            elif action_subtype == "4": # 右下角
                wheel.rotate(3, action_time)
            elif action_subtype == "5": # 左下角
                wheel.rotate(4, action_time)
        elif action_type == "1":
            water.run(action_time)

        
        
