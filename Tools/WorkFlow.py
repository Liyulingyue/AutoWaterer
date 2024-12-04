import cv2
import pickle

import sklearn
from PIL import Image
import numpy as np

from Tools.DINO import DINO_with_camera
from Tools.NzHelper import NzHelper
from Tools.PhiVision import PhiVision
from Tools.udp_helper import create_socket, udp_send
from Tools.Camera import Camera
from Tools.CarDetection import CarDetection
# from Tools.Qwen import QwenClass
from Tools.ernie import ErnieClass
import time

from Tools.utils import find_closest_point_on_line, calculate_iou

camera = Camera(camera_id=0, scale=1) # 相机, 尺度默认为1
car_detection = CarDetection(camera=camera)
dino = DINO_with_camera(camera=camera)
phivision = PhiVision(device="GPU")
nzhelper = NzHelper()
server_socket = create_socket() # 控制器
# qwen = QwenClass()
llm = ErnieClass(access_token="*********")

def init_plants():
    # 初始化，检索植物所在位置，记录植物的基础信息
    print("Init Plants, we first find the plants and record it...")

    dino.infer(prompt="Plant")
    dino.save_draw_img()

    dino_img = dino.get_frame()
    dino_results = dino.get_results()
    dino_draw_img = dino.get_draw_img()

    plants_records = {
        "InitImage": dino_img,
        "InitResults": dino_results,
        "InitDrawImg": dino_draw_img,
        "CarInfo": {},
        "Plants": [],
    }
    plants_list = []
    # 遍历 dino_results 中的每个结果，删除bbox占据了整个屏幕90%以上的结果，删除bbox中心点在左1/3区域的结果
    for result in dino_results:
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            xmin, ymin, xmax, ymax = box.tolist()
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)

            # 默认可能会检测到整个图像区域
            if (xmax - xmin) / dino_img.shape[1] > 0.9:
                pass
            elif (xmin + xmax) / 2 < dino_img.shape[1] * 1 / 3:
                pass
            else:
                cropped_image = dino_img[ymin:ymax, xmin:xmax]
                plants_list.append({
                    "score": score,
                    "label": label,
                    "box": [xmin, ymin, xmax, ymax],
                    "cropped_image": cropped_image
                })
    plants_records["Plants"] = plants_list

    # 保存植物列表到文件中
    with open("Source/Logs/1.pkl", "wb") as f:
        pickle.dump(plants_records, f)

    return plants_records

def init_plants2():
    # 初始化，检索植物所在位置，记录植物的基础信息
    print("Init Plants, we first find the plants and record it...")

    dino.infer(prompt="Plant")
    dino.save_draw_img()

    dino_img = dino.get_frame()
    dino_results = dino.get_results()
    dino_draw_img = dino.get_draw_img()

    plants_records = {
        "InitImage": dino_img,
        "InitResults": dino_results,
        "InitDrawImg": dino_draw_img,
        "CarInfo": {},
        "Plants": [],
    }
    plants_list = []
    # 遍历 dino_results 中的每个结果，删除bbox占据了整个屏幕90%以上的结果，删除bbox中心点在左1/3区域的结果
    for result in dino_results:
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            xmin, ymin, xmax, ymax = box.tolist()
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)

            iou_flag = False
            for item in plants_list:
                old_bbox = item["box"]
                new_bbox = [xmin, ymin, xmax, ymax]
                iou = calculate_iou(old_bbox, new_bbox)
                print(iou)
                if iou > 0.5:
                    iou_flag = True
                    break

            if iou_flag:
                continue

            # 默认可能会检测到整个图像区域
            if (xmax - xmin) / dino_img.shape[1] > 0.9:
                pass
            elif (xmin + xmax) / 2 < dino_img.shape[1] * 1 / 3:
                pass
            else:
                cropped_image = dino_img[ymin:ymax, xmin:xmax]
                plants_list.append({
                    "score": score,
                    "label": label,
                    "box": [xmin, ymin, xmax, ymax],
                    "cropped_image": cropped_image
                })
    plants_records["Plants"] = plants_list

    # 保存植物列表到文件中
    with open("Source/Logs/1.pkl", "wb") as f:
        pickle.dump(plants_records, f)

    return plants_records


def car_search():
    # 辅助函数，计算车辆的box
    print("In car search, we find the car position...")

    fix_car_direction(dir="y")

    car_detection.infer()
    car_detection.save_draw_img()

    car_img = car_detection.get_frame()
    car_results = car_detection.get_results()

    car_flag = -1
    car_minx = 0
    car_miny = 0
    car_maxx = 0
    car_maxy = 0
    for item in car_results:
        cls, value, xmin, ymin, xmax, ymax = item
        if value > 0.5:
            cls, xmin, ymin, xmax, ymax = [int(x) for x in [cls, xmin, ymin, xmax, ymax]]
            if cls == 0 and car_flag != 0:
                car_minx = xmin
                car_miny = ymin
                car_maxx = xmax
                car_maxy = ymax
                car_flag = cls
            elif cls != 0 and car_flag == -1:
                car_minx = xmin
                car_miny = ymin
                car_maxx = xmax
                car_maxy = ymax
                car_flag = cls
            else:
                pass

    car_dict = {
        "box": [car_minx, car_miny, car_maxx, car_maxy],
        "cropped_image": car_img[car_miny:car_maxy, car_minx:car_maxx],
        "center": [int((car_minx + car_maxx) / 2), int((car_miny + car_maxy) / 2)],
        "width": abs(car_maxx - car_minx),
        "height": abs(car_maxy - car_miny),
    }

    return car_dict

def car_search2():
    # 辅助函数，计算车辆的box
    print("In car search, we find the car position...")

    fix_car_direction(dir="x")

    car_detection.infer()
    car_detection.save_draw_img()

    car_img = car_detection.get_frame()
    car_results = car_detection.get_results()

    car_flag = -1
    car_minx = 0
    car_miny = 0
    car_maxx = 0
    car_maxy = 0
    for item in car_results:
        cls, value, xmin, ymin, xmax, ymax = item
        if value > 0.5:
            cls, xmin, ymin, xmax, ymax = [int(x) for x in [cls, xmin, ymin, xmax, ymax]]
            if cls == 0 and car_flag != 0:
                car_minx = xmin
                car_miny = ymin
                car_maxx = xmax
                car_maxy = ymax
                car_flag = cls
            elif cls != 0 and car_flag == -1:
                car_minx = xmin
                car_miny = ymin
                car_maxx = xmax
                car_maxy = ymax
                car_flag = cls
            else:
                pass

    car_dict = {
        "box": [car_minx, car_miny, car_maxx, car_maxy],
        "cropped_image": car_img[car_miny:car_maxy, car_minx:car_maxx],
        "center": [int((car_minx + car_maxx) / 2), int((car_miny + car_maxy) / 2)],
        "width": abs(car_maxy - car_miny),
        "height": abs(car_maxx - car_minx),
    }

    return car_dict


def init_car(plants_records):
    print("Init Car, we find out the movement ability of car...")

    car_detection.infer()
    init_car_img = car_detection.get_draw_img()

    action_type = 0  # 转轮
    action_sub_forward = 0  # 向前
    action_sub_backward = 1  # 向后
    action_time = 1.5  # 持续时间，单位：秒

    car_dict1 = car_search()

    # message = f"{action_type} {action_sub} {action_time:.3f}"
    message = f"{action_type} {action_sub_forward} {action_time:.3f}"
    udp_send(server_socket, message)
    time.sleep(action_time * 3)

    car_dict2 = car_search()

    message = f"{action_type} {action_sub_backward} {action_time:.3f}"
    udp_send(server_socket, message)
    time.sleep(action_time * 3)

    x1 = car_dict1["center"][0]
    y1 = car_dict1["center"][1]
    x2 = car_dict2["center"][0]
    y2 = car_dict2["center"][1]
    # dx = x2 - x1
    # dy = y2 - y1
    # angle = np.arctan2(dy, dx) * 180 / np.pi
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    dis_per_sec = distance / action_time

    plants_records["CarInfo"]["init_car_img"] = init_car_img
    plants_records["CarInfo"]["movement_check_dict1"] = car_dict1
    plants_records["CarInfo"]["movement_check_dict2"] = car_dict2
    plants_records["CarInfo"]["dis_per_sec"] = dis_per_sec
    plants_records["CarInfo"]["width"] = (car_dict1["width"] + car_dict2["width"]) / 2
    plants_records["CarInfo"]["height"] = (car_dict1["height"] + car_dict2["height"]) / 2

    print(f"The distance per second is {dis_per_sec}")

    # 保存植物列表到文件中
    with open("Source/Logs/2.pkl", "wb") as f:
        pickle.dump(plants_records, f)

    return plants_records

def init_car2 (plants_records):
    print("Init Car, we find out the movement ability of car...")

    car_detection.infer()
    init_car_img = car_detection.get_draw_img()

    action_type = 0  # 转轮
    action_sub_forward = 0  # 向前
    action_sub_backward = 1  # 向后
    action_time = 1  # 持续时间，单位：秒

    car_dict1 = car_search2()

    # message = f"{action_type} {action_sub} {action_time:.3f}"
    message = f"{action_type} {action_sub_forward} {action_time:.3f}"
    udp_send(server_socket, message)
    time.sleep(action_time * 3)

    car_dict2 = car_search2()

    message = f"{action_type} {action_sub_backward} {action_time:.3f}"
    udp_send(server_socket, message)
    time.sleep(action_time * 3)

    x1 = car_dict1["center"][0]
    y1 = car_dict1["center"][1]
    x2 = car_dict2["center"][0]
    y2 = car_dict2["center"][1]
    # dx = x2 - x1
    # dy = y2 - y1
    # angle = np.arctan2(dy, dx) * 180 / np.pi
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    dis_per_sec = distance / action_time

    plants_records["CarInfo"]["init_car_img"] = init_car_img
    plants_records["CarInfo"]["movement_check_dict1"] = car_dict1
    plants_records["CarInfo"]["movement_check_dict2"] = car_dict2
    plants_records["CarInfo"]["dis_per_sec"] = dis_per_sec
    plants_records["CarInfo"]["width"] = (car_dict1["width"] + car_dict2["width"]) / 2
    plants_records["CarInfo"]["height"] = (car_dict1["height"] + car_dict2["height"]) / 2

    print(f"The distance per second is {dis_per_sec}")

    # 保存植物列表到文件中
    with open("Source/Logs/2.pkl", "wb") as f:
        pickle.dump(plants_records, f)

    return plants_records

def preprocessing(plants_records):
    print("Preprocessing, we do some preprocessing to make the movement more convenience...")

    ret, frame = camera.get_frame()

    # 读取车辆宽度
    car_width = plants_records["CarInfo"]["width"]
    car_height = plants_records["CarInfo"]["height"]
    for i in range(len(plants_records["Plants"])):
        xmin, ymin, xmax, ymax = plants_records["Plants"][i]["box"]
        center_x = int((xmin + xmax) / 2)
        center_y = int((ymin + ymax) / 2)

        target_x = int(center_x - car_width / 6 * 1)
        target_y = int(center_y + car_width / 4 * 2)
        rotate_start_y = target_y - car_height // 4  # 这么计算好像有点问题

        plants_records["Plants"][i]["target_point"] = [target_x, target_y]
        plants_records["Plants"][i]["rotate_start_y"] = rotate_start_y

        # 在frame上绘制目标点
        cv2.circle(frame, (target_x, target_y), 6, (0, 0, 255), -1)

    # 保存植物列表到文件中
    with open("Source/Logs/3.pkl", "wb") as f:
        pickle.dump(plants_records, f)

    # cv2.imwrite("3.jpg", frame)

    return plants_records

def preprocessing2(plants_records):
    print("Preprocessing, we do some preprocessing to make the movement more convenience...")

    ret, frame = camera.get_frame()

    # 读取车辆宽度
    car_width = plants_records["CarInfo"]["width"]
    car_height = plants_records["CarInfo"]["height"]
    for i in range(len(plants_records["Plants"])):
        xmin, ymin, xmax, ymax = plants_records["Plants"][i]["box"]
        center_x = int((xmin + xmax) / 2)
        center_y = int((ymin + ymax) / 2)

        target_x = int(center_x - car_width / 8 * 0)
        target_y = int(center_y + car_width / 4 * 2)
        rotate_start_y = target_y - car_height // 4  # 这么计算好像有点问题

        plants_records["Plants"][i]["target_point"] = [target_x, target_y]
        plants_records["Plants"][i]["rotate_start_y"] = rotate_start_y

        # 在frame上绘制目标点
        cv2.circle(frame, (target_x, target_y), 6, (0, 0, 255), -1)

    # 保存植物列表到文件中
    with open("Source/Logs/3.pkl", "wb") as f:
        pickle.dump(plants_records, f)

    # cv2.imwrite("3.jpg", frame)

    return plants_records

def preprocessing3(plants_records):
    print("Preprocessing, we do some preprocessing to make the movement more convenience...")

    ret, frame = camera.get_frame()

    # 读取车辆宽度
    car_width = plants_records["CarInfo"]["width"]
    car_height = plants_records["CarInfo"]["height"]
    for i in range(len(plants_records["Plants"])):
        xmin, ymin, xmax, ymax = plants_records["Plants"][i]["box"]
        center_x = int((xmin + xmax) / 2)
        center_y = int((ymin + ymax) / 2)

        target_x = center_x
        target_y = center_y
        rotate_start_y = target_y - car_height // 4  # 这么计算好像有点问题

        plants_records["Plants"][i]["target_point"] = [target_x, target_y]
        plants_records["Plants"][i]["rotate_start_y"] = rotate_start_y

        # 在frame上绘制目标点
        cv2.circle(frame, (target_x, target_y), 6, (0, 0, 255), -1)

    # 保存植物列表到文件中
    with open("Source/Logs/3.pkl", "wb") as f:
        pickle.dump(plants_records, f)

    # cv2.imwrite("3.jpg", frame)

    return plants_records

def get_car_center():
    # 辅助函数，计算车辆的box

    car_detection.infer()
    car_detection.save_draw_img()

    car_img = car_detection.get_frame()
    car_results = car_detection.get_results()

    car_flag = -1
    car_minx = 0
    car_miny = 0
    car_maxx = 0
    car_maxy = 0
    for item in car_results:
        cls, value, xmin, ymin, xmax, ymax = item
        if value > 0.5:
            cls, xmin, ymin, xmax, ymax = [int(x) for x in [cls, xmin, ymin, xmax, ymax]]
            if cls == 0 and car_flag != 0:
                car_minx = xmin
                car_miny = ymin
                car_maxx = xmax
                car_maxy = ymax
                car_flag = cls
            elif cls != 0 and car_flag == -1:
                car_minx = xmin
                car_miny = ymin
                car_maxx = xmax
                car_maxy = ymax
                car_flag = cls
            else:
                pass

    return int((car_minx + car_maxx) / 2), int((car_miny + car_maxy) / 2)


def fix_car_direction(dir="x"):
    print("Fix Direction X, we fix the direction of the car on the x axis...")
    action_type = 0  # 转轮
    action_sub_forward = 0  # 向前
    action_sub_backward = 1  # 向后
    action_time = 0.5  # 持续时间，单位：秒

    current_x1, current_y1 = get_car_center()
    # message = f"{action_type} {action_sub} {action_time:.3f}"
    message = f"{action_type} {action_sub_forward} {action_time:.3f}"
    udp_send(server_socket, message)
    time.sleep(action_time * 2)

    current_x2, current_y2 = get_car_center()
    message = f"{action_type} {action_sub_backward} {action_time:.3f}"
    udp_send(server_socket, message)
    time.sleep(action_time * 2)

    dx = current_x2 - current_x1
    dy = current_y2 - current_y1
    print("d_value is ", dx, dy)
    if dir == "x":
        check_value = dy
    else:
        check_value = dx
    # 如果dy 大于10，转动0.1s
    if dir == "x" and check_value > 30:
        action_type = 0
        action_sub = 1 + 2
        action_time = 0.1
        message = f"{action_type} {action_sub} {action_time:.3f}"
        udp_send(server_socket, message)
        time.sleep(action_time * 2)
        fix_car_direction(dir=dir)
    elif dir == "x" and check_value < -30:
        action_type = 0
        action_sub = 1 + 1
        action_time = 0.1
        message = f"{action_type} {action_sub} {action_time:.3f}"
        udp_send(server_socket, message)
        time.sleep(action_time * 2)
        fix_car_direction(dir=dir)
    elif dir == "y" and check_value > 30:
        action_type = 0
        action_sub = 1 + 1
        action_time = 0.1
        message = f"{action_type} {action_sub} {action_time:.3f}"
        udp_send(server_socket, message)
        time.sleep(action_time * 2)
        fix_car_direction(dir=dir)
    elif dir == "y" and check_value < -30:
        action_type = 0
        action_sub = 1 + 2
        action_time = 0.1
        message = f"{action_type} {action_sub} {action_time:.3f}"
        udp_send(server_socket, message)
        time.sleep(action_time * 2)
        fix_car_direction(dir=dir)


def fix_car_pos(target_value, dir="x"):
    print("Fix Direction X, we fix the direction of the car on the x axis...")
    action_type = 0  # 转轮
    action_sub_forward = 0  # 向前
    action_sub_backward = 1  # 向后
    action_time = 0.1  # 持续时间，单位：秒

    current_x, current_y = get_car_center()

    if dir == "x":
        check_value = current_x - target_value
    else:
        check_value = current_y - target_value
    # 如果dy 大于10，转动0.1s
    if abs(check_value) > 20:
        if check_value > 0:
            message = f"{action_type} {action_sub_backward} {action_time:.3f}"
            udp_send(server_socket, message)
            time.sleep(action_time * 2)
            fix_car_pos(target_value=target_value, dir=dir)
        else:
            message = f"{action_type} {action_sub_forward} {action_time:.3f}"
            udp_send(server_socket, message)
            time.sleep(action_time * 2)
            fix_car_pos(target_value=target_value, dir=dir)

def fit_car_pos_closely(target_x, target_y):
    # 初始化

    x1, y1 = get_car_center()
    dis1 = ((target_x - x1) ** 2 + (target_y - y1) ** 2) ** 0.5

    # 向前
    action_type2 = 0
    action_sub2 = 0
    action_time2 = 0.1
    message = f"{action_type2} {action_sub2} {action_time2:.3f}"
    udp_send(server_socket, message)
    time.sleep(action_time2 * 1.5)

    x2, y2 = get_car_center()
    dis2 = ((target_x - x2) ** 2 + (target_y - y2) ** 2) ** 0.5

    # 如果dis1 < dis2，向前是靠近的
    if dis1 < dis2:
        action_sub2 = 0
    else:
        action_sub2 = 1
    action_type2 = 0
    action_time2 = 0.1

    prev_dis = dis2
    while 1:
        # 步进式操作
        message = f"{action_type2} {action_sub2} {action_time2:.3f}"
        udp_send(server_socket, message)
        time.sleep(action_time2 * 1.5)

        current_x, current_y = get_car_center()
        now_dis = ((target_x - current_x) ** 2 + (target_y - current_y) ** 2) ** 0.5

        if now_dis > prev_dis:
            break
        prev_dis = now_dis


def move_to_y(target_y, dis_per_sec):
    current_x, current_y = get_car_center()

    # 校准y方向
    fix_car_direction(dir="y")

    # 移动到转向位置
    dy = target_y - current_y
    if dy > 0:
        action_type1 = 0
        action_sub1 = 0
        action_time1 = abs(dy) / dis_per_sec
        message = f"{action_type1} {action_sub1} {action_time1:.3f}"
        udp_send(server_socket, message)
    else:
        action_type1 = 0
        action_sub1 = 1
        action_time1 = abs(dy) / dis_per_sec
        message = f"{action_type1} {action_sub1} {action_time1:.3f}"
        udp_send(server_socket, message)
    time.sleep(action_time1 * 1.5)

    fix_car_pos(target_value=target_y, dir="y")


def move_to_x(target_x, dis_per_sec):
    current_x, current_y = get_car_center()

    # 校准y方向
    fix_car_direction(dir="x")

    # 移动到转向位置
    dx = target_x - current_x
    if dx > 0:
        action_type1 = 0
        action_sub1 = 0
        action_time1 = abs(dx) / dis_per_sec
        message = f"{action_type1} {action_sub1} {action_time1:.3f}"
        udp_send(server_socket, message)
    else:
        action_type1 = 0
        action_sub1 = 1
        action_time1 = abs(dx) / dis_per_sec
        message = f"{action_type1} {action_sub1} {action_time1:.3f}"
        udp_send(server_socket, message)
    time.sleep(action_time1 * 1.5)

    fix_car_pos(target_value=target_x, dir="x")

def move_line_closely(target_x, target_y, dis_per_sec):
    x1, y1 = get_car_center()
    dis1 = ((target_x - x1) ** 2 + (target_y - y1) ** 2) ** 0.5

    # 向前
    action_type2 = 0
    action_sub2 = 0
    action_time2 = 0.1
    message = f"{action_type2} {action_sub2} {action_time2:.3f}"
    udp_send(server_socket, message)
    time.sleep(action_time2 * 1.5)

    x2, y2 = get_car_center()
    dis2 = ((target_x - x2) ** 2 + (target_y - y2) ** 2) ** 0.5

    # 拟合移动路线方程，计算目标点到拟合路线的最近一个点的坐标
    closest_x, closest_y = find_closest_point_on_line(x1, y1, x2, y2, target_x, target_y)

    # 如果dis1 < dis2，向前是靠近的
    if dis1 < dis2:
        action_sub2 = 0
    else:
        action_sub2 = 1
    action_type2 = 0
    dis = ((x2 - closest_x) ** 2 + (y2 - closest_y) ** 2) ** 0.5
    action_time2 = dis / dis_per_sec
    message = f"{action_type2} {action_sub2} {action_time2:.3f}"
    udp_send(server_socket, message)
    time.sleep(action_time2 * 1.5)

    fit_car_pos_closely(target_x, target_y)

def move_to(target_x, target_y, dis_per_sec):
    move_to_y(target_y, dis_per_sec)

    # 转向
    action_type2 = 0
    action_sub2 = 1 + 2
    action_time2 = 1.2 # 转向时间
    message = f"{action_type2} {action_sub2} {action_time2:.3f}"
    udp_send(server_socket, message)
    time.sleep(action_time2 * 1.5)

    move_to_x(target_x, dis_per_sec)

def move_to_simple(target_x, target_y, dis_per_sec):

    move_to_x(target_x, dis_per_sec)

def move_to_simple2(target_x, target_y, dis_per_sec):
    move_line_closely(target_x, target_y, dis_per_sec)


def move_back(target_x, target_y, dis_per_sec):
    move_to_x(target_x, dis_per_sec)

    # 转向
    action_type2 = 0
    action_sub2 = 1 + 4
    action_time2 = 1.2 # 转向时间
    message = f"{action_type2} {action_sub2} {action_time2:.3f}"
    udp_send(server_socket, message)
    time.sleep(action_time2 * 1.5)

    move_to_y(target_y, dis_per_sec)

def move_back_simple(target_x, target_y, dis_per_sec):
    move_to_x(target_x, dis_per_sec)


def move_action1(plants_records):
    print("Move Action, we start moving the car...")
    car_x, car_y = get_car_center()

    for i in range(len(plants_records["Plants"])):
        target_x, target_y = plants_records["Plants"][i]["target_point"]
        dis_per_sec = plants_records["CarInfo"]["dis_per_sec"]

        move_to(target_x, target_y, dis_per_sec)
        img, img_flag = nzhelper.get_image()
        plants_records["Plants"][i]["car_image"] = img
        plants_records["Plants"][i]["new_image"] = img

        cv2.imwrite(f"{i}.jpg", img)

        move_back(car_x, car_y, dis_per_sec)

    # 保存植物列表到文件中
    with open("Source/Logs/4.pkl", "wb") as f:
        pickle.dump(plants_records, f)

    return plants_records

def move_action1_simple(plants_records):
    print("Move Action, we start moving the car...")
    car_x, car_y = get_car_center()

    for i in range(len(plants_records["Plants"])):
        target_x, target_y = plants_records["Plants"][i]["target_point"]
        dis_per_sec = plants_records["CarInfo"]["dis_per_sec"]

        move_to_simple(target_x, target_y, dis_per_sec)
        img, img_flag = nzhelper.get_image()
        plants_records["Plants"][i]["car_image"] = img
        plants_records["Plants"][i]["new_image"] = img

        cv2.imwrite(f"{i}.jpg", img)

        move_back_simple(car_x, car_y, dis_per_sec)

    # 保存植物列表到文件中
    with open("Source/Logs/4.pkl", "wb") as f:
        pickle.dump(plants_records, f)

    return plants_records

def move_action1_simple2(plants_records):
    print("Move Action, we start moving the car...")
    car_x, car_y = get_car_center()

    for i in range(len(plants_records["Plants"])):
        target_x, target_y = plants_records["Plants"][i]["target_point"]
        dis_per_sec = plants_records["CarInfo"]["dis_per_sec"]

        move_to_simple2(target_x, target_y, dis_per_sec)
        img, img_flag = nzhelper.get_image()
        plants_records["Plants"][i]["car_image"] = img
        plants_records["Plants"][i]["new_image"] = img

        cv2.imwrite(f"{i}.jpg", img)

        move_to_simple2(car_x, car_y, dis_per_sec)

    # 保存植物列表到文件中
    with open("Source/Logs/4.pkl", "wb") as f:
        pickle.dump(plants_records, f)

    return plants_records

def get_water_results(plants_records):
    for i in range(len(plants_records["Plants"])):
        img = plants_records["Plants"][i]["car_image"]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB
        pil_image = Image.fromarray(img_rgb)
        prompt = "描述植物的生长状态，，判断土壤是否潮湿，植物是否有蔫的情况，你的描述将用于指导植物的浇水，请尽可能简单回答。"
        phi_results = phivision.infer_with_single_img(pil_image, prompt)
        plants_records["Plants"][i]["phi_results"] = phi_results

        prompt = f"""
请给出对植物的浇水分量，通常，小型盆摘的浇水量为2秒，大型盆摘的浇水量为10s。

我通过视觉模型获得的对图片的描述为：{phi_results}

请通过json格式输出，输出的格式为：
{{
    "浇水时间": int,
}}
"""
        result_dict = llm.get_llm_json_answer(prompt)
        plants_records["Plants"][i]["water_time"] = result_dict["浇水时间"]

    # 保存植物列表到文件中
    with open("Source/Logs/5.pkl", "wb") as f:
        pickle.dump(plants_records, f)

    return plants_records


def move_action2(plants_records):
    print("Move Action, we start moving the car...")
    car_x, car_y = get_car_center()

    for i in range(len(plants_records["Plants"])):
        target_x, target_y = plants_records["Plants"][i]["target_point"]
        dis_per_sec = plants_records["CarInfo"]["dis_per_sec"]

        move_to(target_x, target_y, dis_per_sec)
        img, img_flag = nzhelper.get_image()
        try:
            plants_records["Plants"][i]["old_image"] = plants_records["Plants"][i]["new_image"]
        except:
            plants_records["Plants"][i]["old_image"] = plants_records["Plants"][i]["car_image"]
        plants_records["Plants"][i]["new_image"] = img

        cv2.imwrite(f"{i}.jpg", img)
        water_time = plants_records["Plants"][i]["water_time"]
        action_type = 1
        action_sub = 0
        action_time = water_time
        message = f"{action_type} {action_sub} {action_time:.3f}"
        udp_send(server_socket, message)
        time.sleep(action_time * 2)

        move_back(car_x, car_y, dis_per_sec)

    # 保存植物列表到文件中
    with open("Source/Logs/6.pkl", "wb") as f:
        pickle.dump(plants_records, f)

    return plants_records

def move_action2_simple(plants_records):
    print("Move Action, we start moving the car...")
    car_x, car_y = get_car_center()

    for i in range(len(plants_records["Plants"])):
        target_x, target_y = plants_records["Plants"][i]["target_point"]
        dis_per_sec = plants_records["CarInfo"]["dis_per_sec"]

        move_to_simple(target_x, target_y, dis_per_sec)
        img, img_flag = nzhelper.get_image()
        try:
            plants_records["Plants"][i]["old_image"] = plants_records["Plants"][i]["new_image"]
        except:
            plants_records["Plants"][i]["old_image"] = plants_records["Plants"][i]["car_image"]
        plants_records["Plants"][i]["new_image"] = img

        cv2.imwrite(f"{i}.jpg", img)
        water_time = plants_records["Plants"][i]["water_time"]
        action_type = 1
        action_sub = 0
        action_time = water_time
        message = f"{action_type} {action_sub} {action_time:.3f}"
        udp_send(server_socket, message)
        time.sleep(action_time * 2)

        move_back_simple(car_x, car_y, dis_per_sec)

    # 保存植物列表到文件中
    with open("Source/Logs/6.pkl", "wb") as f:
        pickle.dump(plants_records, f)

    return plants_records

def move_action2_simple2(plants_records):
    print("Move Action, we start moving the car...")
    car_x, car_y = get_car_center()

    for i in range(len(plants_records["Plants"])):
        target_x, target_y = plants_records["Plants"][i]["target_point"]
        dis_per_sec = plants_records["CarInfo"]["dis_per_sec"]

        move_to_simple2(target_x, target_y, dis_per_sec)
        img, img_flag = nzhelper.get_image()
        try:
            plants_records["Plants"][i]["old_image"] = plants_records["Plants"][i]["new_image"]
        except:
            plants_records["Plants"][i]["old_image"] = plants_records["Plants"][i]["car_image"]
        plants_records["Plants"][i]["new_image"] = img

        cv2.imwrite(f"{i}.jpg", img)
        water_time = plants_records["Plants"][i]["water_time"]
        action_type = 1
        action_sub = 0
        action_time = water_time
        message = f"{action_type} {action_sub} {action_time:.3f}"
        udp_send(server_socket, message)
        time.sleep(action_time * 2)

        move_to_simple2(car_x, car_y, dis_per_sec)

    # 保存植物列表到文件中
    with open("Source/Logs/6.pkl", "wb") as f:
        pickle.dump(plants_records, f)

    return plants_records

def get_feedback(plants_records):
    for i in range(len(plants_records["Plants"])):
        old_img = plants_records["Plants"][i]["old_image"]
        new_img = plants_records["Plants"][i]["new_image"]
        old_img = Image.fromarray(old_img)
        new_img = Image.fromarray(new_img)

        com_img = Image.new('RGB', (old_img.width + new_img.width, old_img.height))
        com_img.paste(old_img, (0, 0))  # 粘贴image到左边
        com_img.paste(new_img, (new_img.width, 0))  # 粘贴image2到右边
        prompt = "左边的图是前一天拍摄的植物照片，右边的图是今天拍摄的植物照片，请你比较前一天和今天拍摄的照片中植物的生长状态并给出养护建议，尽可能简单回答。"
        phi_results = phivision.infer_with_single_img(com_img, prompt)

        com_img_np = np.array(com_img)
        com_img_bgr = cv2.cvtColor(com_img_np, cv2.COLOR_RGB2BGR)
        plants_records["Plants"][i]["feedback_img"] = com_img_bgr
        plants_records["Plants"][i]["feedback_str"] = phi_results

        com_img.save(f"com{i}.jpg")

    # 保存植物列表到文件中
    with open("Source/Logs/7.pkl", "wb") as f:
        pickle.dump(plants_records, f)

    return plants_records


