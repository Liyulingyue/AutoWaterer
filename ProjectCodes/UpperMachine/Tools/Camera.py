import time

import cv2


class Camera(object):
    def __init__(self, camera_id=0, scale=1):
        # 初始化摄像头
        self.cap = cv2.VideoCapture(camera_id)  # 0是默认摄像头的索引
        self.cap.set(3, 3840)  # width=3840
        self.cap.set(4, 2160)  # height=2160
        self.scale = scale

        # 激活摄像头
        _, _ = self.cap.read()
        time.sleep(3)

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret == True and self.scale < 1:
            # 裁剪frame, 裁剪周围的边框
            ori_width = frame.shape[1]
            ori_height = frame.shape[0]

            width_gap = int((1-self.scale)/2 * ori_width)
            height_gap = int((1-self.scale)/2 * ori_height)

            frame = frame[height_gap:ori_height-height_gap, width_gap:ori_width-width_gap]
            # 将frame转化为原始尺寸
            frame = cv2.resize(frame, (ori_width, ori_height))
        return ret, frame

    def __del__(self):
        # 释放摄像头并关闭所有窗口
        self.cap.release()