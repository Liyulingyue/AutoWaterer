# 本文件用于对ESP32-CAM进行图像抓取操作，配置对应的IP和端口，即可进行测试。

import socket
import cv2
import numpy as np


def fetch_image_from_server(server_ip='192.168.2.113', port=80):
    # 创建一个TCP/IP套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # 连接到服务器
        sock.connect((server_ip, port))


        http_request = "GET"
        sock.sendall(http_request.encode())

        # 接收响应数据
        response = b''
        while True:
            data = sock.recv(4096)
            if not data:
                break
            response += data

        image_data = response

        # 将图像数据转换为OpenCV格式
        # 将字节数据转换为一维numpy数组
        img_array = np.frombuffer(image_data, dtype=np.uint8)

        # 使用OpenCV解码图像
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return img

    finally:
        # 关闭套接字
        sock.close()


if __name__ == "__main__":
    # 从ESP32-CAM获取图像
    img = fetch_image_from_server(server_ip='192.168.2.140')  # 替换为您的ESP32-CAM IP地址

    if img is not None:
        # 保存图像到本地文件
        cv2.imwrite('esp32cam.jpg', img)

        # 显示图像
        cv2.imshow('ESP32-CAM Image', img)
        cv2.waitKey(0)  # 按任意键关闭窗口
        cv2.destroyAllWindows()
    else:
        print("Failed to fetch image from server.")