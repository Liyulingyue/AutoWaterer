import cv2
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import base64

class NzHelper(object):
    def __init__(self, nz_ip='***********', nz_port=5000):
        self.ip = nz_ip
        self.port = nz_port

    def get_image(self, img_type="cv2"):
        # img_type is either "cv2" or "PIL"
        route = f"http://{self.ip}:{self.port}/capture"

        # 清空request的缓存，以便于重新获取图像
        response = requests.get(route)

        img = None
        img_flag = False

        # 检查请求是否成功
        if response.status_code == 200:
            # 解析JSON响应
            data = response.json()
            # 获取Base64编码的图像字符串
            img_base64 = data['image']
            # 将Base64字符串解码为字节数据
            img_bytes = base64.b64decode(img_base64)
            # 将字节数据转换为PIL.Image对象
            img = Image.open(BytesIO(img_bytes))
            if img_type == "cv2":
                # 如果你需要numpy.ndarray，可以使用以下代码转换
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img_flag = True

        return img, img_flag