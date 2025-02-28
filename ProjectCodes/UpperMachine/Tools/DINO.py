import copy
import time

import cv2
import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class DINO_with_camera(object):
    def __init__(self, model_id="Source/GroundingDINO", device="cpu", camera=None):
        self.device = device
        self.camera = camera

        # 创建模型
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


        self.frame = None
        self.results = None



    def infer(self, prompt="Everything", box_threshold=0.1, text_threshold=0.1):
        ret, frame = self.camera.get_frame()
        # 将捕获的帧从BGR转换为RGB，然后转换为PIL图像
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # 处理图像和文本输入
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 后处理结果
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )
        # 打印结果
        print(results) # [{'scores': tensor([0.1286]), 'labels': ['plant'], 'boxes': tensor([[ 9.0370e-02, -3.9783e-01,  6.4009e+02,  4.7960e+02]])}]

        self.results = copy.deepcopy(results)
        self.frame = copy.deepcopy(frame)

    def get_results(self):
        return self.results

    def get_frame(self):
        return self.frame

    def get_draw_img(self):
        img = copy.deepcopy(self.frame)

        # 在图片上绘制边框
        for result in self.results:
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                # if "plant" not in label.lower():
                #     continue
                xmin, ymin, xmax, ymax = box.tolist()
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

        return img

    def save_draw_img(self, save_path="Source/Images/Detection.jpg"):
        img = self.get_draw_img()
        cv2.imwrite(save_path, img)


