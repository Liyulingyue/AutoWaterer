import copy
import os

import cv2
import numpy as np
import openvino as ov
import yaml


class CarDetection(object):
    def __init__(self, model_path="Source/CarDetection", device="CPU", camera=None):
        self.camera = camera

        # Initialize OpenVINO Runtime for detection.
        self.core = ov.Core()

        pd_model_path = os.path.join(model_path, "model.pdmodel")
        self.det_model = self.core.read_model(model=pd_model_path)
        self.det_compiled_model = self.core.compile_model(model=self.det_model, device_name=device)

        # Get input and output nodes for text detection.
        self.det_input_layer = self.det_compiled_model.input(0)
        self.det_output_layer = self.det_compiled_model.output(0)

        yaml_path = os.path.join(model_path, "infer_cfg.yml")
        with open(yaml_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.label_list = self.config["label_list"]
        yaml_size = self.config["Preprocess"][0]["target_size"]
        self.input_size = (yaml_size[1], yaml_size[0])

        self.frame = None
        self.results = None


    def _infer(self, input_image):
        input_size = self.input_size
        scale_factor = [input_size[1] / input_image.shape[0], input_size[0] / input_image.shape[1]]
        factor = np.array(scale_factor, dtype=np.float32).reshape((1, 2))

        img = cv2.resize(input_image, input_size)
        img = np.transpose(img, [2, 0, 1]) / 255
        img = np.expand_dims(img, 0)

        test_image = img

        det_results = self.det_compiled_model([factor, test_image, ])[self.det_output_layer]

        return det_results

    def infer(self):
        ret, frame = self.camera.get_frame()
        self.frame = copy.deepcopy(frame)

        results = self._infer(frame)

        self.results = copy.deepcopy(results)

    def get_results(self):
        return self.results

    def get_frame(self):
        return self.frame

    def get_draw_img(self):
        ori_img = copy.deepcopy(self.frame)
        det_results = self.results
        for item in det_results:
            cls, value, xmin, ymin, xmax, ymax = item
            if value > 0.5:
                cls, xmin, ymin, xmax, ymax = [int(x) for x in [cls, xmin, ymin, xmax, ymax]]
                cv2.rectangle(ori_img, (xmin, ymin), (xmax, ymax), (0, 0, 0), 5)
                cv2.putText(ori_img, f"class: {self.label_list[cls]} conf: {value:.3f}", (xmin, ymin),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                # print(item)

        return ori_img

    def save_draw_img(self, save_path="Source/Images/Detection.jpg"):
        img = self.get_draw_img()
        cv2.imwrite(save_path, img)
