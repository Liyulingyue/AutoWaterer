import cv2
import openvino_genai as ov_genai
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import openvino as ov

class MiniCPMVL(object):
    def __init__(self):
        self.model_flag = False

    def set_model(self, model_path="Source/MiniCPM-V-2_6-ov", device="CPU"):
        self.ov_model = ov_genai.VLMPipeline(model_path, device=device)
        self.config = ov_genai.GenerationConfig()
        self.config.max_new_tokens = 100

        self.model_flag = True

    def release_model(self):
        self.ov_model = None
        self.config = None

        self.model_flag = False

    def infer(self, prompt, image_file_or_array):
        image, image_tensor = self.load_image(image_file_or_array)
        self.ov_model.start_chat()
        output = self.ov_model.generate(prompt, image=image_tensor, generation_config=self.config, streamer=self.streamer)
        self.ov_model.finish_chat()

        return output


    def load_image(self, image_file_or_array):
        if isinstance(image_file_or_array, str):
            if image_file_or_array.startswith("http") or image_file_or_array.startswith("https"):
                response = requests.get(image_file_or_array)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_file_or_array).convert("RGB")
        elif isinstance(image_file_or_array, np.ndarray):  # Add this branch for cv2 images
            # Assume the input array is in BGR format (common for cv2 images)
            image_bgr = image_file_or_array
            image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError(
                "Unsupported input type. Please provide a string (file path or URL) or an np.ndarray (cv2 image).")

        image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
        return image, ov.Tensor(image_data)

    def streamer(self, subword: str) -> bool:
        """

        Args:
            subword: sub-word of the generated text.

        Returns: Return flag corresponds whether generation should be stopped.

        """
        print(subword, end="", flush=True)

if __name__ == "__main__":
    image_path = "cat.png"

    if not Path(image_path).exists():
        url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
        image = Image.open(requests.get(url, stream=True).raw)
        image.save(image_path)

    minicpm = MiniCPMVL()

    minicpm.set_model(model_path="Source/MiniCPM-V-2_6-ov", device="CPU")
    output = minicpm.infer(prompt="what is that? ", image_file_or_array=image_path)
    print(f"\nAnswer:\n{output}")

    output = minicpm.infer(prompt="describe the image", image_file_or_array=image_path)
    print(f"\nAnswer:\n{output}")

    minicpm.release_model()
