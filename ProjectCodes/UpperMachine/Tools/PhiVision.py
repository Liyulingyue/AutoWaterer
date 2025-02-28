from .ov_phi3_vision_helper import OvPhi3Vision
import requests
from PIL import Image
from transformers import AutoProcessor, TextStreamer

class PhiVision(object):
    def __init__(self, model_dir = "Source/phi-3.5-vision-instruct-ov", device = "GPU"):
        self.model = OvPhi3Vision(model_dir, device) # "GPU" or "CPU"
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    def infer_with_single_img(self, image, text):
        messages = [
            {
                "role": "user",
                "content": f"<|image_1|>\n{text}"
            },
        ]
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt, [image], return_tensors="pt")
        generation_args = {"max_new_tokens": 1024, # 3072
                           "do_sample": False,
                           "streamer": TextStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)}
        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args)
        words = self.processor.tokenizer.decode(generate_ids[0])
        results = words.split("<|assistant|>")[-1].split("<|end|>")[0]

        return results

    def infer_with_single_path(self, img_path, text):
        image = Image.open(img_path)
        results = self.infer_with_single_img(image, text)
        return results

