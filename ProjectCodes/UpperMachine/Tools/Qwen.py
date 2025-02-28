from openai import OpenAI
import json

class QwenClass(object):
    def __init__(self, access_token="sk-**************"):
        self.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=access_token,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.chat_history = []

    def get_llm_answer(self, prompt):

        completion = self.client.chat.completions.create(
            model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                # {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ],
        )

        result = completion.choices[0].message.content

        return result

    def extract_json_from_llm_answer(self, result, start_str="```json", end_str="```", replace_list=["\n"]):
        s_id = result.index(start_str)
        e_id = result.index(end_str, s_id+len(start_str))
        json_str = result[s_id+len(start_str):e_id]
        for replace_str in replace_list:
            json_str = json_str.replace(replace_str,"")
        # print(json_str)
        try:
            json_dict = json.loads(json_str)
        except Exception as e:
            print("Error: ", e)
            print("json_str: ", json_str)
            json_dict = {}
        return json_dict

    def extract_markdown_from_llm_answer(self, result, start_str="```markdown", end_str="```", replace_list=["\n"]):
        s_id = result.index(start_str)
        e_id = result.index(end_str, s_id + len(start_str))
        markdown_str = result[s_id + len(start_str):e_id]
        return markdown_str

    def get_llm_json_answer(self, prompt):
        result = self.get_llm_answer(prompt)
        try:
            json_dict = self.extract_json_from_llm_answer(result)
        except Exception as e:
            print("Error: ", e)
            print("result: ", result)
            json_dict = {}
        return json_dict

    def get_llm_markdown_answer(self, prompt, raw_flag=False):
        result = self.get_llm_answer(prompt)
        if raw_flag == True:
            markdown_str = result
        else:
            try:
                markdown_str = self.extract_markdown_from_llm_answer(result)
            except Exception as e:
                print("Error: ", e)
                print("result: ", result)
                markdown_str = ""
        return markdown_str