import gradio as gr
from Tools.WorkFlow import *

def fn_refresh_camera():
    ret, frame = camera.get_frame()
    return frame

def fn_refresh_car_camera():
    img, img_flag = nzhelper.get_image()
    return img

def fn_refresh_dino():
    img = None
    return img

def fn_refresh_yoloe():
    img = None
    return img

def fn_init_plants(plants_records):
    plants_records = init_plants() # save 1
    img = plants_records['InitDrawImg']
    plants_number = len(plants_records["Plants"])
    return img, f'{plants_number}个植物', plants_records

def fn_init_car(plants_records):
    with open("Source/Logs/1.pkl", "rb") as f:
        plants_records = pickle.load(f)
    plants_records = init_car(plants_records)
    img = plants_records["CarInfo"]["init_car_img"]
    return img, f'车辆初始化完成，车辆长度{plants_records["CarInfo"]["height"]}pixels，车辆宽度{plants_records["CarInfo"]["width"]}pixels', plants_records

def fn_init_preprocess(plants_records):
    with open("Source/Logs/2.pkl", "rb") as f:
        plants_records = pickle.load(f)
    plants_records = preprocessing(plants_records) # save 3
    img = plants_records['InitDrawImg']
    return img, f'预处理已完成', plants_records

def fn_move1(plants_records):
    with open("Source/Logs/3.pkl", "rb") as f:
        plants_records = pickle.load(f)
    plants_records = move_action1(plants_records)
    img = plants_records["Plants"][0]["car_image"]
    return img, f'移动并采集到植物图像', plants_records

def fn_calculate(plants_records):
    with open("Source/Logs/4.pkl", "rb") as f:
        plants_records = pickle.load(f)
    plants_records = get_water_results(plants_records)
    img = plants_records["Plants"][0]["car_image"]
    return img, f'第一个植物的浇水时常为{plants_records["Plants"][0]["water_time"]}', plants_records

def fn_move2(plants_records):
    with open("Source/Logs/5.pkl", "rb") as f:
        plants_records = pickle.load(f)
    plants_records = move_action2(plants_records)
    return None, f'移动并浇水已完成', plants_records

def fn_get_feedback(plants_records):
    with open("Source/Logs/6.pkl", "rb") as f:
        plants_records = pickle.load(f)
    plants_records = get_feedback(plants_records)
    img = plants_records["Plants"][0]["feedback_img"]
    return img, f'反馈信息已计算完成，植物1的反馈内容为{plants_records["Plants"][0]["feedback_str"]}', plants_records

def fn_all():
    plants_records = init_plants()  # save 1
    plants_records = init_car(plants_records)  # save 2
    plants_records = preprocessing(plants_records)  # save 3
    plants_records = move_action1(plants_records)  # save 4
    plants_records = get_water_results(plants_records)  # save 5
    plants_records = move_action2(plants_records)  # save 6
    plants_records = get_feedback(plants_records)  # save 7

def fn_all_without_init():
    with open("Source/Logs/5.pkl", "rb") as f:
        plants_records = pickle.load(f)
    plants_records = move_action2(plants_records)  # save 6
    plants_records = get_feedback(plants_records)  # save 7

with gr.Blocks() as demo:
    gr.Markdown("""
    # Plants Watering System Demo
    这是一个用于自动浇水的小车运行DEMO，你可以点击其中每个按钮触发小车的行动，观察每一步的结果。
    """)

    state = gr.State()
    with gr.Row():
        with gr.Column():
            img_head = gr.Image(interactive=False)
            text_notion = gr.Textbox()
        with gr.Column():
            btn_refresh_camera = gr.Button("刷新摄像头")
            btn_refresh_car_camera = gr.Button("刷新车辆摄像头")
            btn_init_plants = gr.Button("Step1: 初始化植物状态")
            btn_init_car = gr.Button("Step2: 初始化车辆状态")
            btn_init_preprocess = gr.Button("Step3: 初始化预处理")
            btn_move1 = gr.Button("Step4: 移动并采集图像")
            btn_calculate = gr.Button("Step5: 计算浇水量")
            btn_move2 = gr.Button("Step6: 移动并浇水")
            btn_get_feedback = gr.Button("Step8: 计算反馈信息")
            btn_all = gr.Button("一键运行")
            btn_all_without_init = gr.Button("一键运行(无初始化)")
            btn_refresh_dino = gr.Button("刷新DINO")
            btn_refresh_yoloe = gr.Button("刷新YOLOE")


    btn_refresh_camera.click(fn=fn_refresh_camera, inputs=[], outputs=[img_head])
    btn_refresh_car_camera.click(fn=fn_refresh_car_camera, inputs=[], outputs=[img_head])

    btn_refresh_dino.click(fn=fn_refresh_dino, inputs=[], outputs=[img_head])
    btn_refresh_yoloe.click(fn=fn_refresh_yoloe, inputs=[], outputs=[img_head])

    btn_init_plants.click(fn=fn_init_plants, inputs=[state], outputs=[img_head, text_notion, state])
    btn_init_car.click(fn=fn_init_car, inputs=[state], outputs=[img_head, text_notion, state])
    btn_init_preprocess.click(fn=fn_init_preprocess, inputs=[state], outputs=[img_head, text_notion, state])
    btn_move1.click(fn=fn_move1, inputs=[state], outputs=[img_head, text_notion, state])
    btn_calculate.click(fn=fn_calculate, inputs=[state], outputs=[img_head, text_notion, state])
    btn_move2.click(fn=fn_move2, inputs=[state], outputs=[img_head, text_notion, state])
    btn_get_feedback.click(fn=fn_get_feedback, inputs=[state], outputs=[img_head, text_notion, state])
    btn_all.click(fn=fn_all, inputs=[], outputs=[])
    btn_all_without_init.click(fn=fn_all_without_init, inputs=[], outputs=[])

demo.launch()