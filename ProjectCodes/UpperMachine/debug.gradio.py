import gradio as gr
from Tools.LowerMachine.CarCamera import get_car_camera_image
from Tools.LowerMachine.CarContorller import CarContorller
from Tools.Camera import Camera
from Tools.CarDetection import CarDetection
from Tools.DINO import DINO_with_camera

car_controller = CarContorller()

camera = Camera()
car_detection = CarDetection()
dino = DINO_with_camera()


with gr.Blocks() as App:
    gr.Markdown("""# 小车调试工具""")

    with gr.Tab("车载摄像头调试"):
        img_esp32cam = gr.Image(label="车载摄像头画面")
        txt_esp32cam_ip = gr.Textbox(label="ESP32CAM IP", value="192.168.2.140")
        txt_esp32cam_port = gr.Number(label="ESP32CAM Port", value=80)
        btn_esp32cam = gr.Button("获取车载摄像头画面")
    with gr.Tab("车载传感器控制"):
        gr.Markdown("""
        调试前请创建Socket，调试结束后务必关闭Socket，你可以通过获取Socket状态来判断Socket是否创建成功或关闭成功。
        通常，参数中，只需要修改Target Address和Target Port即可，其他参数取默认即可。
        """)

        with gr.Row():
            txt_pico_socket_state = gr.Textbox(label="PICO Socket State", value=False)
            btn_pico_socket_state = gr.Button("获取PICO Socket状态")
        txt_pico_host = gr.Textbox(label="Host", value="")
        txt_pico_port = gr.Number(label="Port", value=8083)
        txt_pico_target_address = gr.Textbox(label="Target Address", value="192.168.2.236")
        txt_pico_target_port = gr.Number(label="Target Port", value=5000)
        btn_pico_socket_connect = gr.Button("创建Socket")
        txt_pico_data = gr.Textbox(label="传输报文", value="0 0 1.0", info="格式：{执行编号1} {执行编号2} {参数}，其中执行编号1取值0或1，对应马达和水泵，执行编号2取值0到5，对应马达的不同动作，参数为浮点数，对应执行时间")
        btn_pico_socket_send = gr.Button("发送报文")
        btn_pico_socket_close = gr.Button("关闭Socket")
    with gr.Tab("电脑端摄像头调试"):
        gr.Markdown("""
        调试前请创建模型，调试完毕请关闭模型，以便于释放资源。你可以通过获取状态来判断模型是否创建成功或释放成功。请不要多次重复点击创建模型或释放模型按钮，你可以通过观察打印输出或者内存占用，了解程序是否工作。
        """)
        with gr.Row():
            img_usb = gr.Image(label="电脑摄像头画面")
            img_usb_processed = gr.Image(label="模型处理结果")
        with gr.Row():
            txt_usb_det_model_path = gr.Textbox(label="Det模型路径", value="Source/CarDetection")
            txt_usb_det_model_state = gr.Textbox(label="Det模型状态", value=False)
            btn_usb_det_model_state = gr.Button("获取Det模型状态")
            btn_usb_det_model_set = gr.Button("加载Det模型")
            btn_usb_det_model_close = gr.Button("关闭Det模型")
        with gr.Row():
            txt_usb_dino_model_path = gr.Textbox(label="DINO模型路径", value="Source/GroundingDINO")
            txt_usb_dino_model_state = gr.Textbox(label="DINO模型状态", value=False)
            btn_usb_dino_model_state = gr.Button("获取DINO模型状态")
            btn_usb_dino_model_set = gr.Button("加载DINO模型")
            btn_usb_dino_model_close = gr.Button("关闭DINO模型")

        btn_usb = gr.Button("获取电脑摄像头画面")
        btn_usb_det = gr.Button("获取Det模型检测结果")
        txt_usb_dino_text = gr.Textbox(label="DINO模型Prompt", value="Plants")
        btn_usb_dino = gr.Button("获取DINO模型检测结果")


    btn_esp32cam.click(fn=get_car_camera_image, inputs=[txt_esp32cam_ip, txt_esp32cam_port], outputs=[img_esp32cam])

    btn_pico_socket_state.click(fn=car_controller.get_socket_status, outputs=[txt_pico_socket_state])
    btn_pico_socket_connect.click(fn=car_controller.set_socket, inputs=[txt_pico_host, txt_pico_port, txt_pico_target_address, txt_pico_target_port], outputs=[])
    btn_pico_socket_send.click(fn=car_controller.send_to_pico, inputs=[txt_pico_data], outputs=[])
    btn_pico_socket_close.click(fn=car_controller.close_socket, outputs=[])

    btn_usb.click(fn=lambda: camera.get_frame()[1], outputs=[img_usb])
    btn_usb_det_model_state.click(fn=lambda: car_detection.model_flag, outputs=[txt_usb_det_model_state])
    btn_usb_det_model_set.click(fn=lambda x: car_detection.set_model(model_path=x, device="CPU", camera=camera), inputs=txt_usb_det_model_path)
    btn_usb_det_model_close.click(fn=car_detection.release_model)
    btn_usb_det.click(fn=lambda: (car_detection.infer(), car_detection.get_draw_img())[1], outputs=[img_usb_processed])
    btn_usb_dino_model_state.click(fn=lambda: dino.model_flag, outputs=[txt_usb_dino_model_state])
    btn_usb_dino_model_set.click(fn=lambda x: dino.set_model(model_id=x, device="cpu", camera=camera), inputs=txt_usb_dino_model_path)
    btn_usb_dino_model_close.click(fn=dino.release_model)
    btn_usb_dino.click(fn=lambda x: (dino.infer(x), dino.get_draw_img())[1], inputs=txt_usb_dino_text, outputs=[img_usb_processed])



if __name__ == "__main__":
    # App.launch(server_name="0.0.0.0", server_port=8501)
    App.launch()

