import gradio as gr
from Tools.LowerMachine.CarCamera import get_car_camera_image

with gr.Blocks() as App:
    gr.Markdown("""# 小车调试工具""")

    with gr.Tab("车载摄像头调试"):
        img_car = gr.Image(label="车载摄像头画面")
        txt_esp32cam_ip = gr.Textbox(label="ESP32CAM IP", value="192.168.2.140")
        txt_esp32cam_port = gr.Number(label="ESP32CAM Port", value=80)
        btn_car = gr.Button("获取车载摄像头画面")

    with gr.Tab("控制台"):
        pass


    btn_car.click(fn=get_car_camera_image, inputs=[txt_esp32cam_ip, txt_esp32cam_port], outputs=[img_car])

if __name__ == "__main__":
    # App.launch(server_name="0.0.0.0", server_port=8501)
    App.launch()

