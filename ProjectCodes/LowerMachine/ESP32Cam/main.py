import network
import usocket as socket
import camera
import time

def connect_to_wifi(ssid, password):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        print('当前已连接到 Wi-Fi，正在断开连接...')
        wlan.disconnect()
    wlan.connect(ssid, password)
    while not wlan.isconnected():
        time.sleep(1)
    print('WiFi connected')
    print('IP address:', wlan.ifconfig()[0])

def start_server():
    # 初始化摄像头
    camera.deinit()
    camera.init(2, format=camera.JPEG)
    camera.quality(90)  # 设置JPEG质量

    # 创建套接字并绑定到端口80
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 80))
    s.listen(1)
    print('Listening on socket...')

    while True:
        print("Listening")
        conn, addr = s.accept()
        print('Connected by', addr)
        request = conn.recv(1024)
        print('Request:', request.decode())

        # 检查是否为 GET 请求
        if request.decode().startswith('GET'):
            # 捕获图像
            img = camera.capture()
            # 发送图像
            conn.send(img)
        else:
            # 发送 404 响应
            conn.send(b"ERROR 404")

        conn.close()

# 连接到 Wi-Fi
connect_to_wifi('***', '********')

# 启动服务器
start_server()
