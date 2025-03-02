# UpperMachine

上位机模块需要在电脑上运行，主要负责与下位机进行通信，并展示数据。

开发者使用的软硬件资源如下所示：
- 处理器: 13th Gen Intel(R) Core(TM) i7-13700K   3.40 GHz
- 内存: 32.0 GB
- Python 3.10

## 上位机环境布置
本项目中，需要依赖顶端USB摄像头对小车进行监视，因此你需要
1. 将USB摄像头固定在天花板或足够高的位置，并保证摄像头可以垂直拍摄地面。小车、植物均在摄像头画面中。
2. 通过USB连接线或USB延长线，将USB摄像头连接到电脑。
3. 测试过程中，请将小车摆放在摄像头画面的左上角，并保证小车朝向为摄像头画面y轴正方向，植物位于摄像头画面的右半部分。

## 安装依赖
依赖参考 requirements.txt 文件，可以使用以下命令安装:
```bash
pip install -r requirements.txt
```

## 文件结构
- 代码: 所有的代码都模块化封装在Tools文件夹中，大部分文件名即表明对应文件的功能和作用。
- 模型与资源文件: 所有文件都保存在Source文件夹中
  - CarDetection: 该文件夹保存了PPyoloE+的导出模型文件，如何训练以及导出模型请参考 https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyoloe
  - GroundingDINO: 该文件夹保存了GroundingDINO的导出模型文件，如何训练以及导出模型请参考 https://huggingface.co/IDEA-Research/grounding-dino-base/tree/main
  - MiniCPM-V-2.6-ov: 该文件夹保存了Openvino量化后的MiniCPM模型文件，如何导出模型请参考 https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/minicpm-v-multimodal-chatbot , 中国地区可以通过 modelscope 下载原始模型文件，以避免网络问题 modelscope download --model 'OpenBMB/MiniCPM-V-2_6' --local_dir 'openbmb/MiniCPM-V-2_6'
  - Images: 运行过程中本地的图片会保存在这里
  - Logs: 运行过程中生成的日志会保存在这里

## 运行与调试
为了方便调试，你可以运行当前目录下的 gradio 文件，里面界面化地展示了所有功能。