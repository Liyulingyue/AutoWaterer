# UpperMachine

上位机模块需要在电脑上运行，主要负责与下位机进行通信，并展示数据。

开发者使用的软硬件资源如下所示：
- 处理器: 13th Gen Intel(R) Core(TM) i7-13700K   3.40 GHz
- 内存: 32.0 GB
- Python 3.10

## 安装依赖
依赖参考 requirements.txt 文件，可以使用以下命令安装:
```bash
pip install -r requirements.txt
```

## 文件结构
- 代码: 所有的代码都模块化封装在Tools文件夹中，大部分文件名即表明对应文件的功能和作用。
- 模型与资源文件: 所有文件都保存在Source文件夹中

## 运行与调试
为了方便调试，你可以运行当前目录下的 gradio 文件，里面界面化地展示了所有功能。