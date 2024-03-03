# TL_SLR（Transfer Learning—Sign Language Recognition）

# 基于迁移学习的手语识别方法

## 姿态网络使用movenet/openvino官方模型，手部节点网络使用squeenet，分类网络使用micronet-m3/mobilenet-v3/GoogleNet/ResNet

movenet参考项目：[lee-man/movenet-pytorch: A PyTorch port of Google Movenet (inference only) (github.com)](https://github.com/lee-man/movenet-pytorch)

squeenet手部识别参考项目：[Eric.Lee2021 / handpose_x · GitCode](https://codechina.csdn.net/EricLee/handpose_x)

movenet参考项目：[liyunsheng13/micronet (github.com)](https://github.com/liyunsheng13/micronet)

openvino官方模型：[openvinotoolkit/open_model_zoo: Pre-trained Deep Learning models and demos (high quality and extremely fast) (github.com)](https://github.com/openvinotoolkit/open_model_zoo)



项目的本地模式可以部署于PC、树莓派上

在线模式的客户端可以选择PC、树莓派

在线模式的服务端可以选择带有可连接IP和端口进行TCP连接的PC、服务器



要将项目部署于树莓派上，需使用NCS2设备，设置MYRAID服务，操作方法参考：

[Install OpenVINO™ Runtime for Raspbian OS — OpenVINO™ documentationCopy to clipboardCopy to clipboardCopy to clipboardCopy to clipboardCopy to clipboardCopy to clipboardCopy to clipboardCopy to clipboardCopy to clipboardCopy to clipboard — Version(2022.3)](https://docs.openvino.ai/2022.3/openvino_docs_install_guides_installing_openvino_raspbian.html)





# 一、环境配置

## 1、环境创建与激活

首先需安装 python>=3.10.2，然后将项目移至全英文路径下

进入项目路径打开cmd/bash，根据以下命令创建并激活环境 

```bash
<Windows! -->
python -m venv my_env
my_env\Scripts\activate 
python -m pip install --upgrade pip

<Linux! -->
python3 -m venv my_env
source my_env/bin/activate 
python -m pip install --upgrade pip
```

## 2、特殊库配置安装

### pytorch

安装torch>=2.1.1,torchaudio>=2.1.1 torchvision>=0.16.1

在有nvidia服务的设备上，使用以下命令安装

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

在没有nvidia服务的设备上，使用以下命令安装

```bash
pip3 install torch torchvision torchaudio
```



### openvino

若使用openvino框架进行推理，请使用以下命令安装环境

```bash
pip install openvino-dev==2022.3.1
```

在树莓派上部署详见我的博客：[树莓派4B配置Openvino-CSDN博客](https://blog.csdn.net/2301_76725922/article/details/136389051)



### *TensorRT（可选）

若使用TensorRt框架进行推理，请确保

tensorrt>=8.6.1,cuda>=11.8,cudnn>=8.7

tensorrt安装包可按照torch与cuda版本在官网选择下载：[NVIDIA TensorRT Download | NVIDIA Developer](https://developer.nvidia.com/tensorrt-download)

或者按照项目已包含的包安装tensorrt，执行以下命令

```bash
cd common/Tensorrt
pip install tensorrt-8.6.1-cp310-none-win_amd64.whl   <根据操作系统与python版本选择对应wheel包安装>
```

安装后可使用以下命令依次查看torch，cuda、cudnn以及tensorrt的版本

```bash
python -c "import torch;print(torch.__version__);print(torch.version.cuda);print(torch.backends.cudnn.version())"
python -c "import tensorrt;print(tensorrt.__version__)"
```

## 3、安装其他环境依赖

```bash
pip install -r requirements.txt
```

## *4、安装torch2trt（TensorRT安装后可选择）

具体参考：[NVIDIA-AI-IOT/torch2trt: An easy to use PyTorch to TensorRT converter (github.com)](https://github.com/NVIDIA-AI-IOT/torch2trt)

执行以下命令：

```bash
cd common/Torch2trt
python setup.py install
```



# 快速开始

## 1、下载模型

前往 [Release models-v1.1 · guoX66/TL_SLR (github.com)](https://github.com/guoX66/TL_SLR/releases/tag/release-v1.1)

下载release中的模型压缩包，解压后将得到的models文件夹全部移到根目录下

## 2、配置参数

修改 Cfg.yaml 中inference相关参数，具体取值和对应含义如下：

```yaml
base:
  device: NVIDIA pytorch                 # 计算平台，CPU、MYRIAD(NCS2)计算棒、NVIDIA pytorch、NVIDIA tensorrt
  platform: pc                           # 平台类型，树莓派填 rp,其余填pc
  pose_net: ov                           # movenet填 mv openvino模型填 ov , MYRIAD服务下只能使用ov
  env_path: ../my_env                    # 填写python环境路径
  size: 256x256                          # 设置输入分类模型的图片大小
  view_mode: 3                           # 显示转换效果，0为不显示，1为显示骨架图，2为显示骨架在实际图像上的效果，3为全部显示

inference:
  IP: 203.135.98.31                      # 服务器IP
  model: micronet_m3                     # 填写放在models对应子文件夹中的训练好的分类模型类型
  port: 37942                            # 服务器端口号
  source: assets/test.mp4                # 视频源，可以填本地视频路径，摄像头填 0


train:
  Normalized_matrix:                     # 标准化设置
  - - 0.485
    - 0.456
    - 0.406
  - - 0.229
    - 0.224
    - 0.225
  batch_size: 16
  divide_present: 0.8                    # 拆分验证集比例
  epoch: 50                              # 设置迭代次数
  gamma: 0.95                            # 学习率衰减系数，也即每个epoch学习率变为原来的0.95
  learn_rate: 0.001                      # 设置学习率
  model: resnet18                        # 选择模型，可选 micronet_m3,mobilenet_v3,googlenet,resnet18,resnet50
  show_mode: Simple                      # 设模型层数信息写入log中的模式:'All'  'Simple'  'No'
  step_size: 1                           # 学习率衰减步长
  write_process: false                   # 设置是否将训练过程写入log中


```

## 3、运行系统

运行main.py程序启动系统，选择相应的模式开始识别

```bash
python main.py
```



# 训练流程

## 分类模型训练

### 1、数据集准备

将拍摄好的手语视频按帧分成图像，存入data/static文件夹中，按语义分类放入各个子文件夹中，然后以类的名称命名子文件夹。结构示例如下

```
--data
    --static
        --谢谢
            --img1.jpg
            --img2.jpg
            ...
        --我们
            --img1.jpg
            --img2.jpg
            ...
        --生活
        ...
```



### 2、转换程序

根据实际情况，可以选择不同的device和view_mode参数

可以直接修改 Cfg.yaml 中device和view_mode参数，也可以通过命令行传参设置，但需要注意：

1、NVIDIA tensorrt和NVIDIA pytorch都是使用pytorch进行转换 

2、当device为CPU时，将使用movenet进行姿态提取，且Cfg.yaml 中的pose_net参数设置将无效

3、命令行传参设置会覆盖Cfg.yaml中的设置，示例如下:

```bash
python pose.py --device "NVIDIA pytorch" --view_mode 0
```

### 3、模型训练

与2中类似，训练参数可以直接修改 Cfg.yaml 中训练相关的参数，也可以通过命令行传参设置，标准化设置须在Cfg.yaml中进行

请注意命令行传参设置会覆盖Cfg.yaml中的设置，示例如下:

```bash
python train.py --model micronet_m3 --epoch 100 --batch_size 4 --divide_present 0.8 --lr 0.001 --step_size 1 --gamma 0.95 --write_process false --show_mode Simple
```



## 模型转换

将log/train_process中训练好的pth模型移到models/pth文件夹下 ，并重命名为model-{模型名称}.pth

### 1、转openvino框架

需要修改 Cfg.yaml 中 base-env_path 参数,填写python环境路径，然后运行以下程序

```bash
cd models
python pth2ov_demo.py --model googlenet
```

### 2、转TensorRt框架

```bash
cd models
python pth2trt_demo.py --model googlenet  --fp16 True
```



# 模型部署

## 本地端

部署和转换完成后，运行main.py程序启动系统，选择转换好的分类网络进行推理

最后运行main.py程序进入系统

## 服务端

配置好环境后，在云服务器上，用以下命令开启服务端

```bash
python TR_ser.py --port 8800 <可改为需要的端口>
```

在使用设备上，修改Cfg.yaml，设置为相应的IP和端口，运行main.py程序启动系统，选择本地模式连接服务器

```bash
python main.py
```
