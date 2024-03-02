# TL_SLR（Transfer Learning—Sign Language Recognition）

# 基于迁移学习的手语识别方法

## situation 1

在openvino框架下姿态网络使用movenet，手部节点网络使用squeenet，分类网络使用micronet-m3；在TensorRt框架下姿态网络使用movenet，手部节点网络使用resnet50，分类网络使用googlenet或resnet18.

## situation 2

要将项目部署于树莓派上，需使用NCS2设备，设置MYRAID服务，操作方法参考：

[Install OpenVINO™ Runtime for Raspbian OS — OpenVINO™ documentationCopy to clipboardCopy to clipboardCopy to clipboardCopy to clipboardCopy to clipboardCopy to clipboardCopy to clipboardCopy to clipboardCopy to clipboardCopy to clipboard — Version(2022.3)](https://docs.openvino.ai/2022.3/openvino_docs_install_guides_installing_openvino_raspbian.html)

姿态网络使用openvino官方模型，其余与情况1相同。



movenet参考项目：[lee-man/movenet-pytorch: A PyTorch port of Google Movenet (inference only) (github.com)](https://github.com/lee-man/movenet-pytorch)

squeenet手部识别参考项目：[Eric.Lee2021 / handpose_x · GitCode](https://codechina.csdn.net/EricLee/handpose_x)

movenet参考项目：[liyunsheng13/micronet (github.com)](https://github.com/liyunsheng13/micronet)

openvino官方模型：[openvinotoolkit/open_model_zoo: Pre-trained Deep Learning models and demos (high quality and extremely fast) (github.com)](https://github.com/openvinotoolkit/open_model_zoo)



# 环境部署

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

然后安装torch>=2.1.1,torchaudio>=2.1.1 torchvision>=0.16.1

在有nvidia服务的设备上，使用以下命令安装

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

在没有nvidia服务的设备上，使用以下命令安装

```bash
pip3 install torch torchvision torchaudio
```

若使用openvino框架进行推理，请使用以下命令安装环境

```bash
pip install openvino-dev==2022.3.1
```

若使用TensorRt框架进行推理，请确保

tensorrt>=8.6.1,cuda>=11.8,cudnn>=8.7

tensorrt安装包可按照torch与cuda版本在官网选择下载：[NVIDIA TensorRT Download | NVIDIA Developer](https://developer.nvidia.com/tensorrt-download)



或者按照项目已包含的包安装tensorrt，执行以下命令

```bash
cd Tensorrt
pip install tensorrt-8.6.1-cp310-none-win_amd64.whl   <根据操作系统与python版本选择对应wheel包安装>
```



安装后可使用以下命令依次查看torch，cuda、cudnn以及tensorrt的版本

```bash
python -c "import torch;print(torch.__version__);print(torch.version.cuda);print(torch.backends.cudnn.version())"
python -c "import tensorrt;print(tensorrt.__version__)"
```

安装其他环境依赖

```bash
pip install -r requirements.txt
```



安装torch2trt, 具体参考：[NVIDIA-AI-IOT/torch2trt: An easy to use PyTorch to TensorRT converter (github.com)](https://github.com/NVIDIA-AI-IOT/torch2trt)

执行以下命令：

```bash
cd Torch2trt
python setup.py install
```



# 快速开始

## 1、下载模型

前往 [Release models · guoX66/TL_SLR (github.com)](https://github.com/guoX66/TL_SLR/releases/tag/release-v1.0.0)

下载release中的模型压缩包，解压后将得到的文件夹全部移到models文件夹下

将movenet_lightning.pth移到movenet/_models目录下

## 2、配置参数

前往 _utils/Cfg.yaml 中修改inference相关参数

```yaml
device: NVIDIA pytorch                 # 计算平台，CPU、MYRAID、NVIDIA pytorch、NVIDIA tensorrt

inference:
  IP: 203.135.98.31                      # 服务器IP
  platform: pc                           # 平台类型，树莓派填 rp,其余填pc
  port: 37942                            # 服务器端口号
  pose_net: mv                           # movenet填 mv openvino模型填 ov
  source: assets/test.mp4                # 视频源，可以填本地视频路径，摄像头填 0

python_path: ../my_env

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
  size: 256x256                          # 设置输入分类模型的图片大小
  step_size: 1                           # 学习率衰减步长
  write_process: false                   # 设置是否将训练过程写入log中

python_path: ../my_env
view_mode: 3                           # 显示转换效果，0为不显示，1为显示骨架图，2为显示骨架在实际图像上的效果，3为全部显示
```

## 3、运行系统

运行main.py程序

```bash
python main.py
```



# 训练部署流程

## 分类模型训练

### 1、数据集准备

将拍摄好的手语视频按帧分成图像，存入static文件夹中，按语义分类放入各个子文件夹中，然后以类的名称命名子文件夹。结构示例如下

```
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

### 2、配置参数

前往 _utils/Cfg.yaml 中修改inference相关参数



### 3、转换程序

```bash
python pose.py
```

### 4、模型训练

```bash
python train.py
```



## 模型转换

将train_process中训练好的pth模型移到models/pth文件夹下 ，并重命名为model-{模型名称}.pth

### 1、转openvino框架

```bash
cd models
python pth2ov_demo.py --env_path ../my_env
```

### 2、转TensorRt框架

```bash
cd models
python pth2trt_demo.py --model googlenet
```

运行main.py程序



## 模型部署

转换完成后，前往 _utils/Cfg.yaml 中修改inference相关参数

在树莓派上部署详见我的博客：[树莓派4B配置Openvino-CSDN博客](https://blog.csdn.net/2301_76725922/article/details/136389051)

最后运行main.py程序进入系统
