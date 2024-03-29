import shutil
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision import models
import json
import torch
import os
import matplotlib.pyplot as plt
import time
import matplotlib.font_manager as fm


def my_convert_model(model_path, ie, device):
    model = ie.read_model(model_path)
    compiled_model = ie.compile_model(model=model, device_name=device)
    # compiled_model = ie.compile_model(model=model, device_name="MYRIAD")
    return compiled_model


def inference_(compiled_model, input_image):
    input_image = np.expand_dims(input_image, 0)
    # input_image = input_image.astype('float16')
    input_layer_ir = compiled_model.input(0)
    output_layer_ir = compiled_model.output(0)
    result = compiled_model([input_image])[output_layer_ir]
    return result


def ini_sys():
    trt = False  # tensorrt 是否可用
    MYD = False  # 神经棒NCS2 是否可用
    try:  # pytorch 是否可用
        import torch
        pyt = torch.cuda.is_available()
    except:
        pyt = False
    if pyt:  # pytorch可用时 tensorrt才可用
        try:  # tensorrt 是否可用
            import tensorrt
            trt = True
        except:
            pass
    try:  # openvino 是否可用
        from openvino.runtime import Core
        _ = Core()
        OV = True
        OV_devices = Core().available_devices
        if 'MYRIAD' in OV_devices:  # NCS2 是否可用
            MYD = True
    except:
        OV = False
    device_list = ['CUDA', 'TensorRt', 'Openvino', 'MYRIAD']
    avi_list = [pyt, trt, OV, MYD]
    d_list = [device_list[i] for i in range(len(device_list)) if avi_list[i]]  # 可用服务列表
    return d_list


def remove_file(old_path, new_path):
    filelist = os.listdir(old_path)  # 列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    for file in filelist:
        src = os.path.join(old_path, file)
        dst = os.path.join(new_path, file)
        shutil.move(src, dst)


def crop(img, crop_size):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = gray_img.shape[1]
    width = gray_img.shape[0]
    # print(gray_img)
    not_none = np.where(gray_img != 0)
    min_x = min(not_none[1])
    min_y = min(not_none[0])
    max_x = max(not_none[1])
    max_y = max(not_none[0])
    if min_x <= crop_size:
        min_x = 0
    else:
        min_x = min_x - crop_size
    if min_y <= crop_size:
        min_y = 0
    else:
        min_y = min_y - crop_size

    if max_x + crop_size >= width:
        max_x = width
    else:
        max_x = max_x + crop_size

    if max_y + crop_size >= height:
        max_y = height
    else:
        max_y = max_y + crop_size
    crop_img = img[min_y:max_y, min_x:max_x]
    crop_img = cv2.resize(crop_img, (width, height), interpolation=cv2.INTER_CUBIC)
    return crop_img


def cv2ImgAddText(img, text, left, top, textColor, textSize=20):
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(fm.findfont(fm.FontProperties(family='SimHei')), textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def make_model(modelinfo, n, path, device):
    model_ft = sele_model(modelinfo)
    layer_list = get_layers_name(model_ft)
    last_layers_name = layer_list[-1][0]
    in_features = layer_list[-1][1].in_features
    layer1 = nn.Linear(in_features, n)
    _set_module(model_ft, last_layers_name, layer1)
    model_ft.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(path, map_location=device).items()})
    if isinstance(model_ft, torch.nn.DataParallel):
        model_ft = model_ft.module
    return model_ft


def make_train_model(modelinfo, n):
    model_ft = sele_model(modelinfo, train=True)
    layer_list = get_layers_name(model_ft)
    last_layers_name = layer_list[-1][0]
    in_features = layer_list[-1][1].in_features
    layer1 = nn.Linear(in_features, n)
    _set_module(model_ft, last_layers_name, layer1)
    return model_ft


def sele_model(Model, train=False):
    model_dict = {
        'resnet18': models.resnet18(weights=models.ResNet18_Weights.DEFAULT),  # 残差网络
        'resnet34': models.resnet34(weights=models.ResNet34_Weights.DEFAULT),
        'resnet50': models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        'resnet101': models.resnet101(weights=models.ResNet101_Weights.DEFAULT),
        'googlenet': models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT),
        'mobilenet_v3': models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT),
    }
    if train:
        return model_dict[Model.model]
    else:
        return model_dict[Model]


def show(c, model, txt_list):
    layer_list = get_layers_name(model)  # 获取模型各层信息
    if c.show_mode == 'All':
        for layers in layer_list:
            txt_list.append(str(layers) + '\r\n')

    elif c.show_mode == 'Simple':
        for layers in layer_list:
            txt_list.append(str(layers[0]) + '\r\n')


def get_label_list(imgpath):
    file_path = f'./{imgpath}'

    path_list = []

    for i in os.walk(file_path):
        path_list.append(i)

    label_dict = dict()
    label_name_list = []
    label_list = []

    for i in range(len(path_list[0][1])):
        label = path_list[0][1][i]
        label_dict[label] = path_list[i + 1][2]

    for i in label_dict.keys():
        label_list.append(i)
        for j in label_dict[i]:
            label_name_list.append([i, j])

    return label_name_list, label_dict, label_list


def bar(i, t, start, des, train=True, loss=0, acc=0):
    l = 50
    f_p = i / t
    n_p = (t - i) / t
    finsh = "▓" * int(f_p * l)
    need_do = "-" * int(n_p * l)
    progress = f_p * 100
    dur = time.perf_counter() - start
    if train:
        proc = "\r{}({}/{}轮):{:^3.2f}%[{}->{}] 用时:{:.2f}s 验证集上损失:{:.3f} 正确率: {:.2f} %".format(des, i, t,
                                                                                                          progress,
                                                                                                          finsh,
                                                                                                          need_do, dur,
                                                                                                          loss, acc)
    else:
        proc = "\r{}:{:^3.2f}%[{}->{}] 用时:{:.2f}s".format(des, progress, finsh, need_do, dur)
    print(proc, end="")


def add_log(txt, txt_list, is_print=True):
    if is_print:
        print(txt)
    txt_list.append(txt + '\r\n')


def write_log(in_path, filename, txt_list):
    try:
        os.mkdir(in_path)
    except:
        pass
    path = os.path.join(in_path, filename + '.txt')
    content = ''
    for txt in txt_list:
        content += txt
    with open(path, 'w+', encoding='utf8') as f:
        f.write(content)


def train_dir(filename):
    try:
        os.mkdir('log/train_process')
    except:
        pass
    file_path = 'log/train_process/' + filename
    try:
        os.mkdir(file_path)
    except:
        pass


def process(img_paths):
    seq = []
    for i in img_paths:
        train_label = int(i[1])
        print(train_label)
        train_label = torch.IntTensor([train_label])
        print(train_label)
    return seq


def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def get_layers_name(model):
    layer_list = []
    for layer in model.named_modules():
        layer_list.append(layer)
    return layer_list


def get_layers(model):
    layer_list = []
    for layer in model.modules():
        layer_list.append(layer)
    return layer_list


def write_json(test_dict, path):
    try:
        os.mkdir('log')
    except:
        pass
    path = os.path.join('log', path)
    json_str = json.dumps(test_dict)
    with open(f'{path}.json', 'w') as json_file:
        json_file.write(json_str)


def make_plot(data, mode, filename, epoch):
    file_path = 'log/train_process/' + filename
    if mode == 'loss':
        title = 'LOSS'
        path = os.path.join(file_path, 'LOSS-' + filename)
    elif mode == 'acc':
        title = 'ACC曲线'
        path = os.path.join(file_path, 'ACC-' + filename)
    plt.figure(figsize=(12.8, 9.6))
    x = [i + 1 for i in range(epoch)]
    plt.plot(x, data)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(title + '-' + filename, fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(f'{path}.png')


def npy_crop(feature, height, width, crop_rate):
    not_zero = np.array([i for i in feature if (i[0] != 0 and i[1] != -1)])
    min_x = min(not_zero[:, 1])
    min_y = min(not_zero[:, 0])
    max_x = max(not_zero[:, 1])
    max_y = max(not_zero[:, 0])

    crop_size_y = crop_rate * height
    crop_size_x = crop_rate * width
    crop_arr = np.array([crop_size_y, crop_size_x])
    arr = np.array([min_y, min_x])
    one_arr = np.array([1, 1])
    out_put = np.array([i - arr + crop_arr if (i[0] != 0 and i[1] != -1) else i + one_arr for i in feature])
    out_put[:, 0] = out_put[:, 0] / (max_y + 2 * crop_size_y)
    out_put[:, 1] = out_put[:, 1] / (max_x + 2 * crop_size_x)
    return out_put


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def train_process_np(path_dict, seq_len, class_dict):
    seq = []
    count = 0
    class_to_id_dict = {}
    for label in class_dict.keys():
        real_label = label.split('-')[0]

        if real_label not in class_to_id_dict.keys():
            count += 1
        class_to_id_dict[real_label] = count - 1

    for label in path_dict.keys():
        data = path_dict[label]
        for i in range(len(data) - seq_len):
            train_seq = []
            for j in range(i, i + seq_len):
                path = data[j]
                directory = os.path.dirname(path)
                file = os.path.basename(directory)
                real_label = file.split('-')[0]
                x = np.load(path, allow_pickle=True)
                train_seq.append(x)

            train_label = np.array([class_to_id_dict[real_label]])
            train_seq = torch.FloatTensor(np.array(train_seq))
            train_label = torch.from_numpy(train_label).type(torch.int64).view(-1)
            seq.append((train_seq, train_label))

    seq = MyDataset(seq)

    return seq, {int(class_to_id_dict[k]): k for k in class_to_id_dict.keys()}
