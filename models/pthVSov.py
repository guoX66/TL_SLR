import numpy as np
import torch
import json
import sys

sys.path.append("..")
from _utils.OV_detect import make_model, my_convert_model, inference_
from handpose.models.squeezenet import squeezenet1_1
from movenet.models.model_factory import load_model
import time
from openvino.runtime import Core
from mico.backbone import MicroNet
from mico.utils.defaults import _C as cfg
from _utils.myutils import add_log


def te_fast(model, ov_path, data, txt_list):
    device = torch.device('cpu')
    ie = Core()
    ov_model = my_convert_model(ov_path, ie, device='CPU')
    t_data = torch.FloatTensor(data).to(device)
    data = data[0]
    data = data.astype('float16')
    start_time = time.time()
    with torch.no_grad():
        for i in range(1000):
            out1 = model(t_data)
        end_time = time.time()
        add_log(f'    Torch-cpu cost {round((end_time - start_time), 2)}s', txt_list)
        start_time2 = time.time()
        for i in range(1000):
            out2 = inference_(ov_model, data)
        end_time2 = time.time()
        add_log(f'    openvino cost {round((end_time2 - start_time2), 2)}s', txt_list)
        rou = round((end_time - start_time) / (end_time2 - start_time2), 1)
        err = np.sum(np.abs(out2 - out1.cpu().numpy()))
        add_log(f'    openvino is {rou} times as Torch-cpu', txt_list)
        add_log(f'    err={err}', txt_list)


def te_ov():
    with open("../log/class_id.json", 'r', encoding='UTF-8') as f:
        class_dict = json.load(f)
    class_dict = {int(k): class_dict[k] for k in class_dict.keys()}
    n_label = len(list(class_dict.keys()))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    pose_model = load_model("movenet_lightning", ft_size=48)
    pose_model.to(device)
    pose_model.eval()

    model_path = '../handpose/weights/squeezenet1_1-size-256-loss-0.0732.pth'
    hand_model = squeezenet1_1(num_classes=42)
    chkpt = torch.load(model_path, map_location=device)
    hand_model.load_state_dict(chkpt)
    hand_model.to(device)
    hand_model.eval()

    path1 = 'pth/model-micronet_m3.pth'
    classify_model1 = MicroNet(cfg, num_classes=n_label)
    classify_model1.load_state_dict(torch.load(path1, map_location=device), strict=False)
    classify_model1.to(device)
    classify_model1.eval()

    path2 = 'pth/model-mobilenet_v3.pth'
    classify_model2 = make_model('mobilenet_v3', n_label, path2, device)
    classify_model2.load_state_dict(torch.load(path2, map_location=device), strict=False)
    classify_model2.to(device)
    classify_model2.eval()

    rd = np.random.RandomState(888)
    data1 = rd.random((1, 3, 256, 256))  # 随机生成一个 [0,1) 的浮点数 ，5x5的矩阵
    data2 = rd.random((1, 192, 192, 3))  # 随机生成一个 [0,1) 的浮点数 ，5x5的矩阵

    ov_path1 = 'openvino_model/micronet_m3.xml'
    ov_path2 = 'openvino_model/mobilenet_v3.xml'
    ov_pose_path = 'openvino_model/pose_model.xml'
    ov_hand_path = 'openvino_model/hand_model.xml'

    txt_list = []
    t = time.strftime('%Y-%m-%d %H:%M', time.localtime())
    add_log(f'time:{t}:', txt_list)
    add_log('pose_model(Movenet):', txt_list)
    te_fast(pose_model, ov_pose_path, data2, txt_list)
    add_log('hand_model(squeezenet):', txt_list)
    te_fast(hand_model, ov_hand_path, data1, txt_list)
    add_log('classify_model(micronet_m3):', txt_list)
    te_fast(classify_model1, ov_path1, data1, txt_list)
    add_log('classify_model(mobilenet_v3):', txt_list)
    te_fast(classify_model2, ov_path2, data1, txt_list)

    path = '../log/pth-VS-ov.txt'
    content = ''
    for txt in txt_list:
        content += txt
    with open(path, 'w+', encoding='utf8') as f:
        f.write(content)


if __name__ == '__main__':
    te_ov()
