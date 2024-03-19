import numpy as np
import torch
import json
import sys

sys.path.append("..")
from common.handpose.models.squeezenet import squeezenet1_1
from common.movenet.models.model_factory import load_model
import time
from openvino.runtime import Core
from common.mico.backbone import MicroNet
from common.mico.utils.defaults import _C as cfg
from _utils.myutils import add_log, my_convert_model, inference_


def te_fast(model, ov_path, data, txt_list, mode='class'):
    device = torch.device('cpu')
    ie = Core()
    ov_model = my_convert_model(ov_path, ie, device='CPU')
    t_data = torch.FloatTensor(data).to(device)
    data = data[0]
    data = data.astype('float16')
    start_time = time.time()
    model.eval()
    model.to(device)
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
        add_log(f'    openvino is {rou} times as Torch-cpu', txt_list)
        if mode == 'handpose':
            err = np.sum(np.abs(out2 - out1.cpu().numpy()))
            add_log(f'    err={err}', txt_list)


def te_ov(pose_model, hand_model):
    device = torch.device('cpu')
    pose_model.to(device)
    hand_model.to(device)

    rd = np.random.RandomState(888)
    data1 = rd.random((1, 3, 256, 256))  # 随机生成一个 [0,1) 的浮点数 ，5x5的矩阵
    data2 = rd.random((1, 192, 192, 3))  # 随机生成一个 [0,1) 的浮点数 ，5x5的矩阵

    ov_pose_path = 'openvino_model/pose_model.xml'
    ov_hand_path = 'openvino_model/hand_model.xml'

    txt_list = []
    t = time.strftime('%Y-%m-%d %H:%M', time.localtime())
    add_log(f'time:{t}:', txt_list)
    add_log('pose_model(Movenet):', txt_list)
    te_fast(pose_model, ov_pose_path, data2, txt_list, 'handpose')
    add_log('hand_model(squeezenet):', txt_list)
    te_fast(hand_model, ov_hand_path, data1, txt_list, 'handpose')

    path = '../log/pth-VS-ov_handpose.txt'
    content = ''
    for txt in txt_list:
        content += txt
    with open(path, 'w+', encoding='utf8') as f:
        f.write(content)
    print(f'IR model saved in openvino_model ,log file saved in {path}')
