import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import json
from common.mico.backbone import MicroNet
from common.mico.utils.defaults import _C as cfg
from _utils.myutils import make_model
from _utils.configs import ba_Cfg, tr_Cfg

from pthVSov import te_fast

if __name__ == '__main__':
    s1, s2 = int(ba_Cfg['size'].split('x')[0]), int(ba_Cfg['size'].split('x')[1])
    m1, m2, m3 = tr_Cfg['Normalized_matrix'][0][0], tr_Cfg['Normalized_matrix'][0][1], tr_Cfg['Normalized_matrix'][0][2]
    st1, st2, st3 = tr_Cfg['Normalized_matrix'][1][0], tr_Cfg['Normalized_matrix'][1][1], \
        tr_Cfg['Normalized_matrix'][1][2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    args = parser.parse_args()
    with open("../log/class_id.json", 'r', encoding='UTF-8') as f:
        class_dict = json.load(f)
    class_dict = {int(k): class_dict[k] for k in class_dict.keys()}
    n_label = len(list(class_dict.keys()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    openvino_path = ba_Cfg['env_path']
    path = f'pth/model-{args.model}.pth'
    save_path = f"openvino_model/{args.model}.xml"
    if args.model == 'micronet_m3':
        model = MicroNet(cfg, num_classes=n_label)
        model.load_state_dict(torch.load(path, map_location=device), strict=False)
    else:
        model = make_model(args.model, n_label, path, device)

    model.eval()
    sample_input = torch.rand((1, 3, s1, s2))
    torch.onnx.export(
        model,
        sample_input,  # Input tensor
        f'onnx_model/{args.model}.onnx',  # Output file (eg. 'output_model.onnx')
        # opset_version=12,       # Operator support version
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],  # Input tensor name (arbitary)
        output_names=['output']  # Output tensor name (arbitary)

    )

    py_sys = f'python {openvino_path}/site-packages/openvino/tools/mo/mo_onnx.py'
    sys = f'{py_sys} --input_model onnx_model/{args.model}.onnx  --output_dir ./openvino_model --input_shape "[1,3,{s1},{s2}]" --mean_values "[{255 * m1}, {255 * m2} , {255 * m3}]" --scale_values "[{255 * st1}, {255 * st1} , {255 * st3}]" --compress_to_fp16=True'
    os.system(sys)

    rd = np.random.RandomState(888)
    data = rd.random((s1, s2, 3))  # 随机生成一个 [0,1) 的浮点数 ，5x5的矩阵
    txt_list = []
    te_fast(model, save_path, data, txt_list)
    txt_path = '../log/pth-VS-OV.txt'
    content = ''
    for txt in txt_list:
        content += txt
    with open(txt_path, 'w+', encoding='utf8') as f:
        f.write(content)
    print()
    print(f'IR model saved in {save_path} ,log file saved in log/pth-VS-tr.txt')
