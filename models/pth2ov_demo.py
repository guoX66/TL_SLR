import argparse
import os
import shutil
from pathlib import Path
import sys

sys.path.append("..")
import numpy as np
import torch
import json
from common.mico.backbone import MicroNet
from common.mico.utils.defaults import _C as cfg
from _utils.myutils import make_model
from _utils.configs import read_cfg, ModelInfo, LSTM, TrainImg
from pthVSov import te_fast

if __name__ == '__main__':
    modelInfo = ModelInfo()
    Train = TrainImg()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(base_dir)
    tr_Cfg, Ir_Cfg, ba_Cfg, ini_Cfg = read_cfg(base_dir)
    s1, s2 = modelInfo.size
    m1, m2, m3 = modelInfo.ms[0][0], modelInfo.ms[0][1], modelInfo.ms[0][2]
    st1, st2, st3 = modelInfo.ms[1][0], modelInfo.ms[1][1], modelInfo.ms[1][2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LSTM')
    args = parser.parse_args()
    with open("../log/class_id.json", 'r', encoding='UTF-8') as f:
        class_dict = json.load(f)
    class_dict = {int(k): class_dict[k] for k in class_dict.keys()}
    n_label = len(list(class_dict.keys()))
    device = torch.device('cpu')
    env_path = ba_Cfg['env_path']
    import site

    envs = site.getsitepackages()
    for i in envs:
        if 'site-packages' in i:
            openvino_path = i
            break

    path = f'pth/model-{args.model}.pth'
    save_path = f"openvino_model/{args.model}.xml"
    sample_input = torch.rand((1, 3, s1, s2))
    if args.model == 'micronet_m3':
        model = MicroNet(cfg, num_classes=n_label)
        model.load_state_dict(torch.load(path, map_location=device), strict=False)
    elif args.model == "LSTM":
        model = LSTM(128, 1, n_label, Train.batch_size, 20, device)
        model.load_state_dict(torch.load(path, map_location=device), strict=False)
        sample_input = torch.rand((1, 20, 48, 2))
    else:
        model = make_model(args.model, n_label, path, device)

    model.eval()
    model.to(device)
    sample_input = sample_input.to(device)
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
    rd = np.random.RandomState(888)
    py_sys = f'python {openvino_path}/openvino/tools/mo/mo_onnx.py'
    if args.model == "LSTM":
        sys = f'{py_sys} --input_model onnx_model/{args.model}.onnx  --output_dir ./openvino_model --input_shape "[1,20,48,2]" --compress_to_fp16=True'
        data = rd.random((1, 20, 48, 2))
    else:
        sys = f'{py_sys} --input_model onnx_model/{args.model}.onnx  --output_dir ./openvino_model --input_shape "[1,3,{s1},{s2}]" --mean_values "[{255 * m1}, {255 * m2} , {255 * m3}]" --scale_values "[{255 * st1}, {255 * st1} , {255 * st3}]" --compress_to_fp16=True'
        data = rd.random((1, 3, s1, s2))
    os.system(sys)

    txt_list = []
    te_fast(model, save_path, data, txt_list)
    txt_path = '../log/pth-VS-OV_class.txt'
    content = ''
    for txt in txt_list:
        content += txt
    with open(txt_path, 'w+', encoding='utf8') as f:
        f.write(content)
    print()
    print(f'IR model saved in {save_path} ,log file saved in log/pth-VS-OV_class.txt')
