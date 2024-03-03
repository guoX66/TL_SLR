import argparse
import os
import shutil
import torch
import json
import sys
sys.path.append("..")
from common.movenet.models.model_factory import load_model
from common.handpose.models.squeezenet import squeezenet1_1
from common.mico.backbone import MicroNet
from common.mico.utils.defaults import _C as cfg
from _utils.myutils import make_model, remove_file
from pthVSov import te_ov
from pathlib import Path


if __name__ == '__main__':
    # try:
    #     shutil.rmtree('onnx_model')
    # except:
    #     pass
    # try:
    #     shutil.rmtree('openvino_model')
    # except:
    #     pass
    # os.mkdir('onnx_model')

    with open("../log/class_id.json", 'r', encoding='UTF-8') as f:
        class_dict = json.load(f)
    class_dict = {int(k): class_dict[k] for k in class_dict.keys()}
    n_label = len(list(class_dict.keys()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path1 = 'pth/model-micronet_m3.pth'
    path2 = 'pth/model-mobilenet_v3.pth'

    import site
    envs = site.getsitepackages()
    print(envs)
    for i in envs:
        if 'site-packages' in i:
            openvino_path = i
            break

    pose_model = load_model("movenet_lightning", ft_size=48,model_dir='pth')
    pose_model.to(device)
    pose_model.eval()

    model_path = 'pth/squeezenet1_1-size-256-loss-0.0732.pth'
    hand_model = squeezenet1_1(num_classes=42)
    chkpt = torch.load(model_path, map_location=device)
    hand_model.load_state_dict(chkpt)
    hand_model.to(device)
    hand_model.eval()
    # model_list = [classify_model1, classify_model2, pose_model, hand_model]

    model_dict = {
        'pose_model': [pose_model, torch.rand((1, 192, 192, 3)).to(device)],
        'hand_model': [hand_model, torch.rand((1, 3, 256, 256)).to(device)]}

    for model_name in model_dict.keys():
        model = model_dict[model_name][0]
        sample_input = model_dict[model_name][1]
        torch.onnx.export(
            model,
            sample_input,  # Input tensor
            f'onnx_model/{model_name}.onnx',  # Output file (eg. 'output_model.onnx')
            # opset_version=12,       # Operator support version
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],  # Input tensor name (arbitary)
            output_names=['output']  # Output tensor name (arbitary)

        )

    sys_com_pose = f'python {openvino_path}/openvino/tools/mo/mo_onnx.py --input_model onnx_model/pose_model.onnx  --output_dir ./openvino_model --input_shape "[1,192,192,3]" --compress_to_fp16=True'
    sys_com_hand = f'python {openvino_path}/openvino/tools/mo/mo_onnx.py --input_model onnx_model/hand_model.onnx  --output_dir ./openvino_model --input_shape "[1,3,256,256]" --compress_to_fp16=True'
    os.system(sys_com_pose)
    os.system(sys_com_hand)
    te_ov(pose_model,hand_model)
