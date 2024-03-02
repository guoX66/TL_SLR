import argparse
import os
import shutil
import torch
import json
import sys

sys.path.append("..")
from movenet.models.model_factory import load_model
from handpose.models.squeezenet import squeezenet1_1
from mico.backbone import MicroNet
from mico.utils.defaults import _C as cfg
from _utils.myutils import make_model, remove_file
from pthVSov import te_ov
from pathlib import Path
from _utils.configs import Cfg


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

    openvino_path = Cfg['python_path']

    classify_model1 = MicroNet(cfg, num_classes=n_label)
    classify_model1.load_state_dict(torch.load(path1, map_location=device), strict=False)
    classify_model1.eval()

    # classify_model2 = make_model('mobilenet_v3', n_label, path2, device)
    # classify_model2.load_state_dict(torch.load(path2, map_location=device), strict=False)
    # classify_model2.eval()

    pose_model = load_model("movenet_lightning", ft_size=48)
    pose_model.eval()

    model_path = '../handpose/weights/squeezenet1_1-size-256-loss-0.0732.pth'
    hand_model = squeezenet1_1(num_classes=42)
    chkpt = torch.load(model_path, map_location=device)
    hand_model.load_state_dict(chkpt)
    hand_model.eval()
    # model_list = [classify_model1, classify_model2, pose_model, hand_model]

    model_dict = {
        'micronet_m3': [classify_model1, torch.rand((1, 3, 256, 256))],
        # 'mobilenet_v3': [classify_model2, torch.rand((1, 3, 256, 256))],
        'classify_model': [classify_model1, torch.rand((1, 3, 256, 256))],
        'pose_model': [pose_model, torch.rand((1, 192, 192, 3))],
        'hand_model': [hand_model, torch.rand((1, 3, 256, 256))]}

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
        # onnx_model = lm(f'onnx_model/{model_name}.onnx')
        # trans_model = float16_converter.convert_float_to_float16(onnx_model, keep_io_types=True)
        # sm(trans_model, f'onnx_model/{model_name}.onnx')
    # try:
    #     shutil.rmtree('openvino_model')
    # except:
    #     pass
    sys_com_micronet_m3 = f'python {openvino_path}/Lib/site-packages/openvino/tools/mo/mo_onnx.py --input_model onnx_model/micronet_m3.onnx  --output_dir ./openvino_model --input_shape "[1,3,256,256]" --compress_to_fp16=True '
    sys_com_mobilenet_v3 = f'python {openvino_path}/Lib/site-packages/openvino/tools/mo/mo_onnx.py --input_model onnx_model/mobilenet_v3.onnx  --output_dir ./openvino_model --input_shape "[1,3,256,256]" --compress_to_fp16=True'
    sys_com_classify_model = f'python {openvino_path}/Lib/site-packages/openvino/tools/mo/mo_onnx.py --input_model onnx_model/classify_model.onnx  --output_dir ./openvino_model --input_shape "[1,3,256,256]" --mean_values "[123.675, 116.28 , 103.53]" --scale_values "[58.395, 57.12 , 57.375]" --compress_to_fp16=True'
    sys_com_pose = f'python {openvino_path}/Lib/site-packages/openvino/tools/mo/mo_onnx.py --input_model onnx_model/pose_model.onnx  --output_dir ./openvino_model --input_shape "[1,192,192,3]" --compress_to_fp16=True'
    sys_com_hand = f'python {openvino_path}/Lib/site-packages/openvino/tools/mo/mo_onnx.py --input_model onnx_model/hand_model.onnx  --output_dir ./openvino_model --input_shape "[1,3,256,256]" --compress_to_fp16=True'

    os.system(sys_com_micronet_m3)
    os.system(sys_com_mobilenet_v3)
    os.system(sys_com_classify_model)  # micronet_m3
    os.system(sys_com_pose)
    os.system(sys_com_hand)

    # remove_file('temp_ov', 'openvino_model')
    te_ov()
