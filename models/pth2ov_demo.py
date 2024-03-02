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

if __name__ == '__main__':
    with open("../log/class_id.json", 'r', encoding='UTF-8') as f:
        class_dict = json.load(f)
    class_dict = {int(k): class_dict[k] for k in class_dict.keys()}
    n_label = len(list(class_dict.keys()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    openvino_path = 'D:\code\python3.10\Lib\site-packages'
    path = 'pth/classify_model.pth'
    model = MicroNet(cfg, num_classes=n_label)
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    model.eval()
    sample_input = torch.rand((1, 3, 256, 256))

    torch.onnx.export(
        model,
        sample_input,  # Input tensor
        f'onnx_model/classify_model.onnx',  # Output file (eg. 'output_model.onnx')
        # opset_version=12,       # Operator support version
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],  # Input tensor name (arbitary)
        output_names=['output']  # Output tensor name (arbitary)

    )
    py_sys = f'python {openvino_path}/openvino/tools/mo/mo_onnx.py'
    sys = f'{py_sys} --input_model onnx_model/classify_model.onnx  --output_dir ./openvino_model --input_shape "[1,3,256,256]" --mean_values "[123.675, 116.28 , 103.53]" --scale_values "[58.395, 57.12 , 57.375]" --compress_to_fp16=True'
    os.system(sys)
