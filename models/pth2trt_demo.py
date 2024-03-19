import argparse
import sys
from common.mico.backbone import MicroNet
from common.mico.utils.defaults import _C as cfg
from _utils.myutils import *
import torch2trt
from pthVSrt import te_fast
import yaml
from _utils.myutils import make_model
from _utils.configs import read_cfg, ModelInfo, TrainImg, LSTM

parser = argparse.ArgumentParser()
parser.add_argument('--fp16', type=bool, default=True)
parser.add_argument('--model', type=str, default='LSTM')
args = parser.parse_args()


def save_rt(model, data, path, fp16):
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=fp16)
    torch.save(model_trt.state_dict(), path)


if __name__ == '__main__':
    # try:
    #     shutil.rmtree('onnx_model')
    # except:
    #     pass
    # os.mkdir('onnx_model')
    # try:
    #     shutil.rmtree('tensorrt_model')
    # except:
    #     pass
    #
    # os.mkdir('tensorrt_model')
    modelInfo = ModelInfo()
    Train = TrainImg()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(base_dir)
    tr_Cfg, Ir_Cfg, ba_Cfg, ini_Cfg = read_cfg(base_dir)
    s1, s2 = modelInfo.size
    m1, m2, m3 = modelInfo.ms[0][0], modelInfo.ms[0][1], modelInfo.ms[0][2]
    st1, st2, st3 = modelInfo.ms[1][0], modelInfo.ms[1][1], modelInfo.ms[1][2]
    fp16 = args.fp16
    with open("../log/class_id.json", 'r', encoding='UTF-8') as f:
        class_dict = json.load(f)
    class_dict = {int(k): class_dict[k] for k in class_dict.keys()}
    n_label = len(list(class_dict.keys()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = torch.ones((1, 3, modelInfo.size[0], modelInfo.size[1])).to(device)

    path = f'pth/model-{args.model}.pth'
    if args.model == 'micronet_m3':
        classify_model = MicroNet(cfg, num_classes=n_label)
        classify_model.load_state_dict(torch.load(path, map_location=device), strict=False)
    elif args.model == "LSTM":
        classify_model = LSTM(128, 1, n_label, 1, 20, device)
        classify_model.load_state_dict(torch.load(path, map_location=device), strict=False)
        data = torch.rand((1, 20, 48, 2)).to(device)
    else:
        classify_model = make_model(args.model, n_label, path, device)

    classify_model.to(device)
    classify_model.eval()

    # if fp16:
    #     data = data.half()
    #     classify_model.half()

    save_path = f"tensorrt_model/tr-{args.model}.pth"
    save_rt(classify_model, data, save_path, fp16)
    txt_list = []
    te_fast(classify_model, save_path, data, txt_list)
    txt_path = '../log/pth-VS-tr.txt'
    content = ''
    for txt in txt_list:
        content += txt
    with open(txt_path, 'w+', encoding='utf8') as f:
        f.write(content)
    print()
    print(f'RT model saved in {save_path} ,log file saved in log/pth-VS-tr.txt')
