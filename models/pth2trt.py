import shutil
import sys

from handpose.models.resnet import resnet50

sys.path.append("..")
from _utils.myutils import *
import torch2trt
from handpose.models.squeezenet import squeezenet1_1
from mico.backbone import MicroNet
from mico.utils.defaults import _C as cfg
from multiprocessing import Process
from pthVSrt import te_sr


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

    model_list = ['hand_model(squeezenet1_1)',    # 0
                  'hand_model(resnet_50)',        # 1
                  'classify_model(micronet_m3)',  # 2
                  'classify_model(googlenet)',    # 3
                  'classify_model(resnet18)']     # 4

    m_count = 2
    fp16 = True
    with open("../log/class_id.json", 'r', encoding='UTF-8') as f:
        class_dict = json.load(f)
    class_dict = {int(k): class_dict[k] for k in class_dict.keys()}
    n_label = len(list(class_dict.keys()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data1 = torch.ones((1, 3, 256, 256)).to(device)
    model_dict = {}
    hand_model_path = '../handpose/weights/squeezenet1_1-size-256-loss-0.0732.pth'
    hand_model = squeezenet1_1(num_classes=42)
    chkpt = torch.load(hand_model_path, map_location=device)
    hand_model.load_state_dict(chkpt)
    hand_model.to(device).eval()
    if fp16:
        hand_model.half()
        data1 = data1.half()
    model_dict['hand_model(squeezenet1_1)'] = ("tensorrt_model/tr-hand_model-squ.pth", hand_model)

    hand_model_path = '../handpose/weights/resnet_50-size-256-loss-0.0642.pth'
    hand_model = resnet50(num_classes=42, img_size=256)
    chkpt = torch.load(hand_model_path, map_location=device)
    hand_model.load_state_dict(chkpt)
    hand_model.to(device)
    hand_model.eval()
    if fp16:
        hand_model.half()
    model_dict['hand_model(resnet_50)'] = ("tensorrt_model/tr-hand_model-res50.pth", hand_model)

    path = 'pth/model-micronet_m3.pth'
    classify_model = MicroNet(cfg, num_classes=n_label)
    classify_model.load_state_dict(torch.load(path, map_location=device), strict=False)
    classify_model.to(device)
    classify_model.eval()
    if fp16:
        classify_model.half()
    model_dict['classify_model(micronet_m3)'] = ("tensorrt_model/tr-micronet_m3.pth", classify_model)

    path = 'pth/model-googlenet.pth'
    classify_model = make_model('googlenet', n_label, path, device)
    classify_model.to(device)
    classify_model.eval()
    if fp16:
        classify_model.half()
    model_dict['classify_model(googlenet)'] = ("tensorrt_model/tr-googlenet.pth", classify_model)

    path = 'pth/model-resnet18.pth'
    classify_model = make_model('resnet18', n_label, path, device)
    classify_model.to(device)
    classify_model.eval()
    if fp16:
        classify_model.half()
    model_dict['classify_model(resnet18)'] = ("tensorrt_model/tr-resnet18.pth", classify_model)

    m_key = model_list[m_count]
    path, model = model_dict[m_key]
    save_rt(model, data1, path, fp16)
    print(f'{m_key} saved')
    if m_count > 1:
        shutil.copy(path, "tensorrt_model/tr-classify_model.pth")
        txt_list = []
        t = time.strftime('%Y-%m-%d %H:%M', time.localtime())

        add_log(f'time:{t}', txt_list)
        add_log(f'model:{m_key}', txt_list)
        add_log(f'fp16:{fp16}', txt_list)
        path = '../log/pth2tr.txt'
        content = ''
        for txt in txt_list:
            content += txt
        with open(path, 'w+', encoding='utf8') as f:
            f.write(content)
