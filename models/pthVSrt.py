def te_fast(model, trt_path, data, txt_list):
    from torch2trt import TRTModule
    import time
    from _utils.myutils import add_log
    import torch
    import numpy as np
    model_trt1 = TRTModule()
    model_trt1.load_state_dict(torch.load(trt_path))
    start_time = time.time()
    with torch.no_grad():
        for i in range(1000):
            out1 = model(data)
        end_time = time.time()
        add_log(f'    Torch cost {round((end_time - start_time), 2)}s', txt_list)
        start_time2 = time.time()
        data = data.half()
        for i in range(1000):
            out2 = model_trt1(data)

        # print(out1)
        # print(out2)
        end_time2 = time.time()
        add_log(f'    TensorRT cost {round((end_time2 - start_time2), 2)}s', txt_list)
        err = np.sum(np.abs(out2.cpu().numpy() - out1.cpu().numpy()))
        rou = round((end_time - start_time) / (end_time2 - start_time2), 1)
        add_log(f'    TensorRT is {rou} times as Torch', txt_list)
        add_log(f'    err={err}', txt_list)


def te_sr():
    import time
    import torch
    import json
    from PIL import Image
    from torchvision import transforms
    from common.handpose.models.resnet import resnet50
    from _utils.myutils import make_model
    from common.handpose.models.squeezenet import squeezenet1_1
    from common.mico.backbone import MicroNet
    from common.mico.utils.defaults import _C as cfg
    from _utils.myutils import add_log
    with open("../log/class_id.json", 'r', encoding='UTF-8') as f:
        class_dict = json.load(f)
    class_dict = {int(k): class_dict[k] for k in class_dict.keys()}
    n_label = len(list(class_dict.keys()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = Image.open('test.png')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(img)
    img = img.unsqueeze_(0)
    data1 = img.to(device)

    txt_list = []
    t = time.strftime('%Y-%m-%d %H:%M', time.localtime())
    add_log(f'time:{t}', txt_list)
    hand_model_path = '../handpose/weights/squeezenet1_1-size-256-loss-0.0732.pth'
    hand_model = squeezenet1_1(num_classes=42)
    chkpt = torch.load(hand_model_path, map_location=device)
    hand_model.load_state_dict(chkpt)
    hand_model.to(device).eval()
    tr_hand_path = 'tensorrt_model/tr-hand_model-squ.pth'
    add_log('hand_model(resnet50):', txt_list)
    te_fast(hand_model, tr_hand_path, data1, txt_list)

    hand_model_path = '../handpose/weights/resnet_50-size-256-loss-0.0642.pth'
    hand_model = resnet50(num_classes=42, img_size=256)
    chkpt = torch.load(hand_model_path, map_location=device)
    hand_model.load_state_dict(chkpt)
    hand_model.to(device).eval()
    tr_hand_path = 'tensorrt_model/tr-hand_model-res50.pth'
    add_log('hand_model(resnet50):', txt_list)
    te_fast(hand_model, tr_hand_path, data1, txt_list)

    path = 'pth/model-micronet_m3.pth'
    classify_model = MicroNet(cfg, num_classes=n_label)
    classify_model.load_state_dict(torch.load(path, map_location=device), strict=False)
    classify_model.to(device).eval()
    add_log('classify_model(micronet_m3):', txt_list)
    te_fast(classify_model, 'tensorrt_model/tr-micronet_m3.pth', data1, txt_list)

    path = 'pth/model-googlenet.pth'
    classify_model = make_model('googlenet', n_label, path, device)
    classify_model.load_state_dict(torch.load(path, map_location=device), strict=False)
    classify_model.to(device).eval()
    add_log('classify_model(googlenet):', txt_list)
    te_fast(classify_model, 'tensorrt_model/tr-googlenet.pth', data1, txt_list)

    path = 'pth/model-resnet18.pth'
    classify_model = make_model('resnet18', n_label, path, device)
    classify_model.load_state_dict(torch.load(path, map_location=device), strict=False)
    classify_model.to(device).eval()
    add_log('classify_model(resnet18):', txt_list)
    te_fast(classify_model, 'tensorrt_model/tr-resnet18.pth', data1, txt_list)

    path = '../log/pth-VS-tr.txt'
    content = ''
    for txt in txt_list:
        content += txt
    with open(path, 'w+', encoding='utf8') as f:
        f.write(content)


if __name__ == '__main__':
    te_sr()
