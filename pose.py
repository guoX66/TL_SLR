import numpy as np
import torch
from common.movenet.models.model_factory import load_model
from common.movenet.moveutils import read_imgfile
import shutil
import cv2
from _utils.myutils import bar, crop
import time
import traceback
import argparse
from common.handpose.models.squeezenet import squeezenet1_1
from _utils.detect import draw_skel_and_kp, view_mode, my_convert_model, inference_, is_show, is_write
from _utils.configs import *


def get_label_list(foldname):
    file_path = f'./{foldname}'

    path_list = []

    for i in os.walk(file_path):
        path_list.append(i)

    label_dict = dict()
    label_name_list = []
    label_list = []

    for i in range(len(path_list[0][1])):
        label = path_list[0][1][i]
        label_dict[label] = path_list[i + 1][2]

    for i in label_dict.keys():
        label_list.append(i)
        for j in label_dict[i]:
            label_name_list.append([i, j])

    return label_name_list, label_dict, label_list


def make_data(foldname, imgpath, model, hand_model, device, LSTM_model=False):
    file_list, label_dict, _ = get_label_list(foldname)
    try:
        shutil.rmtree(imgpath)
    except FileNotFoundError:
        pass
    for i in label_dict.keys():
        if not LSTM_model:
            label = i.split('-')[0]
            os.makedirs(f'{imgpath}/{label}', exist_ok=True)
        else:
            os.makedirs(f'{imgpath}/{i}', exist_ok=True)
    start_time = time.perf_counter()
    if not LSTM_model:
        print('正在提取姿态图......')
    else:
        print('正在提取特征......')
    for j, i in enumerate(file_list):
        label, file = i[0], i[1]
        single_make(foldname, label, file, imgpath, model, hand_model, device, LSTM_model)
        bar(j + 1, len(file_list), start=start_time, des='提取进度', train=False)

    print('转换完成！')


def single_make(foldname, label, filename, imgpath, pose_model, hand_model, device, LSTM_model):
    size = 192
    cam_width = 1280
    cam_height = 720
    conf_thres = 0.3

    p_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.cuda()
    if not LSTM_model:
        new_label = label.split('-')[0]
    else:
        new_label = label
    path = f"./{foldname}/{label}/{filename}"
    new_path = f"./{imgpath}/{new_label}/{filename}"
    if os.path.exists(new_path):
        new_path = f"./{imgpath}/{new_label}/1{filename}"
    input_image, draw_image = read_imgfile(path, size)
    if device == "CUDA" or device == "TensorRt":
        pose_model.eval()
        with torch.no_grad():
            input_image = torch.Tensor(input_image)
            input_image = input_image.to(torch.device('cuda'))
            out = pose_model(input_image)
            if out is not None:
                kpt_with_conf = out[0, 0, :, :]
                kpt_with_conf = kpt_with_conf.cpu().numpy()
    else:
        results = inference_(pose_model, input_image[0])
        kpt_with_conf = results[0, 0, :, :]
    if not LSTM_model:
        draw_img(draw_image, kpt_with_conf, conf_thres, hand_model, new_path)
    else:
        make_npy(draw_image, kpt_with_conf, conf_thres, hand_model, imgpath, label, filename)


def make_npy(draw_image, kpt_with_conf, conf_thres, hand_model, imgpath, label, filename):
    dirStr, ext = os.path.splitext(filename)
    feature = draw_skel_and_kp(draw_image, kpt_with_conf, conf_thres, hand_model, device, LSTM_model=True)
    if feature is not None:
        np.save(f"./{imgpath}/{label}/{dirStr}", feature)


def draw_img(draw_image, kpt_with_conf, conf_thres, hand_model, new_path):
    draw_image = draw_skel_and_kp(draw_image, kpt_with_conf, conf_thres, hand_model, device)
    if draw_image is not None:
        draw_image = crop(draw_image, 15)
        if is_show:
            cv2.imshow('demo_ov', draw_image)
            cv2.waitKey(1)
        cv2.imencode('.jpg', draw_image)[1].tofile(new_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--device", type=str)
    args.add_argument("--view_mode", type=int)
    args = args.parse_args()
    try:
        import yaml

        base_dir = os.path.dirname(os.path.abspath(__file__))
        tr_Cfg, Ir_Cfg, ba_Cfg, ini_Cfg = read_cfg(base_dir)
        if args.device is not None:
            ini_Cfg['base']['device'] = args.device
        if args.view_mode is not None:
            ini_Cfg['base']['view_mode'] = args.view_mode
        with open('Cfg.yaml', "w", encoding="utf-8") as f:
            yaml.dump(ini_Cfg, f)

        tr_Cfg, Ir_Cfg, ba_Cfg, Cfg = read_cfg(base_dir)
        device = Cfg['base']['device']
        model = tr_Cfg['model']
        if model == 'LSTM':
            LSTM_model = True
        else:
            LSTM_model = False

        p_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Train = TrainImg()

        if device == "CUDA" or device == "TensorRt":
            p_pose = load_model("movenet_lightning", ft_size=48)
            p_pose.to(p_device)
            p_hand = squeezenet1_1(num_classes=42)
            chkpt = torch.load('models/pth/squeezenet1_1-size-256-loss-0.0732.pth')
            p_hand.load_state_dict(chkpt)
            p_hand.to(p_device)

        else:
            from openvino.runtime import Core
            ie = Core()
            p_pose = my_convert_model('models/openvino_model/pose_model.xml', ie, device='CPU')
            p_hand = my_convert_model('models/openvino_model/hand_model.xml', ie, device='CPU')
        if LSTM_model:
            make_data(Train.foldname, Train.npypath, p_pose, p_hand, device, LSTM_model=LSTM_model)
        else:
            make_data(Train.foldname, Train.imgpath, p_pose, p_hand, device, LSTM_model=LSTM_model)

    except Exception as e:
        print(e)
        try:
            os.mkdir('log')
        except:
            pass
        traceback.print_exc(file=open('log/pose_err_log.txt', 'w+'))
