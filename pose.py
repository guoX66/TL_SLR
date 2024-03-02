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


def make_img(foldname, imgpath, model, hand_model, device):
    print('正在提取姿态图......')
    file_list, label_dict, _ = get_label_list(foldname)
    try:
        shutil.rmtree(imgpath)
    except FileNotFoundError:
        pass
    for i in label_dict.keys():
        try:
            os.makedirs(f'{imgpath}/{i}')
        except:
            pass
    start_time = time.perf_counter()
    for j, i in enumerate(file_list):
        get_pose(foldname, i[0], i[1], imgpath, model, hand_model, device)
        bar(j + 1, len(file_list), start=start_time, des='提取进度', train=False)

    print('转换完成！')


def get_pose(foldname, label, filename, imgpath, pose_model, hand_model, device):
    size = 192
    cam_width = 1280
    cam_height = 720
    conf_thres = 0.3
    p_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.cuda()
    path = f"./{foldname}/{label}/{filename}"
    input_image, draw_image = read_imgfile(path, size)
    if device == "NVIDIA pytorch" or device == "NVIDIA tensorrt":
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

    draw_image = draw_skel_and_kp(draw_image, kpt_with_conf, conf_thres, hand_model, 'mv', device)
    if draw_image is not None:
        draw_image = crop(draw_image, 15)
        if is_show:
            cv2.imshow('demo_ov', draw_image)
            cv2.waitKey(1)
        cv2.imencode('.jpg', draw_image)[1].tofile(f"./{imgpath}/{label}/{filename}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--device", type=str, default='None')
    args.add_argument("--view_mode", type=str, default='None')
    args = args.parse_args()
    try:
        import yaml

        with open('Cfg.yaml', 'r', encoding='utf-8') as f:
            Cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        if args.device != 'None':
            Cfg['inference']['device'] = args.device
        if args.view_mode != 'None':
            Cfg['inference']['view_mode'] = int(args.view_mode)
        with open('Cfg.yaml', "w", encoding="utf-8") as f:
            yaml.dump(Cfg, f)

        from _utils.detect import draw_skel_and_kp, view_mode, my_convert_model, inference_, is_show, is_write
        from _utils.configs import *

        device = Ir_Cfg['device']
        mode = Ir_Cfg['pose_net']

        p_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Train = TrainImg()

        if device == "NVIDIA pytorch" or device == "NVIDIA tensorrt":
            p_pose = load_model("movenet_lightning", ft_size=48)
            p_pose.to(p_device)
            p_hand = squeezenet1_1(num_classes=42)
            chkpt = torch.load('models/pth/squeezenet1_1-size-256-loss-0.0732.pth')
            p_hand.load_state_dict(chkpt)
            p_hand.to(p_device)

        else:
            if mode == "mv":
                from openvino.runtime import Core

                ie = Core()
                p_pose = my_convert_model('models/openvino_model/pose_model.xml', ie, device='CPU')
            else:
                raise ValueError('pose net must be "mv"')
            p_hand = my_convert_model('models/openvino_model/hand_model.xml', ie, device)

        make_img(Train.foldname, Train.imgpath, p_pose, p_hand, Ir_Cfg['device'])

    except Exception as e:
        print(e)
        try:
            os.mkdir('log')
        except:
            pass
        traceback.print_exc(file=open('log/pose_err_log.txt', 'w+'))
