from _utils.detect import Multiprocess, OV_video_start
from _utils.myutils import *
from multiprocessing import Process, Queue
from _utils.configs import Ir_Cfg, tr_Cfg, Cfg
from common.movenet.models.model_factory import load_model
from common.handpose.models.squeezenet import squeezenet1_1
from common.mico.backbone import MicroNet
from common.mico.utils.defaults import _C as cfg


def main(camera, pf, mode, device):
    with open("log/class_id.json", 'r', encoding='UTF-8') as f:
        class_dict = json.load(f)
    class_dict = {int(k): class_dict[k] for k in class_dict.keys()}
    n_label = len(list(class_dict.keys()))
    c_model = Ir_Cfg['model']
    if device == "NVIDIA pytorch":
        p_pose = load_model("movenet_lightning", ft_size=48)
        path_hand = 'models/pth/squeezenet1_1-size-256-loss-0.0732.pth'
        p_hand = squeezenet1_1(num_classes=42)
        chkpt = torch.load(path_hand)
        p_hand.load_state_dict(chkpt)
        class_path = f'models/pth/model-{c_model}.pth'
        if c_model == 'micronet_m3':
            p_class = MicroNet(cfg, num_classes=n_label)
            p_class.load_state_dict(torch.load(class_path), strict=False)
        else:
            p_class = make_model(c_model, n_label, class_path, torch.device('cuda'))

    elif device == "NVIDIA tensorrt":
        p_class = f'models/tensorrt_model/tr-{c_model}.pth'
        p_hand = 'models/tensorrt_model/tr-hand_model-squ.pth'
        p_pose = load_model("movenet_lightning", ft_size=48)
    else:
        if mode == 'mv':
            p_pose = 'models/openvino_model/pose_model.xml'
        elif mode == 'ov':
            p_pose = 'models/openvino_model/human-pose-estimation-0005.xml'
        else:
            raise ValueError('pose net must be "mv" or "ov"')

        p_hand = 'models/openvino_model/hand_model.xml'
        p_class = f'models/openvino_model/{c_model}.xml'  # resnet18

    iq = Queue(4)
    oq = Queue(4)
    sq = Queue(1)

    # p1 = Process(target=Multiprocess,
    #              args=(iq, oq, path_pose, path_hand, path1, class_dict,))
    #
    p2 = Process(target=Multiprocess,
                 args=(iq, oq, p_pose, p_hand, p_class, class_dict, sq, mode, device))

    # p1.daemon = True
    # p1.start()
    p2.daemon = True
    p2.start()
    OV_video_start(camera, iq, oq, sq, mode=pf)


if __name__ == '__main__':
    main(Ir_Cfg['source'], Ir_Cfg['platform'], Ir_Cfg['pose_net'], Cfg['device'])
