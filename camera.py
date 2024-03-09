from _utils.detect import Multiprocess, video_start
from _utils.myutils import *
from multiprocessing import Process, Queue
from _utils.configs import read_cfg, LSTM
from common.movenet.models.model_factory import load_model
from common.handpose.models.squeezenet import squeezenet1_1
from common.mico.backbone import MicroNet
from common.mico.utils.defaults import _C as cfg


def main(camera, device):
    with open("log/class_id.json", 'r', encoding='UTF-8') as f:
        class_dict = json.load(f)
    class_dict = {int(k): class_dict[k] for k in class_dict.keys()}
    n_label = len(list(class_dict.keys()))

    c_model = Ir_Cfg['model']
    if c_model == 'LSTM':
        LSTM_model = True
    else:
        LSTM_model = False

    if device == "CUDA":
        p_pose = load_model("movenet_lightning", ft_size=48)
        path_hand = 'models/pth/squeezenet1_1-size-256-loss-0.0732.pth'
        p_hand = squeezenet1_1(num_classes=42)
        chkpt = torch.load(path_hand)
        p_hand.load_state_dict(chkpt)
        class_path = f'models/pth/model-{c_model}.pth'
        if c_model == 'micronet_m3':
            p_class = MicroNet(cfg, num_classes=n_label)
            p_class.load_state_dict(torch.load(class_path), strict=False)
        elif c_model == 'LSTM':
            from _utils.configs import TrainImg
            Train = TrainImg()
            p_class = LSTM(128, 1, n_label, Train.batch_size, 20, torch.device('cuda'))
            p_class.load_state_dict(torch.load(class_path))
        else:
            p_class = make_model(c_model, n_label, class_path, torch.device('cuda'))

    elif device == "TensorRt":
        p_class = f'models/tensorrt_model/tr-{c_model}.pth'
        p_hand = 'models/tensorrt_model/tr-hand_model-squ.pth'
        p_pose = load_model("movenet_lightning", ft_size=48)
    else:
        if device == 'MYRIAD':
            p_pose = 'models/openvino_model/human-pose-estimation-0005.xml'
        else:
            p_pose = 'models/openvino_model/pose_model.xml'

        p_hand = 'models/openvino_model/hand_model.xml'
        p_class = f'models/openvino_model/{c_model}.xml'  # resnet18

    iq = Queue(4)
    oq = Queue(4)
    sq = Queue(1)

    # p1 = Process(target=Multiprocess,
    #              args=(iq, oq, path_pose, path_hand, path1, class_dict,))
    #
    p2 = Process(target=Multiprocess,
                 args=(iq, oq, p_pose, p_hand, p_class, class_dict, sq, device, LSTM_model))

    # p1.daemon = True
    # p1.start()
    p2.daemon = True
    p2.start()
    video_start(camera, iq, oq, sq)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tr_Cfg, Ir_Cfg, ba_Cfg, _ = read_cfg(base_dir)
    main(Ir_Cfg['source'], ba_Cfg['device'])
