# -*- coding: UTF-8 -*-
import argparse
import socket
from _utils.detect import get_pose_mean, classify
from _utils.myutils import *
from torchvision import transforms
from common.movenet.models.model_factory import load_model
from multiprocessing import Process, Queue
from _utils.configs import read_cfg, LSTM
from common.handpose.models.squeezenet import squeezenet1_1
from common.mico.backbone import MicroNet
from common.mico.utils.defaults import _C as cfg


def ser_Multiprocess(client_socket1, iq, p_pose, p_hand, p_class, device, class_dict, LSTM_model):
    if device == "CUDA":
        pose_model = p_pose
        hand_model = p_hand
        classify_model = p_class
        pose_model.to(torch.device('cuda'))
        hand_model.to(torch.device('cuda'))
        classify_model.to(torch.device('cuda'))
    elif device == "TensorRt":
        from torch2trt import TRTModule
        hand_model = TRTModule()
        classify_model = TRTModule()
        pose_model, hand_path, class_path = p_pose, p_hand, p_class
        pose_model.to(torch.device('cuda'))
        hand_model.load_state_dict(torch.load(f'{hand_path}'))
        classify_model.load_state_dict(torch.load(f'{class_path}'))
    else:
        raise ValueError(" The device of server must be CUDA or TensorRt")
    count = 0
    seq_len = 10
    input_data = []

    while True:
        if not iq.empty():
            img = iq.get()
            img = cv2.resize(img, [512, 512], interpolation=cv2.INTER_LINEAR)
            st = time.perf_counter()

            if not LSTM_model:
                result = get_pose_mean(img, pose_model, hand_model, classify_model, class_dict, device,
                                       LSTM_model)
            else:
                feature = get_pose_mean(img, pose_model, hand_model, classify_model, class_dict, device,
                                        LSTM_model)
                if feature is not None:
                    input_data.append(feature)
                    count += 1
                if count >= seq_len:
                    input_np = np.stack(input_data, axis=0)
                    result = classify(input_np, classify_model, class_dict, device, LSTM_model)
                    if count > seq_len:
                        input_data.pop(0)
                        input_np = np.stack(input_data, axis=0)
                        result = classify(input_np, classify_model, class_dict, device, LSTM_model)
                else:
                    result = None

            et = time.perf_counter()

            if result is None:
                result = 'None'

            fps = str(round(1 / (et - st), 1))
            res = [result, fps]
            result = json.dumps(res)
            client_socket1.send(bytes(result.encode('utf-8')))


def get_img(client_socket2, iq):
    while True:
        # 接收标志数据
        # try:
        data = client_socket2.recv(1024)
        if data.decode() == 'finish':
            client_socket2.close()
            break
        if data:
            # 通知客户端“已收到标志数据，可以发送图像数据”
            client_socket2.send(b"start")
            # 处理标志数据

            flag = data.decode().split(",")

            # 图像字节流数据的总长度
            if flag[0] != 'flag':
                continue
            total = int(flag[1])
            # 接收到的数据计数
            cnt = 0
            # 存放接收到的数据
            img_bytes = b""
            while cnt < total:
                # 当接收到的数据少于数据总长度时，则循环接收图像数据，直到接收完毕
                data = client_socket2.recv(256000)
                img_bytes += data
                cnt += len(data)
            # 通知客户端“已经接收完毕，可以开始下一帧图像的传输”
            client_socket2.send(b"end")
            # 解析接收到的字节流数据，并显示图像
            img = np.asarray(bytearray(img_bytes), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            cv2.imshow('demo_ser', img)
            cv2.waitKey(1)
            if not iq.full():
                iq.put(img)
        # except:
        #     client_socket2.close()
        #     break


def main(port, iq_num):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tr_Cfg, Ir_Cfg, ba_Cfg, _ = read_cfg(base_dir)
    with open("log/class_id.json", 'r', encoding='UTF-8') as f:
        class_dict = json.load(f)
    class_dict = {int(k): class_dict[k] for k in class_dict.keys()}
    n_label = len(list(class_dict.keys()))
    c_model = Ir_Cfg['model']
    device = ba_Cfg['device']
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
        raise ValueError(" The device of server must be CUDA or TensorRt")
    iq = Queue(iq_num)
    HOST = ''
    PORT = port
    ADDRESS = (HOST, PORT)
    # 创建一个套接字
    tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpServer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
    # 绑定本地ip
    tcpServer.bind(ADDRESS)
    # 开始监听
    tcpServer.listen(5)
    print(f'端口{ADDRESS} 正在等待连接')
    while True:
        num = 0
        while num != 2:
            client_socket, client_address1 = tcpServer.accept()
            mode = client_socket.recv(1024)
            if mode.decode() == 'img':
                print(f"图片通道连接{client_address1}成功！")
                p1 = Process(target=get_img,
                             args=(client_socket, iq,))
                num += 1
            elif mode.decode() == 'res':
                print(f"结果通道连接{client_address1}成功！")
                p2 = Process(target=ser_Multiprocess,
                             args=(
                                 client_socket, iq, p_pose, p_hand, p_class, device, class_dict, LSTM_model))
                num += 1

        p1.start()
        # p2.daemon = True
        p2.start()
        p1.join()
        # p2.join()
        p2.terminate()
        print('本次连接结束')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8800)
    parser.add_argument('--iq_num', type=int, default=10)
    args = parser.parse_args()
    main(args.port, args.iq_num)
