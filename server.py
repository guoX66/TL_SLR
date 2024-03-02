# -*- coding: UTF-8 -*-
import argparse
import socket
from _utils.detect import get_pose_mean
from _utils.myutils import *
from torchvision import transforms
from _utils.configs import ModelInfo
from common.movenet.models.model_factory import load_model
from multiprocessing import Process, Queue
from torch2trt import TRTModule
from _utils.configs import tr_Cfg, Ir_Cfg


def ser_Multiprocess(client_socket1, iq, p_pose, p_hand, p_class, device, mode, class_dict):
    if device == "NVIDIA pytorch":
        pose_model = p_pose
        hand_model = p_hand
        classify_model = p_class
        pose_model.to(torch.device('cuda'))
        hand_model.to(torch.device('cuda'))
        classify_model.to(torch.device('cuda'))
    elif device == "NVIDIA tensorrt":
        from torch2trt import TRTModule
        hand_model = TRTModule()
        classify_model = TRTModule()
        pose_model, hand_path, class_path = p_pose, p_hand, p_class
        pose_model.to(torch.device('cuda'))
        hand_model.load_state_dict(torch.load(f'{hand_path}'))
        classify_model.load_state_dict(torch.load(f'{class_path}'))
    while True:
        if not iq.empty():
            img = iq.get()
            img = cv2.resize(img, [512, 512], interpolation=cv2.INTER_LINEAR)
            st = time.perf_counter()
            result = get_pose_mean(img, pose_model, hand_model, classify_model, class_dict, mode, device)

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
            cv2.imshow('dehio', img)
            cv2.waitKey(1)
            if not iq.full():
                iq.put(img)
        # except:
        #     client_socket2.close()
        #     break


def main(port, iq_num):
    with open("log/class_id.json", 'r', encoding='UTF-8') as f:
        class_dict = json.load(f)
    class_dict = {int(k): class_dict[k] for k in class_dict.keys()}
    c_model = tr_Cfg['model']
    device = Ir_Cfg['device']
    p_class = f'models/tensorrt_model/tr-{c_model}.pth'  # resnet18
    p_hand = 'models/tensorrt_model/tr-hand_model-res50.pth'
    p_pose = load_model("movenet_lightning", ft_size=48)
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
                                 client_socket, iq, p_pose, p_hand, p_class, device, mode, class_dict))
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
