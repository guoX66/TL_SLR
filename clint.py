import json
import platform
import time
import cv2
from multiprocessing import Process, Queue
import socket
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
import matplotlib
from _utils.configs import Ir_Cfg,ba_Cfg


def cv2ImgAddText(img, text, left, top, textColor, textSize=20):
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(fm.findfont(fm.FontProperties(family='SimHei')), textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def web_send(iq, ADDRESS, wq, eq, sq):
    tcpClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 连接远程ip
    tcpClient.connect(ADDRESS)
    tcpClient.send(b"img")

    while True:
        if not iq.empty():
            cv_image = iq.get()
            cv_image = cv2.resize(cv_image, [256, 256], interpolation=cv2.INTER_LINEAR)

            # 压缩图像
            img_encode = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, 99])[1]
            # 转换为字节流
            bytedata = img_encode.tobytes()
            # # 标志数据，包括待发送的字节流长度等数据，用‘,’隔开
            flag_data = 'flag,'.encode() + (str(len(bytedata))).encode() + ",".encode() + " ".encode()
            tcpClient.send(flag_data)
            data = tcpClient.recv(1024)
            if ("start" == data.decode()):
                # 服务端已经收到标志数据，开始发送图像字节流数据
                # 接收服务端的应答
                tcpClient.send(bytedata)
            data = tcpClient.recv(1024)
            if "end" == data.decode():
                sq.put(1)
            break

    while True:
        if not eq.empty():
            tcpClient.send(b'finish')
            tcpClient.shutdown(socket.SHUT_RDWR)
            tcpClient.close()
            break

        # 读取图像
        if not iq.empty():

            cv_image = iq.get()
            cv_image = cv2.resize(cv_image, [256, 256], interpolation=cv2.INTER_LINEAR)

            # 压缩图像
            img_encode = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, 99])[1]
            # 转换为字节流
            bytedata = img_encode.tobytes()
            # # 标志数据，包括待发送的字节流长度等数据，用‘,’隔开
            flag_data = 'flag,'.encode() + (str(len(bytedata))).encode() + ",".encode() + " ".encode()
            tcpClient.send(flag_data)
            data = tcpClient.recv(1024)
            start = time.perf_counter()
            if ("start" == data.decode()):
                # 服务端已经收到标志数据，开始发送图像字节流数据
                # 接收服务端的应答
                tcpClient.send(bytedata)
            data = tcpClient.recv(1024)
            if "end" == data.decode():
                wt = (time.perf_counter() - start) * 1000
                if not wq.full():
                    wq.put(wt)
                # print("\r延时：{:.1f}ms".format(wt), end='')


def web_get(oq, ADDRESS, eq, sq):
    tcpClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 连接远程ip
    tcpClient.connect(ADDRESS)
    tcpClient.send(b"res")
    sq.put(1)
    while True:
        if not eq.empty():
            break

        data = tcpClient.recv(1024)
        if data:
            data = data.decode('utf-8')
            try:
                res = json.loads(data)
            except:
                print(data)
                res = ['None', 1]

            if not oq.full():
                oq.put(res)


def web_video_start(camera, iq, oq, wq, eq, mode, sq):
    if mode == 'rp' and camera == 0:
        from picamera2 import Picamera2
        cap = Picamera2()
        cap.configure(cap.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
        cap.start()

    elif camera == 0:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # 图像高度
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # 视频帧率
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(camera)
    if mode == 'rp' and camera == 0:
        img = cap.capture_array()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        ret, img = cap.read()
    iq.put(img)
    while True:
        if sq.full():
            break

    i = 0
    result_list = ['None', 'None']
    tmp_list = []
    fps_list = [0]
    wt_list = [0]
    word = 'None'
    s_time = time.perf_counter()
    s1_time = time.perf_counter()
    while True:
        if mode == 'rp' and camera == '0':
            img = cap.capture_array()
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        else:
            ret, img = cap.read()

        m, n, l = img.shape
        txt_m = int(0.8 * m)
        txt_n = int(0.5 * m)
        if not iq.full():
            iq.put(img)
        if not wq.empty():
            wt = wq.get()
            e1_time = time.perf_counter()
            if e1_time - s1_time > 0.3:
                wt_list.append(wt)
                s1_time = e1_time

        if not oq.empty():
            res, fps = oq.get()
            e_time = time.perf_counter()
            # if e_time - s_time > 0.3:
            if res != 'None' and e_time - s_time > 0.1:
                fps_list.append(fps)
                s_time = e_time
            tmp_list.append(res)
            i += 1
        if i == 4:
            if len(set(tmp_list)) == 1:
                result_list.append(tmp_list[0])
            i = 0
            tmp_list = []
        if result_list[-1] == result_list[-2]:
            word = result_list[-1]

        Fps = 'FPS:' + str(fps_list[-1])
        Wt = str(round(wt_list[-1], 1)) + 'ms'
        img = cv2ImgAddText(img, Wt, n - 60, 10, textColor=(255, 0, 0), textSize=15)
        img = cv2ImgAddText(img, Fps, 10, 10, textColor=(255, 0, 0), textSize=15)
        if word != 'None' and word != 'not_classify':
            img = cv2ImgAddText(img, word, txt_n, txt_m, textColor=(255, 0, 0), textSize=30)
        cv2.imshow('demo', img)
        if cv2.waitKey(1) & 0xFF == 27:
            eq.put(1)
            break
    if mode == 'pc' or camera != '0':
        cap.release()
    cv2.destroyAllWindows()
    while not iq.empty():
        iq.get()
    while not oq.empty():
        oq.get()


def main(camera, IP, port, pf):
    iq = Queue(10)  # 图片队列
    oq = Queue(10)  # 结果队列
    wq = Queue(10)  # 延迟队列
    eq = Queue(1)  # 结束标志队列
    sq = Queue(2)  # 开始标志队列
    ADDRESS = (IP, port)
    # 创建一个套接字

    p1 = Process(target=web_send,
                 args=(iq, ADDRESS, wq, eq, sq))
    p2 = Process(target=web_get,
                 args=(oq, ADDRESS, eq, sq))
    p1.daemon = True
    p1.start()
    p2.daemon = True
    p2.start()

    web_video_start(camera, iq, oq, wq, eq, pf, sq)
    p1.join()
    p2.terminate()


if __name__ == '__main__':
    platform = ba_Cfg['platform']
    pose_net = ba_Cfg['pose_net']
    main(Ir_Cfg['source'], Ir_Cfg['IP'], Ir_Cfg['port'], ba_Cfg['platform'])
