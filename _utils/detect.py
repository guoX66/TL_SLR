# coding:utf-8
from _utils.myutils import *
from common.movenet.moveutils import _process_input, handDetect, get_adjacent_keypoints
from common.handpose.hand_data_iter.datasets import draw_bd_handpose
from _utils.configs import ModelInfo, read_cfg
from torchvision import transforms
from common.open_zoo.pose_demo import pose_inference

modelinfo = ModelInfo()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(modelinfo.ms[0], modelinfo.ms[1])
])
base_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(base_dir)
tr_Cfg, Ir_Cfg, ba_Cfg, Cfg = read_cfg(os.getcwd())
view_mode = Cfg['base']['view_mode']
if view_mode == 1:
    is_show, is_write = True, False
elif view_mode == 2:
    is_show, is_write = False, True
elif view_mode == 3:
    is_show, is_write = True, True
else:
    is_show, is_write = False, False


def Multiprocess(iq, oq, p_pose, p_hand, p_class, class_dict, sq, device, LSTM_model):
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
        hand_model.load_state_dict(torch.load(f'{hand_path}'))
        classify_model.load_state_dict(torch.load(f'{class_path}'))
        pose_model.to(torch.device('cuda'))
        hand_model.to(torch.device('cuda'))
        classify_model.to(torch.device('cuda'))
    else:
        pose_path, hand_path, class_path = p_pose, p_hand, p_class
        while True:
            if not iq.empty():
                img = iq.get()
                break
        if device == 'MYRIAD':
            from common.open_zoo.pose_demo import ov_ini
            pose_model, ie = ov_ini(pose_path, img, 'MYRIAD')
            hand_model = my_convert_model(hand_path, ie, 'MYRIAD')
            classify_model = my_convert_model(class_path, ie, 'MYRIAD')
        else:
            from openvino.runtime import Core
            ie = Core()
            pose_model = my_convert_model(pose_path, ie, device='CPU')
            hand_model = my_convert_model(hand_path, ie, device='CPU')
            classify_model = my_convert_model(class_path, ie, device='CPU')

    if not sq.full():
        sq.put(1)
    count = 0
    seq_len = 10
    input_data = []
    while True:
        if not iq.empty():
            img = iq.get()
            s_time = time.perf_counter()
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

            e_time = time.perf_counter()
            if result is not None:
                oq.put([result, round(1 / (e_time - s_time), 1)])


def video_start(camera, iq, oq, sq):
    mode = 'pc'
    try:
        from picamera2 import Picamera2
        cap = Picamera2()
        cap.configure(cap.create_preview_configuration(main={"format": 'XRGB8888', "size": (800, 500)}))
        cap.start()
        mode = 'rp'
    except:
        if camera == 0:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
            # 图像高度
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
            # 视频帧率
            cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            cap = cv2.VideoCapture(camera)
            cap.set(cv2.CAP_PROP_FPS, 30)

    i = 0
    result_list = [None, None]
    tmp_list = []
    fps_list = [0]
    if mode == 'rp' and camera == 0:
        img = cap.capture_array()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    else:
        ret, img = cap.read()
    iq.put(img)
    word = None
    s_time = time.perf_counter()

    while sq.empty():
        continue
    while True:
        if mode == 'rp' and camera == 0:
            img = cap.capture_array()
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            ret, img = cap.read()

        if img is None:
            if mode == 'pc' or camera != 0:
                cap.release()
            cv2.destroyAllWindows()
            break

        m, n, l = img.shape
        # if tmp_list[0] == tmp_list[1]:
        txt_m = int(0.8 * m)
        txt_n = int(0.5 * m)
        if not iq.full():
            iq.put(img)
        if not oq.empty():
            res, fps = oq.get()
            e_time = time.perf_counter()
            if res is not None and e_time - s_time > 0.5:
                fps_list.append(fps)
                s_time = e_time
            tmp_list.append(res)
            i += 1
        if i == 2:
            if len(set(tmp_list)) == 1:
                result_list.append(tmp_list[0])
            i = 0
            tmp_list = []
        if result_list[-1] == result_list[-2]:
            word = result_list[-1]

        Fps = 'FPS:' + str(fps_list[-1])
        img = cv2ImgAddText(img, Fps, 10, 10, textColor=(255, 0, 0), textSize=15)
        if (word is not None) and (word != 'not_classify'):
            img = cv2ImgAddText(img, word, txt_n, txt_m, textColor=(255, 0, 0), textSize=30)

        cv2.imshow('demo', img)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    if mode == 'pc' or camera != 0:
        cap.release()
    cv2.destroyAllWindows()
    while not iq.empty():
        iq.get()
    while not oq.empty():
        oq.get()
    return


def my_convert_model(model_path, ie, device):
    model = ie.read_model(model_path)
    compiled_model = ie.compile_model(model=model, device_name=device)
    # compiled_model = ie.compile_model(model=model, device_name="MYRIAD")
    return compiled_model


def inference_(compiled_model, input_image):
    input_image = np.expand_dims(input_image, 0)
    # input_image = input_image.astype('float16')
    input_layer_ir = compiled_model.input(0)
    output_layer_ir = compiled_model.output(0)
    result = compiled_model([input_image])[output_layer_ir]
    return result


def get_pose_mean(input_image, pose_model, hand_model, classify_model, class_dict, device, LSTM_model):
    size = 192
    cam_width = 1280
    cam_height = 720
    conf_thres = 0.3
    predict_class = None
    if device == "CUDA" or device == "TensorRt":
        input_image, draw_image = _process_input(input_image, size)
        pose_model.eval()
        with torch.no_grad():
            input_image = torch.Tensor(input_image)
            input_image = input_image.to(torch.device('cuda'))
            out = pose_model(input_image)
            if out is not None:
                kpt_with_conf = out[0, 0, :, :]
                kpt_with_conf = kpt_with_conf.cpu().numpy()
            else:
                return None
    else:
        if device == "MYRIAD":
            draw_image = input_image.copy()
            results = pose_inference(pose_model, input_image)
            if results:
                (kpt_with_conf, scores), frame_meta = results
            else:
                return None
        else:
            input_image, draw_image = _process_input(input_image, size)
            results = inference_(pose_model, input_image[0])
            kpt_with_conf = results[0, 0, :, :]

    if not LSTM_model:
        res = draw_skel_and_kp(draw_image, kpt_with_conf, conf_thres, hand_model, device, LSTM_model)
        predict_class = classify(res, classify_model, class_dict, device)
        return predict_class
    else:
        feature = draw_skel_and_kp(draw_image, kpt_with_conf, conf_thres, hand_model, device, LSTM_model)
        return feature


def draw_skel_and_kp(img, kpt_with_conf, conf_thres, hand_model, device, LSTM_model=False):
    out_img = img
    black_img = np.zeros_like(img)
    height, width, _ = img.shape
    adjacent_keypoints = []
    cv_keypoints = []
    if device == "MYRIAD":
        if len(kpt_with_conf) > 0:
            poses = kpt_with_conf[0]
        else:
            return None
        keypoint_coords = poses[:, :2]
        keypoint_scores = poses[:, 2]
        keypoint_coords[:, [0, 1]] = keypoint_coords[:, [1, 0]]
    else:
        keypoint_scores = kpt_with_conf[:, 2]
        keypoint_coords = kpt_with_conf[:, :2]
        keypoint_coords[:, 0] = keypoint_coords[:, 0] * height
        keypoint_coords[:, 1] = keypoint_coords[:, 1] * width

    # right hand: wrist 10, elbow 8, shoulder 6
    # left hand: wrist 9, elbow 7, shoulder 5
    has_left = True
    has_right = True
    if LSTM_model:
        l_hand = np.zeros((21, 2)) - np.ones((21, 2))
        r_hand = np.zeros((21, 2)) - np.ones((21, 2))
    if keypoint_scores[5] < conf_thres or keypoint_scores[7] < conf_thres or keypoint_scores[9] < conf_thres:
        has_left = False

    if keypoint_scores[6] < conf_thres or keypoint_scores[8] < conf_thres or keypoint_scores[10] < conf_thres:
        has_right = False

    if not (has_left or has_right):
        return None

    if keypoint_coords[9][0] > keypoint_coords[7][0] and keypoint_coords[10][0] > keypoint_coords[8][0]:
        return None
    if has_left:
        left_hand = [keypoint_coords[5], keypoint_coords[7], keypoint_coords[9]]
        hand_result = handDetect(left_hand, height, width)
        if hand_result is not None:
            hand_x, hand_y, w = hand_result
            # cv2.rectangle(img, (hand_x, hand_y), (hand_x + w, hand_y + w), (0, 0, 255))
            output, left_x, left_y = get_handpose(out_img, hand_y, hand_x, w, hand_model, black_img, is_write, device)
            keypoint_coords[9][1] = left_x
            keypoint_coords[9][0] = left_y
            if LSTM_model:
                l_hand = np.array([[output[i * 2 + 0], output[i * 2 + 1]] for i in range(21)])
    if has_right:
        right_hand = [keypoint_coords[6], keypoint_coords[8], keypoint_coords[10]]
        hand_result = handDetect(right_hand, height, width)
        if hand_result is not None:
            hand_x, hand_y, w = hand_result
            # cv2.rectangle(img, (hand_x, hand_y), (hand_x + w, hand_y + w), (0, 0, 255))

            output, right_x, right_y = get_handpose(out_img, hand_y, hand_x, w, hand_model, black_img, is_write, device)
            keypoint_coords[10][1] = right_x
            keypoint_coords[10][0] = right_y
            if LSTM_model:
                r_hand = np.array([[output[i * 2 + 0], output[i * 2 + 1]] for i in range(21)])

    keypoint_coords = keypoint_coords[5:11]

    if not LSTM_model:
        new_keypoints = get_adjacent_keypoints(
            keypoint_scores, keypoint_coords, conf_thres)

        adjacent_keypoints.extend(new_keypoints)

        # right hand: wrist 10, elbow 8, shoulder 6
        # left hand: wrist 9, elbow 7, shoulder 5

        for ks, kc in zip(keypoint_scores, keypoint_coords):
            if ks < conf_thres:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 5))
        if cv_keypoints and is_write:
            black_img = cv2.drawKeypoints(
                black_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            w_img = cv2.drawKeypoints(
                out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            w_img = cv2.polylines(w_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
            cv2.imshow('demo_w', w_img)
            cv2.waitKey(1)
        elif cv_keypoints:
            black_img = cv2.drawKeypoints(
                black_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        black_img = cv2.polylines(black_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))

        return black_img
    else:
        feature = np.concatenate((keypoint_coords, l_hand, r_hand), axis=0)
        feature = npy_crop(feature, height, width, 0.01)
        return feature


def get_handpose(img, hand_x, hand_y, w, model_, black_img, is_write, device, LSTM_model=False):
    hand_img = img[hand_x: hand_x + w, hand_y: hand_y + w, :]
    img_width = hand_img.shape[1]
    img_height = hand_img.shape[0]
    # 输入图片预处理
    img_ = cv2.resize(hand_img, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_ = img_.astype(np.float32)
    img_ = (img_ - 128.) / 256.
    img_ = img_.transpose(2, 0, 1)
    if device == "CUDA" or device == "TensorRt":
        model_.eval()
        img_ = torch.from_numpy(img_)
        img_ = img_.to(torch.device('cuda'))
        img_ = img_.unsqueeze_(0)
        with torch.no_grad():
            pre_ = model_(img_)  # 模型推理
            output = pre_.cpu().detach().numpy()
            output = np.squeeze(output)
    else:
        output = inference_(model_, img_)
        output = np.squeeze(output)
    center_x, center_y = (output[0] * float(img_width)) + hand_y, (output[1] * float(img_height)) + hand_x
    if not LSTM_model:
        pts_hand = {}  # 构建关键点连线可视化结构
        for i in range(int(output.shape[0] / 2)):
            x = (output[i * 2 + 0] * float(img_width))
            y = (output[i * 2 + 1] * float(img_height))

            pts_hand[str(i)] = {}
            pts_hand[str(i)] = {
                "x": x,
                "y": y,
            }

        draw_bd_handpose(black_img, pts_hand, hand_y, hand_x)  # 绘制关键点连线
        # ------------- 绘制关键点
        for i in range(int(output.shape[0] / 2)):
            x = (output[i * 2 + 0] * float(img_width)) + hand_y
            y = (output[i * 2 + 1] * float(img_height)) + hand_x

            cv2.circle(black_img, (int(x), int(y)), 3, (255, 50, 60), -1)
            cv2.circle(black_img, (int(x), int(y)), 1, (255, 150, 180), -1)
        if is_write:
            draw_bd_handpose(img, pts_hand, hand_y, hand_x)  # 绘制关键点连线
            # ------------- 绘制关键点
            for i in range(int(output.shape[0] / 2)):
                x = (output[i * 2 + 0] * float(img_width)) + hand_y
                y = (output[i * 2 + 1] * float(img_height)) + hand_x

                cv2.circle(img, (int(x), int(y)), 3, (255, 50, 60), -1)
                cv2.circle(img, (int(x), int(y)), 1, (255, 150, 180), -1)

    else:
        for i in range(int(output.shape[0] / 2)):
            output[i * 2 + 0] = (output[i * 2 + 0] * float(img_width)) + hand_y
            output[i * 2 + 1] = (output[i * 2 + 1] * float(img_height)) + hand_x
    return output, center_x, center_y


def classify(img, model_, class_dict, device, LSTM_model=False):
    if not LSTM_model:
        img_ = cv2.resize(img, (modelinfo.size[0], modelinfo.size[1]), interpolation=cv2.INTER_CUBIC)
        if is_show:
            cv2.imshow('demo_ov', img_)
            cv2.waitKey(1)
        if device == "CUDA" or device == "TensorRt":
            img_ = Image.fromarray(cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))
            # 输入图片预处理
            img_ = transform(img_)
            l, m, n = img_.shape
            img_ = torch.reshape(img_, (1, 3, m, n))
            img_ = img_.to(torch.device('cuda'))
            model_.eval()
            with torch.no_grad():
                output = model_(img_)  # 模型推理
        else:
            img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
            img_ = img_.transpose(2, 0, 1)  # 修改待预测图片尺寸，需要与训练时一致
            output = inference_(model_, img_)
    else:
        if device == "CUDA" or device == "TensorRt":
            inp = torch.FloatTensor(img)
            inp = inp.unsqueeze(0)
            inp = inp.to(torch.device('cuda'))
            output = model_(inp)
        else:
            img_ = img.transpose(2, 0, 1)  # 修改待预测图片尺寸，需要与训练时一致
            output = inference_(model_, img_)

    if max(output[0]) < 0.5:
        return 'not_classify'
    else:
        pre = output.argmax(1)
        predict_class = class_dict[int(pre)]
        return predict_class
