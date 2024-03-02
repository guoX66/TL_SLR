import math
import torch
import cv2
import numpy as np
from common.handpose.hand_data_iter.datasets import draw_bd_handpose
from common.movenet.constants import *


def hanpose_onnx(img, hand_x, hand_y, w, model_, black_img, device):
    hand_img = img[hand_x: hand_x + w, hand_y: hand_y + w, :]
    img_width = hand_img.shape[1]
    img_height = hand_img.shape[0]
    # 输入图片预处理
    img_ = cv2.resize(hand_img, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_ndarray = img_.transpose((2, 0, 1))
    img_ndarray = img_ndarray / 255.
    img_ndarray = np.expand_dims(img_ndarray, 0)

    output = model_.forward(img_ndarray.astype('float32'))[0][0]
    output = np.array(output)

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


def get_handpose(img, hand_x, hand_y, w, model_, black_img, device):
    hand_img = img[hand_x: hand_x + w, hand_y: hand_y + w, :]
    img_width = hand_img.shape[1]
    img_height = hand_img.shape[0]
    # 输入图片预处理
    img_ = cv2.resize(hand_img, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_ = img_.astype(np.float32)
    img_ = (img_ - 128.) / 256.
    img_ = img_.transpose(2, 0, 1)
    model_.eval()
    # if use_cuda:
    #     img_ = img_.cuda()  # (bs, 3, h, w)
    with torch.no_grad():
        img_ = torch.from_numpy(img_)
        img_ = img_.to(device)
        img_ = img_.unsqueeze_(0)
        pre_ = model_(img_)  # 模型推理
        output = pre_.cpu().detach().numpy()
        output = np.squeeze(output)

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


def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def _process_input(source_img, size=192, crop_region=None):
    if crop_region != None:
        input_img = source_img[crop_region['y_min']:crop_region['y_max'], crop_region['x_min']:crop_region['y_max']]
        input_img = cv2.resize(input_img, (size, size),
                               interpolation=cv2.INTER_LINEAR)
    else:
        input_img = cv2.resize(source_img, (size, size),
                               interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    # input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, size, size)
    input_img = input_img.reshape(1, size, size, 3)
    return input_img, source_img


def read_cap(cap, size=192, crop_region=None):
    res, img = cap.read()
    if not res:
        raise IOError("webcam failure")
    return _process_input(img, size, crop_region)


def read_imgfile(path, size=192):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

    return _process_input(img, size)


def draw_keypoints(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_confidence:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
    out_img = cv2.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
    return out_img


def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1],
                      keypoint_coords[right][::-1]]).astype(np.int32),
        )
    return results


def draw_skeleton(img, instance_scores, keypoint_scores, keypoint_coords, min_pose_confidence=0.5,
                  min_part_confidence=0.5):
    out_img = img
    adjacent_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_confidence)
        adjacent_keypoints.extend(new_keypoints)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img


def handDetect(hand_coords, image_height, image_width):
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5

    # right hand: wrist 10, elbow 8, shoulder 6
    # left hand: wrist 9, elbow 7, shoulder 5
    ratioWristElbow = 0.3
    y1, x1 = hand_coords[0]
    y2, x2 = hand_coords[1]
    y3, x3 = hand_coords[2]

    x = x3 + ratioWristElbow * (x3 - x2)
    y = y3 + ratioWristElbow * (y3 - y2)
    distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    width = 1 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
    # x-y refers to the center --> offset to topLeft point
    # handRectangle.x -= handRectangle.width / 2.f;
    # handRectangle.y -= handRectangle.height / 2.f;
    x -= width / 2
    y -= width / 2  # width = height
    # overflow the image
    if x < 0: x = 0
    if y < 0: y = 0

    width1 = width
    width2 = width
    if x + width > image_width: width1 = image_width - x
    if y + width > image_height: width2 = image_height - y
    width = min(width1, width2)

    # the max hand box value is 20 pixels
    detect_result = None
    if width >= 20:
        detect_result = [int(x), int(y), int(width)]

    '''
    return value: [[x, y, w, True if left hand else False]].
    width=height since the network require squared input.
    x, y is the coordinate of top left 
    '''
    return detect_result


def draw_skel_and_kp(img, kpt_with_conf, conf_thres, device, hand_model, is_onnx):
    out_img = img
    black_img = np.zeros_like(img)
    height, width, _ = img.shape
    adjacent_keypoints = []
    cv_keypoints = []
    keypoint_scores = kpt_with_conf[:, 2]
    keypoint_coords = kpt_with_conf[:, :2]
    keypoint_coords[:, 0] = keypoint_coords[:, 0] * height
    keypoint_coords[:, 1] = keypoint_coords[:, 1] * width
    # right hand: wrist 10, elbow 8, shoulder 6
    # left hand: wrist 9, elbow 7, shoulder 5
    # cv2.circle(img, (int(keypoint_coords[10][1]), int(keypoint_coords[10][0])), 5, (0, 0, 225))
    # cv2.circle(img, (int(keypoint_coords[8][1]), int(keypoint_coords[8][0])), 5, (0, 0, 225))
    # cv2.circle(img, (1, 100), 5, (0, 0, 225))
    has_left = True
    has_right = True
    if keypoint_scores[5] < conf_thres or keypoint_scores[7] < conf_thres or keypoint_scores[9] < conf_thres:
        has_left = False

    if keypoint_scores[6] < conf_thres or keypoint_scores[8] < conf_thres or keypoint_scores[10] < conf_thres:
        has_right = False

    if not (has_left or has_right):
        return None

    if keypoint_coords[9][0] > keypoint_coords[7][0] and keypoint_coords[10][0] > keypoint_coords[8][0]:
        return None

    left_hand = [keypoint_coords[5], keypoint_coords[7], keypoint_coords[9]]
    hand_result = handDetect(left_hand, height, width)
    if hand_result is not None:
        hand_x, hand_y, w = hand_result
        # cv2.rectangle(img, (hand_x, hand_y), (hand_x + w, hand_y + w), (0, 0, 255))
        if is_onnx:
            hanpose_onnx(out_img, hand_y, hand_x, w, hand_model, black_img, device)
        else:
            get_handpose(out_img, hand_y, hand_x, w, hand_model, black_img, device)
    else:
        return None

    right_hand = [keypoint_coords[6], keypoint_coords[8], keypoint_coords[10]]
    hand_result = handDetect(right_hand, height, width)
    if hand_result is not None:
        hand_x, hand_y, w = hand_result
        # cv2.rectangle(img, (hand_x, hand_y), (hand_x + w, hand_y + w), (0, 0, 255))
        if is_onnx:
            hanpose_onnx(out_img, hand_y, hand_x, w, hand_model, black_img, device)
        else:
            get_handpose(out_img, hand_y, hand_x, w, hand_model, black_img, device)
    else:

        return None
    new_keypoints = get_adjacent_keypoints(
        keypoint_scores, keypoint_coords, conf_thres)

    adjacent_keypoints.extend(new_keypoints)
    # cv2.circle(out_img, (int(keypoint_coords[6][1]), int(keypoint_coords[6][0])), 10, [0, 0, 255], 0)
    # cv2.circle(out_img, (int(keypoint_coords[8][1]), int(keypoint_coords[8][0])), 10, [0, 0, 255], 0)
    # right hand: wrist 10, elbow 8, shoulder 6
    # left hand: wrist 9, elbow 7, shoulder 5

    for ks, kc in zip(keypoint_scores, keypoint_coords):

        if ks < conf_thres:
            continue

        cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 5))

    if cv_keypoints:
        black_img = cv2.drawKeypoints(
            black_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    black_img = cv2.polylines(black_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))

    return black_img


'''
Intelligent cropping algorithm borrowed from Movenet doc:
https://www.tensorflow.org/hub/tutorials/movenet
'''
# Confidence score to determine whether a keypoint prediction is reliable.
MIN_CROP_KEYPOINT_SCORE = 0.2

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}


def init_crop_region(image_height, image_width):
    """Defines the default crop region.

    The function provides the initial crop region (pads the full image from both
    sides to make it a square image) when the algorithm cannot reliably determine
    the crop region from the previous frame.
    """
    #   if image_width > image_height:
    #     box_height = image_width / image_height
    #     box_width = 1.0
    #     y_min = (image_height / 2 - image_width / 2) / image_height
    #     x_min = 0.0
    #   else:
    #     box_height = 1.0
    #     box_width = image_height / image_width
    #     y_min = 0.0
    #     x_min = (image_width / 2 - image_height / 2) / image_width

    #   return {
    #     'y_min': y_min,
    #     'x_min': x_min,
    #     'y_max': y_min + box_height,
    #     'x_max': x_min + box_width,
    #     'height': box_height,
    #     'width': box_width
    #   }
    return {
        'y_min': 0,
        'x_min': 0,
        'y_max': image_height,
        'x_max': image_width,
        'height': 1.0,  # image_width / image_height,
        'width': 1.0
    }


def torso_visible(keypoints):
    """Checks whether there are enough torso keypoints.

    This function checks whether the model is confident at predicting one of the
    shoulders/hips which is required to determine a good crop region.
    """
    return ((keypoints[KEYPOINT_DICT['left_hip'], 2] >
             MIN_CROP_KEYPOINT_SCORE or
             keypoints[KEYPOINT_DICT['right_hip'], 2] >
             MIN_CROP_KEYPOINT_SCORE) and
            (keypoints[KEYPOINT_DICT['left_shoulder'], 2] >
             MIN_CROP_KEYPOINT_SCORE or
             keypoints[KEYPOINT_DICT['right_shoulder'], 2] >
             MIN_CROP_KEYPOINT_SCORE))


def determine_torso_and_body_range(
        keypoints, target_keypoints, center_y, center_x):
    """Calculates the maximum distance from each keypoints to the center location.

    The function returns the maximum distances from the two sets of keypoints:
    full 17 keypoints and 4 torso keypoints. The returned information will be
    used to determine the crop size. See determineCropRegion for more detail.
    """
    torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    max_torso_yrange = 0.0
    max_torso_xrange = 0.0
    for joint in torso_joints:
        dist_y = abs(center_y - target_keypoints[joint][0])
        dist_x = abs(center_x - target_keypoints[joint][1])
        if dist_y > max_torso_yrange:
            max_torso_yrange = dist_y
        if dist_x > max_torso_xrange:
            max_torso_xrange = dist_x

    max_body_yrange = 0.0
    max_body_xrange = 0.0
    for joint in KEYPOINT_DICT.keys():
        if keypoints[KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
            continue
        dist_y = abs(center_y - target_keypoints[joint][0]);
        dist_x = abs(center_x - target_keypoints[joint][1]);
        if dist_y > max_body_yrange:
            max_body_yrange = dist_y

        if dist_x > max_body_xrange:
            max_body_xrange = dist_x

    return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]


def determine_crop_region(
        keypoints, image_height,
        image_width):
    """Determines the region to crop the image for the model to run inference on.

    The algorithm uses the detected joints from the previous frame to estimate
    the square region that encloses the full body of the target person and
    centers at the midpoint of two hip joints. The crop size is determined by
    the distances between each joints and the center point.
    When the model is not confident with the four torso joint predictions, the
    function returns a default crop which is the full image padded to square.
    """
    target_keypoints = {}
    for joint in KEYPOINT_DICT.keys():
        target_keypoints[joint] = [
            keypoints[KEYPOINT_DICT[joint], 0] * image_height,
            keypoints[KEYPOINT_DICT[joint], 1] * image_width
        ]

    if torso_visible(keypoints):
        center_y = (target_keypoints['left_hip'][0] +
                    target_keypoints['right_hip'][0]) / 2;
        center_x = (target_keypoints['left_hip'][1] +
                    target_keypoints['right_hip'][1]) / 2;

        (max_torso_yrange, max_torso_xrange,
         max_body_yrange, max_body_xrange) = determine_torso_and_body_range(
            keypoints, target_keypoints, center_y, center_x)

        crop_length_half = np.amax(
            [max_torso_xrange * 1.9, max_torso_yrange * 1.9,
             max_body_yrange * 1.2, max_body_xrange * 1.2])

        tmp = np.array(
            [center_x, image_width - center_x, center_y, image_height - center_y])
        crop_length_half = np.amin(
            [crop_length_half, np.amax(tmp)]);

        crop_corner = [center_y - crop_length_half, center_x - crop_length_half];

        if crop_length_half > max(image_width, image_height) / 2:
            return init_crop_region(image_height, image_width)
        else:
            crop_length = crop_length_half * 2;
            return {
                'y_min': crop_corner[0],  # / image_height,
                'x_min': crop_corner[1],  # / image_width,
                'y_max': (crop_corner[0] + crop_length),  # / image_height,
                'x_max': (crop_corner[1] + crop_length),  # / image_width,
                'height': (crop_corner[0] + crop_length) / image_height -
                          crop_corner[0] / image_height,
                'width': (crop_corner[1] + crop_length) / image_width -
                         crop_corner[1] / image_width
            }
    else:
        return init_crop_region(image_height, image_width)


def crop_and_resize(image, crop_region, crop_size):
    """Crops and resize the image to prepare for the model input."""
    boxes = [[crop_region['y_min'], crop_region['x_min'],
              crop_region['y_max'], crop_region['x_max']]]
    output_image = tf.image.crop_and_resize(
        image, box_indices=[0], boxes=boxes, crop_size=crop_size)
    return output_image
