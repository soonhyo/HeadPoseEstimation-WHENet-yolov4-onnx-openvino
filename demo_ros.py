"""
xhost +local: && \
docker run --gpus all -it --rm \
-v `pwd`:/home/user/workdir \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--device /dev/video0:/dev/video0:mwr \
--device /dev/video1:/dev/video1:mwr \
--device /dev/video2:/dev/video2:mwr \
--device /dev/video3:/dev/video3:mwr \
--device /dev/video4:/dev/video4:mwr \
--device /dev/video5:/dev/video5:mwr \
--net=host \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
--privileged \
ghcr.io/pinto0309/openvino2tensorflow:latest

sudo chmod 777 /dev/video4 && python3 demo_video.py
"""

import numpy as np
import cv2
import os
import argparse
from math import cos, sin
import onnxruntime
import numba as nb

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

idx_tensor_yaw = [np.array(idx, dtype=np.float32) for idx in range(120)]
idx_tensor = [np.array(idx, dtype=np.float32) for idx in range(66)]


def softmax(x):
    x -= np.max(x,axis=1, keepdims=True)
    a = np.exp(x)
    b = np.sum(np.exp(x), axis=1, keepdims=True)
    return a/b


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2
    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)
    return img


def resize_and_pad(src, size, pad_color=0):
    img = src.copy()
    h, w = img.shape[:2]
    sh, sw = size
    if h > sh or w > sw:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    aspect = w/h
    if aspect > 1:
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = \
            np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = \
            np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    if len(img.shape) == 3 and not isinstance(pad_color, (list, tuple, np.ndarray)):
        pad_color = [pad_color]*3
    scaled_img = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=interp
    )
    scaled_img = cv2.copyMakeBorder(
        scaled_img,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )
    return scaled_img


@nb.njit('i8[:](f4[:,:],f4[:], f4, b1)', fastmath=True, cache=True)
def nms_cpu(boxes, confs, nms_thresh, min_mode):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]
    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]
        keep.append(idx_self)
        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)
        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    return np.array(keep)

class ImageProcessor:
    def __init__(self, args):
        self.node_name = "image_processor"
        rospy.init_node(self.node_name, anonymous=True)
        self.bridge = CvBridge()

        # Subscriber 설정
        self.image_sub = rospy.Subscriber("/camera/color/image_rect_color", Image, self.image_callback)

        # Publisher 설정
        self.image_pub = rospy.Publisher("/processed/image", Image, queue_size=10)

        self.cv_image = None

        # 모델 및 관련 설정 초기화
        self.initialize_models(args)


    def initialize_models(self, args):
        self.yolov4_head_H = 480
        self.yolov4_head_W = 640
        self.whenet_H = 224
        self.whenet_W = 224

        # YOLOv4-Head 모델 초기화
        yolov4_model_name = 'yolov4_headdetection'
        self.yolov4_head = onnxruntime.InferenceSession(
            f'saved_model_{self.whenet_H}x{self.whenet_W}/{yolov4_model_name}_{self.yolov4_head_H}x{self.yolov4_head_W}.onnx',
            providers=[
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
        )
        self.yolov4_head_input_name = self.yolov4_head.get_inputs()[0].name
        self.yolov4_head_output_names = [output.name for output in self.yolov4_head.get_outputs()]

        # WHENet 모델 초기화
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        if args.whenet_mode == 'onnx':
            self.whenet = onnxruntime.InferenceSession(
                f'saved_model_{self.whenet_H}x{self.whenet_W}/whenet_1x3x224x224_prepost.onnx',
                providers=[
                    'CUDAExecutionProvider',
                    'CPUExecutionProvider',
                ]
            )
            self.whenet_input_name = self.whenet.get_inputs()[0].name
            self.whenet_output_names = [output.name for output in self.whenet.get_outputs()]
        elif args.whenet_mode == 'openvino':
            from openvino.inference_engine import IECore
            model_path = f'saved_model_{self.whenet_H}x{self.whenet_W}/openvino/FP16/whenet_{self.whenet_H}x{self.whenet_W}.xml'
            ie = IECore()
            net = ie.read_network(model_path, os.path.splitext(model_path)[0] + ".bin")
            self.exec_net = ie.load_network(network=net, device_name='CPU', num_requests=2)
            self.input_name = next(iter(net.input_info))

    def image_callback(self, data):
        try:
            # ROS 이미지를 OpenCV 형식으로 변환
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def main(self):
        while not rospy.is_shutdown():
            # 이미지 처리
            if self.cv_image is not None:
                processed_image = self.process_image(self.cv_image)
            else:
                continue

            try:
                # 처리된 이미지를 ROS 메시지로 변환하여 publish
                if processed_image is not None:
                    image_msg = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
                    self.image_pub.publish(image_msg)
            except CvBridgeError as e:
                print(e)


    def process_image(self, cv_image):
        if len(cv_image) > 0:
            # ================================================= YOLOv4
            conf_thresh = 0.60
            nms_thresh = 0.50

            # Resize
            resized_frame = resize_and_pad(
                cv_image,
                (self.yolov4_head_H, self.yolov4_head_W)
            )
            width = resized_frame.shape[1]
            height = resized_frame.shape[0]
            # BGR to RGB
            rgb = resized_frame[..., ::-1]
            # HWC -> CHW
            chw = rgb.transpose(2, 0, 1)
            # normalize to [0, 1] interval
            chw = np.asarray(chw / 255., dtype=np.float32)
            # hwc --> nhwc
            nchw = chw[np.newaxis, ...]

            boxes, confs = self.yolov4_head.run(
                output_names = self.yolov4_head_output_names,
                input_feed = {self.yolov4_head_input_name: nchw}
            )

            # [1, boxcount, 1, 4] --> [boxcount, 4]
            boxes = boxes[0][:, 0, :]
            # [1, boxcount, 1] --> [boxcount]
            confs = confs[0][:, 0]

            argwhere = confs > conf_thresh
            boxes = boxes[argwhere, :]
            confs = confs[argwhere]
            # nms
            heads = []
            keep = nms_cpu(
                boxes=boxes,
                confs=confs,
                nms_thresh=nms_thresh,
                min_mode=False
            )
        if (keep.size > 0):
            boxes = boxes[keep, :]
            confs = confs[keep]
            for k in range(boxes.shape[0]):
                heads.append(
                    [
                        int(boxes[k, 0] * width),
                        int(boxes[k, 1] * height),
                        int(boxes[k, 2] * width),
                        int(boxes[k, 3] * height),
                        confs[k],
                    ]
                )
            # ... 이하 YOLOv4 및 WHENet 처리 코드 ...
            # ================================================= WHENet
        canvas = resized_frame.copy()
        croped_resized_frame = None
        if len(heads) > 0:
            for head in heads:
                x_min, y_min, x_max, y_max, _ = head

                # enlarge the bbox to include more background margin
                y_min = max(0, y_min - abs(y_min - y_max) / 10)
                y_max = min(resized_frame.shape[0], y_max + abs(y_min - y_max) / 10)
                x_min = max(0, x_min - abs(x_min - x_max) / 5)
                x_max = min(resized_frame.shape[1], x_max + abs(x_min - x_max) / 5)

                croped_frame = resized_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                # h,w -> 224,224
                croped_resized_frame = cv2.resize(croped_frame, (self.whenet_W, self.whenet_H))
                # bgr --> rgb
                rgb = croped_resized_frame[..., ::-1]
                # hwc --> chw
                chw = rgb.transpose(2, 0, 1)
                # chw --> nchw
                nchw = np.asarray(chw[np.newaxis, :, :, :], dtype=np.float32)

                yaw, pitch, roll = 0.0, 0.0, 0.0
                if args.whenet_mode == 'onnx':
                    outputs = self.whenet.run(
                        output_names = self.whenet_output_names,
                        input_feed = {self.whenet_input_name: nchw}
                    )
                    yaw, roll, pitch = outputs[0][0]
                elif args.whenet_mode == 'openvino':
                    # Normalization
                    rgb = ((rgb / 255.0) - self.mean) / self.std
                    output = self.exec_net.infer(inputs={self.input_name: nchw})
                    yaw, roll, pitch = output['yaw_new/BiasAdd/Add'], \
                                       output['roll_new/BiasAdd/Add'], \
                                       output['pitch_new/BiasAdd/Add']

                yaw, pitch, roll = np.squeeze([yaw, pitch, roll])

                # BBox draw
                deg_norm = 1.0 - abs(yaw / 180)
                blue = int(255 * deg_norm)
                cv2.rectangle(
                    canvas,
                    (int(x_min), int(y_min)),
                    (int(x_max), int(y_max)),
                    color=(blue, 0, 255 - blue),
                    thickness=2
                )

                # Draw axis
                draw_axis(
                    canvas,
                    yaw,
                    pitch,
                    roll,
                    tdx=(x_min + x_max) / 2,
                    tdy=(y_min + y_max) / 2,
                    size=abs(x_max - x_min) // 2
                )
                cv2.putText(
                    canvas,
                    f'yaw: {np.round(yaw)}',
                    (int(x_min), int(y_min)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (100, 255, 0),
                    1
                )
                cv2.putText(
                    canvas,
                    f'pitch: {np.round(pitch)}',
                    (int(x_min), int(y_min) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (100, 255, 0),
                    1
                )
                cv2.putText(
                    canvas,
                    f'roll: {np.round(roll)}',
                    (int(x_min), int(y_min)-30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (100, 255, 0),
                    1
                )
            return canvas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--whenet_mode",
        type=str,
        default='onnx',
        choices=['onnx', 'openvino'],
        help='Choose whether to infer WHENet with ONNX or OpenVINO. Default: onnx',
    )
    parser.add_argument(
        "--device",
        type=str,
        default='0',
        help='Path of the mp4 file or device number of the USB camera. Default: 0',
    )
    parser.add_argument(
        "--height_width",
        type=str,
        default='480x640',
        help='{H}x{W}. Default: 480x640',
    )
    args = parser.parse_args()

    image_processor = ImageProcessor(args)
    try:
        image_processor.main()
    except KeyboardInterrupt:
        print("Shutting down")

