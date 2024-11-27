import logging
import os
import time

import cv2
import numpy as np
import sophon.sail as sail

from postprocess import PostProcess
from utils import COCO_CLASSES, COLORS

logging.basicConfig(level=logging.INFO)


class YOLO:
    def __init__(self, bmodel_path: str, dev_id: int, conf_thresh: float, nms_thresh: float):
        # load bmodel
        self.net = sail.Engine(bmodel_path, dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(bmodel_path))

        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_names = self.net.get_output_names(self.graph_name)
        # batch_size, channel, height, width
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        for output_name in self.output_names:
            output_shape = self.net.get_output_shape(self.graph_name, output_name)
            if output_shape[1] > output_shape[2]:
                raise ValueError("Python programs do not support the OPT model")

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.dev_id = dev_id
        self.agnostic = False
        self.multi_label = False
        self.max_det = 300

        self.postprocess = PostProcess(
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            agnostic=self.agnostic,
            multi_label=self.multi_label,
            max_det=self.max_det,
        )

        self.init()

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess(self, ori_img):
        """
        pre-processing

        Args:
            img: numpy.ndarray -- (h, w, c)

        Returns:
            (c, h, w) numpy.ndarray after pre-processing
        """
        letterbox_img, ratio, (tx1, ty1) = self.letterbox(
            ori_img,
            new_shape=(self.net_h, self.net_w),
            color=(114, 114, 114),
            auto=False,
            scaleFill=False,
            scaleup=True,
            stride=32,
        )

        img = letterbox_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = img.astype(np.float32)

        # input_data = np.expand_dims(input_data, 0)
        img = np.ascontiguousarray(img / 255.0)
        return img, ratio, (tx1, ty1)

    def letterbox(
        self,
        img,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=False,
        scaleFill=False,
        scaleup=True,
        stride=32,
    ):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

        # divide padding into 2 sides
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border

        return img, ratio, (dw, dh)

    def inference(self, input_img, img_num):
        input_data = {self.input_name: input_img}
        cores = [self.dev_id]
        outputs = self.net.process(self.graph_name, input_data, cores)

        # resort
        out_keys = list(outputs.keys())
        ord = []
        for n in self.output_names:
            for i, k in enumerate(out_keys):
                if n == k:
                    ord.append(i)
                    break
        out = [outputs[out_keys[i]][:img_num] for i in ord]
        return out

    def __call__(self, img_list: list) -> dict:
        img_num = len(img_list)
        ori_size_list = []
        preprocessed_img_list = []
        ratio_list = []
        txy_list = []
        for ori_img in img_list:
            ori_h, ori_w = ori_img.shape[:2]
            ori_size_list.append((ori_w, ori_h))
            start_time = time.time()
            preprocessed_img, ratio, (tx1, ty1) = self.preprocess(ori_img)
            self.preprocess_time += time.time() - start_time
            preprocessed_img_list.append(preprocessed_img)
            ratio_list.append(ratio)
            txy_list.append([tx1, ty1])

        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype="float32")
            input_img[:img_num] = np.stack(preprocessed_img_list)

        start_time = time.time()
        outputs = self.inference(input_img, img_num)
        self.inference_time += time.time() - start_time

        start_time = time.time()
        results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
        self.postprocess_time += time.time() - start_time

        return results

    def predict(self, img_list: list, filename_list: list, output_img_dir: str) -> dict:
        results = self(img_list)
        for i, filename in enumerate(filename_list):
            det = results[i]
            det_draw = det[det[:, -2] > self.conf_thresh]
            res_img = draw_numpy(
                img_list[i],
                det_draw[:, :4],
                masks=None,
                classes_ids=det_draw[:, -1],
                conf_scores=det_draw[:, -2],
            )
            cv2.imwrite(os.path.join(output_img_dir, filename), res_img)
            res_dict = dict()
            res_dict["image_name"] = filename
            res_dict["bboxes"] = []
            for idx in range(det.shape[0]):
                bbox_dict = dict()
                x1, y1, x2, y2, score, category_id = det[idx]
                bbox_dict["bbox"] = [
                    float(round(x1, 3)),
                    float(round(y1, 3)),
                    float(round(x2 - x1, 3)),
                    float(round(y2 - y1, 3)),
                ]
                bbox_dict["category_id"] = int(category_id)
                bbox_dict["score"] = float(round(score, 5))
                res_dict["bboxes"].append(bbox_dict)

        return res_dict


def draw_numpy(image, boxes, masks=None, classes_ids=None, conf_scores=None):
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        if classes_ids is not None:
            color = COLORS[int(classes_ids[idx]) + 1]
        else:
            color = (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        if classes_ids is not None and conf_scores is not None:
            classes_ids = classes_ids.astype(np.int8)
            cv2.putText(
                image,
                COCO_CLASSES[classes_ids[idx] + 1]
                + ":"
                + str(round(conf_scores[idx], 2)),
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                thickness=2,
            )

            logging.debug(
                "class id={:2}, score={:.2f}, (x1={:3}, y1={:3}, x2={:3}, y2={:3})".format(
                    classes_ids[idx], conf_scores[idx], x1, y1, x2, y2)
            )

        if masks is not None:
            mask = masks[:, :, idx]
            image[mask] = image[mask] * 0.5 + np.array(color) * 0.5

    return image
