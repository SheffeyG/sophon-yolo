import json
import os
import time

import cv2
import numpy as np

from yolo import YOLO
from utils import is_img

DEV_ID = 0
CONF_THRESH = 0.25
NMS_THRESH = 0.7

images_path = "./images"
bmodel_path = "./models/yolov11s_int8_1b.bmodel"
output_path = "./results"


def main():
    global images_path, bmodel_path, output_path

    # check files
    if not os.path.exists(bmodel_path):
        raise FileNotFoundError("{} is not existed.".format(bmodel_path))
    if not os.path.isdir(images_path):
        raise NotADirectoryError("{} is not a directory.".format(images_path))
    # if not os.path.exists(target_path):
    #     raise FileNotFoundError("{} is not existed.".format(target_path))

    # creat save path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_img_dir = os.path.join(output_path, "images")
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir)

    yolov11 = YOLO(bmodel_path, DEV_ID, CONF_THRESH, NMS_THRESH)
    batch_size = yolov11.batch_size

    yolov11.init()
    decode_time = 0.0

    img_list = []
    filename_list = []
    results_list = []
    img_count = 0

    for root, _, filenames in os.walk(images_path):
        filenames.sort()
        for filename in filenames:
            if not is_img(filename):
                continue
            img_file = os.path.join(root, filename)
            img_count += 1
            print("img{}: {}".format(img_count, img_file))

            # decode
            start_time = time.time()
            src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
            if src_img is None:
                print("{} imdecode is None.".format(img_file))
                continue
            if len(src_img.shape) != 3:
                src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
            decode_time += time.time() - start_time

            img_list.append(src_img)
            filename_list.append(filename)

            # predict in a full batch size
            if len(img_list) == batch_size:
                res = yolov11.predict(img_list, filename_list, output_img_dir)
                results_list.append(res)
                img_list.clear()
                filename_list.clear()

    # predict the rest ones
    if len(img_list):
        res = yolov11.predict(img_list, filename_list, output_img_dir)
        results_list.append(res)

    print(json.dumps(results_list, indent=4))

    # calculate speed
    print("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / img_count
    preprocess_time = yolov11.preprocess_time / img_count
    inference_time = yolov11.inference_time / img_count
    postprocess_time = yolov11.postprocess_time / img_count
    print("images_count: {}".format(len(results_list)))
    print("decode_time(ms): {:.2f}".format(decode_time * 1000))
    print("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    print("inference_time(ms): {:.2f}".format(inference_time * 1000))
    print("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))


if __name__ == "__main__":
    main()
