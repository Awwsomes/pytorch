import os
import shutil
import json
import cv2
import numpy as np
from tqdm import tqdm
import re
from ultralytics import YOLO
import argparse

import sys
sys.path.append(r"D:\A_myData\RC26-Vision\Pytorch\yolov5-master")
import detect as yolov5_detect
from txt_to_json import txt_to_json
from fix_json_path import batch_fix_image_path

def generate_v5det_model_predict_labels(input_images_path:str, output_json_path:str, detect_model_path:str, data_yaml:str, conf_thres:float, iou_thres:float, max_det:int, labels:list):
    """
    利用已有v5角点检测模型预测图片，生成数据集，同时转换txt为labelme_json
    需要结合官方yolov5-master包运行
    在总数据集的根目录下新建corner_jsons，存放json文件

    :param input_images_path: 需要用模型打标的图片的文件夹
    :param output_json_path: 总数据集的根目录
    :param detect_model_path: v5检测模型路径
    :param data_yaml:
    :param conf_thres: 用模型预测时的置信度阈值
    :param iou_thres:
    :param max_det:
    :param labels: 类别映射
    :return: 无
    """

    print("生成数据集...\n")

    # 创建json
    jsons_path = output_json_path
    os.makedirs(jsons_path, exist_ok=True)
    images_path = input_images_path

    # 调用detct预测
    print("v5模型预测图片...\n")
    opt = yolov5_detect_parse_opt(detect_model_path, images_path, data_yaml, conf_thres, iou_thres, max_det)
    save_path = yolov5_detect.main(opt)
    # print(save_path)

    # 将txt转为json
    print("txt转换为labelme_json...\n")
    txts_path = os.path.join(save_path, "labels")
    txt_to_json(txts_path, images_path, jsons_path, labels)

    # json中的图片路径转为相对于图片路径的相对路径
    batch_fix_image_path(output_json_path, input_images_path)

    print("完成\n")

def yolov5_detect_parse_opt(model_path:str, imgs_path:str, data_yaml:str,conf_thres:float, iou_thres:float, max_det:int):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=model_path, help="model path or triton URL")
    parser.add_argument("--source", type=str, default=imgs_path, help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=data_yaml, help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=conf_thres, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=iou_thres, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=max_det, help="maximum detections per image")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true",default=False, help="show results")
    parser.add_argument("--save-txt", action="store_true", default=True, help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", default=True,help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", default=True,help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=r"D:\A_myData\RC26-Vision\Pytorch\yolov5-master\runs\detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # print_args(vars(opt))
    return opt

if __name__ == "__main__":
    input_images_path = r"F:\RC2026_OFFLINE\datasets\juanZhou_det_mix3\images"
    output_json_path = r"F:\RC2026_OFFLINE\datasets\juanZhou_det_mix3\jsons"
    detect_model_path = r"D:\A_myData\RC26-Vision\Pytorch\yolov5-master\runs\train\卷轴检测_混合2_红蓝一起摆\weights\best.pt"
    data_yaml = r"D:\A_myData\RC26-Vision\Pytorch\pytorch\dataset_yaml\juanZhou_det_mix3.yaml"
    conf_thres = 0.3
    iou_thres = 0.3
    max_det = 50
    labels = ["r1_red","r2_red","fake_red","r1_blue","r2_blue","fake_blue"]
    generate_v5det_model_predict_labels(input_images_path, output_json_path, detect_model_path, data_yaml, conf_thres, iou_thres, max_det, labels)