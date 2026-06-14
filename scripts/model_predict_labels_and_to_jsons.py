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
from generate_car_datasets import yolov5_detect_parse_opt
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

if __name__ == "__main__":
    input_images_path = r"D:\A_myData\RC26-Vision\dataset\juanZhou_det_mix2\images"
    output_json_path = r"D:\A_myData\RC26-Vision\dataset\juanZhou_det_mix2\jsons"
    detect_model_path = r"D:\A_myData\RC26-Vision\Pytorch\yolov5-master\runs\train\卷轴检测_混合1\weights\best.pt"
    data_yaml = r"D:\A_myData\RC26-Vision\Pytorch\pytorch\dataset_yaml\juanZhou_det_mix1.yaml"
    conf_thres = 0.3
    iou_thres = 0.3
    max_det = 50
    labels = ["r1_red","r2_red","fake_red","r1_blue","r2_blue","fake_blue"]
    generate_v5det_model_predict_labels(input_images_path, output_json_path, detect_model_path, data_yaml, conf_thres, iou_thres, max_det, labels)