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

import zbuffer.zb_main as zb
from convert_train_dataset_to_class_dataset import classify_dataset_by_cls_predict_label
from rename_data import rename_data_one_dir
from class_about.change_txt_format import change_txt_format
from txt_to_json import txt_to_json

from config.car_dataset_config import config
from config.car_dataset_config import Config

def generate_global_dataset(root_dir:str, output_root_path:str, start_idx:int):
    """
    复制global图片并重命名为"image_{}"，序号从start_idx开始
    对应的rt.txt转换为"label_{}"的json文件，序号从start_idx开始
    把rt.txt里的label_list从1234转换为01（转换为筛空列表）
    会处理根目录底下所有子目录的文件。
    输出文件全部拷贝到输出目录下（一个文件夹）
    文件夹格式
    ----dataset
        ----images
   	        ----image_1.png
      	    ----image_2.png
      	    ...
    ----roi_images
        ----roi_1
            ----1.png
       	    .
       	    .
       	    ----12.png
	    ----roi_2
    ----labels
   	    ----label_1.json
   	    ----label_2.json
   	    ...
    label格式
	{
		"rvec": [],
		"tvec": [],
		"labels": [],
		"point_size": []
	}
    :param root_dir: 输入数据根文件夹
    :param output_root_path: 输出数据集根文件夹
    :param start_idx: 文件开始序号
    :return: 无
    """
    print("拷贝并重命名 global_images, 处理rt.txt...\n")

    # 递归遍历根文件夹及子文件夹下所有内容
    idx = start_idx
    for root,_,files in tqdm(os.walk(root_dir)):
        # print(len(files))

        # 跳过无文件的文件夹
        if len(files) == 0:
            continue

        # 筛选出名字为image的图片
        global_imgs_list = [x for x in files if
                            os.path.splitext(x)[1] in [".png", ".jpg", ".jpeg", ".bmp"]]
        global_imgs_list = [x for x in global_imgs_list if
                            os.path.splitext(x)[0] == "image"]

        # 筛选出名字为rt的txt
        txts_list = [x for x in files if os.path.splitext(x)[1] == ".txt"]
        txts_list = [x for x in txts_list if os.path.splitext(x)[0] == "rt"]
        # print(len(global_imgs_list), len(txts_list))

        # 跳过无需要的文件的文件夹
        if len(global_imgs_list) == 0 or len(txts_list) == 0:
            continue

        # if idx == 137:
        #     print(f"idx 137: {root}")

        # 只处理第一个文件，因为按照格式只会有一个文件
        # 拷贝图片
        old_img_path = os.path.join(root,global_imgs_list[0])
        _,img_ext = os.path.splitext(global_imgs_list[0])
        img_new_name = f"image_{idx}{img_ext}"
        new_img_path = os.path.join(output_root_path, "images", img_new_name)
        # print(new_img_path)
        shutil.copy(old_img_path,new_img_path)

        # 读取txt，生成json
        # 拼接路径
        old_txt_path = os.path.join(root, txts_list[0])
        new_json_path = os.path.join(output_root_path, "labels", f"label_{idx}.json")
        # print(new_json_path)

        # 读取txt，生成json
        with open(old_txt_path, 'r') as txt_file, \
             open(new_json_path, 'w') as json_file:

            # 读取txt内容
            lines = txt_file.readlines()
            if not len(lines) == 3:
                print(f"[ERROR] rt.txt 's format seems wrong.")
            rvec = [float(x.strip()) for x in lines[0].strip().split(",")]
            tvec = [float(x.strip()) for x in lines[1].strip().split(",")]
            labels = [int(x) for x in lines[2].strip().split()]
            # 转换成 1 存在， 0 不存在
            labels_convert = [0 if x == 4 else 1 for x in labels]
            # 蓝色场地，转换成一号位从左边数起
            # if blue_or_red:
            #     labels_convert_temp = [labels_convert[2], labels_convert[1], labels_convert[0],
            #                            labels_convert[5], labels_convert[4], labels_convert[3],
            #                            labels_convert[8], labels_convert[7], labels_convert[6],
            #                            labels_convert[11], labels_convert[10], labels_convert[9]]
            #     labels_convert = labels_convert_temp

            # 生成字典，写入json文件
            # json_content = {
            #     "rvec": rvec,
            #     "tvec": tvec,
            #     "labels": labels_convert
            # }
            # json.dump(json_content, json_file)
            json_content = (f"{{\n"
                            f"    \"rvec\": {rvec},\n"
                            f"    \"tvec\": {tvec},\n"
                            f"    \"labels\": {labels_convert}\n"
                            f"}}")
            json_file.writelines(json_content)

        # 序号加一
        idx += 1
    print("完成.\n")

def generate_11class_dataset(class_model_path, input_roi_img_path, output_root_dir, start_idx,
                             img_root_name:str, label_name_list:list[str], model_predict_output_dir:str):
    """
    用已有模型预测， 从车上log数据生成分类数据集
    使用模型预测出的最高置信度的类别作为图片的类别
    默认文件名起始序号为0

    :param class_model_path: 用于预测的分类模型
    :param input_roi_img_path: 存放待预测图片的文件夹
    :param output_root_dir: 输出分类数据集的根文件夹
    :param start_idx: 文件名起始序号
    :param img_root_name: 图片根名字
    :param label_name_list: 类别名列表
    :param model_predict_output_dir: 模型预测输出txt存放项目目录
    :return: 无
    """
    print("生成分类数据集中...\n")

    # 创建临时的检测数据集
    root_path = "temp"
    imgs_path = os.path.join(root_path, "images")
    # labels_path = os.path.join(root_path, "labels_no_fix")
    # os.makedirs(labels_path, exist_ok=True)
    os.makedirs(imgs_path, exist_ok=True)

    # # 获取输出图片根文件名（和数据集文件夹名相同）
    # img_root_name = os.path.split(output_root_dir)[1]
    # img_root_name += "_"

    # 重命名并复制roi图片
    print("拷贝并重命名roi图片...\n")
    rename_data_one_dir(input_roi_img_path, imgs_path, img_root_name=img_root_name, start_idx=start_idx, list_ext=[".png", ".jpg", ".jpeg", ".bmp"])

    # 读取模型
    model = YOLO(class_model_path)
    if model.task != "classify":
        raise TypeError(f"[ERROR] Not a class model: {class_model_path}")

    # roi给YOLO分类预测
    print("分类模型预测roi图片...\n")
    if not os.path.exists(model_predict_output_dir):
        model_predict_output_dir = r"model_predict"
    result = model.predict(source=imgs_path,
                  show=False,
                  save_txt=True,
                  save_conf=True,
                  save=False,
                  project=model_predict_output_dir,
                  task="classify"
                  )

    # # 拷贝txt到临时检测数据集内（原路径 yolo11_cls\runs\classify）
    # txt_list = os.listdir(os.path.join(result[0].save_dir, "labels"))
    # for txt in txt_list:
    #     txt_path = os.path.join(result[0].save_dir, "labels", txt)
    #     new_txt_path = os.path.join(labels_path, txt)
    #     shutil.copy(txt_path, new_txt_path)

    # 转换txt格式（conf label -> label conf）
    new_labels_path = os.path.join(root_path, "labels")
    change_txt_format(os.path.join(result[0].save_dir, "labels"), new_labels_path)

    # 生成分类数据集
    print("构建分类数据集中...\n")
    output_path = os.path.join(output_root_dir, "class_dataset")
    classify_dataset_by_cls_predict_label(imgs_path, new_labels_path, output_path, label_name_list)

    # 删除临时的检测数据集
    shutil.rmtree(root_path)

    # 保存信息：能从生成的数据集找到原始模型预测出来的txt，包含top5置信度
    raw_model_predict_path = os.path.join(output_path, "raw_model_predict_path.txt")
    with open(raw_model_predict_path, 'w') as txt:
        txt.write(f"{result[0].save_dir}")
    with open(os.path.join(result[0].save_dir, "对应数据集文件夹路径.txt"), 'w') as txt:
        txt.write(f"{output_path}")

    print("完成.\n")

def generate_roi_data(output_root_path):
    """
    利用zb生成roi图片，写入输出根目录下的roi_images文件夹，格式如下
    ----dataset
        ----images
        ----roi_images
            ----roi_1
                ----1.png
       	        .
       	        .
       	        ----12.png
	        ----roi_2
        ----labels
    同时在label json中写入有效点数point_size
    label格式
	{
		"rvec": [],
		"tvec": [],
		"labels": [],
		"point_size": []
	}
    只能利用generate_global_dataset生成的数据集格式生成global图像对应的roi图像和point_size
    :param output_root_path: 输出根目录
    :return: 无
    """
    # 生成并写入roi图像和point_size

    # 读图像，读rt
    print("读取全局图，读取label_json...")
    global_img_names_list = os.listdir(os.path.join(output_root_path, "images"))
    global_img_names_list = [x for x in global_img_names_list if
                             os.path.splitext(os.path.join(output_root_path, x))[1] in [".png", ".jpg", ".jpeg",
                                                                                        ".bmp"]]
    global_img_names_list.sort(key=lambda f: int(re.findall(r'\d+', f)[0]) if re.findall(r'\d+', f) else float('inf'))
    global_imgs_list = [cv2.imread(os.path.join(output_root_path, "images", img)) for img in global_img_names_list]

    rvec_list = []
    tvec_list = []
    labels_list = []
    idx_list = []
    for img_name in tqdm(global_img_names_list):
        # 获取图片名字里的序号
        idx = os.path.splitext(img_name)[0].split("_")[1]
        # print(idx)
        idx_list.append(idx)

        label_name = f"label_{idx}.json"
        label_path = os.path.join(output_root_path, "labels", label_name)
        # print(label_path)
        if not os.path.exists(label_path):
            # raise FileNotFoundError(f"[WARN]: {img_name} 's label not exists")
            print(f"[WARN]: {img_name} 's label not exists, skip...")
            continue
        with open(label_path, 'r') as label_file:
            json_content = json.load(label_file)
            rvec_list.append(np.stack(json_content["rvec"]).reshape(3, 1))
            tvec_list.append(np.stack(json_content["tvec"]).reshape(3, 1))
            labels_list.append(np.stack(json_content["labels"]))

    # 拼接成batch
    # global_imgs_batch = np.stack(global_imgs_list, axis=0)
    # rvec_batch = np.stack(rvec_list, axis=0)
    # tvec_batch = np.stack(tvec_list, axis=0)
    # labels_batch = np.stack(labels_list, axis=0)

    # 每十张拼成一个小batch，再拼回一个大batch
    # full_batch_num = len(global_imgs_list) // 10
    # last_batch_amount = len(global_imgs_list) % 10
    #
    # global_imgs_batch = np.zeros((full_batch_num+1, 10))
    # rvec_batch = np.zeros((full_batch_num+1, 10, 1, 3))
    # tvec_batch = np.zeros((full_batch_num+1, 10, 1, 3))
    # labels_batch = np.zeros((full_batch_num+1, 10, 12))
    # for i in range(full_batch_num+1):
    #     if i != full_batch_num:
    #         global_imgs_batch[i, :] = np.stack(global_imgs_list[10*i:10*(i+1)], axis=0)
    #         rvec_batch[i, :] = np.stack(rvec_list[10*i:10*(i+1)], axis=0)
    #         tvec_batch[i, :] = np.stack(tvec_list[10 * i:10 * (i + 1)], axis=0)
    #         labels_batch[i, :] = np.stack(labels_list[10 * i:10 * (i + 1)], axis=0)
    #     else:
    #         global_imgs_batch[i, :last_batch_amount] = np.stack(global_imgs_list[10*i:10*i+last_batch_amount], axis=0)
    #         rvec_batch[i, :last_batch_amount] = np.stack(rvec_list[10*i:10*i+last_batch_amount], axis=0)
    #         tvec_batch[i, :last_batch_amount] = np.stack(tvec_list[10*i:10*i+last_batch_amount], axis=0)
    #         labels_batch[i, :last_batch_amount] = np.stack(labels_list[10*i:10*i+last_batch_amount], axis=0)
    # print(rvec_batch.shape)

    # 每十张一个batch处理
    full_batch_num = len(global_imgs_list) // 10
    last_batch_amount = len(global_imgs_list) % 10
    # print(full_batch_num)
    # print(last_batch_amount)
    print("生成roi图片并保存point_size...")
    for i in tqdm(range(full_batch_num+1), desc="处理批次(10imgs/batch)"):
        # print(f"i:{i}")
        if i != full_batch_num:
            # print(f"{10 * i}:{10 * (i + 1)}")
            global_imgs_batch = np.stack(global_imgs_list[10 * i:10 * (i + 1)], axis=0)
            rvec_batch = np.stack(rvec_list[10*i:10*(i+1)], axis=0)
            tvec_batch = np.stack(tvec_list[10 * i:10 * (i + 1)], axis=0)
            labels_batch = np.stack(labels_list[10 * i:10 * (i + 1)], axis=0)
        else:
            # print(f"{10 * i}:{10*i+last_batch_amount}")
            global_imgs_batch = np.stack(global_imgs_list[10*i:10*i+last_batch_amount], axis=0)
            rvec_batch = np.stack(rvec_list[10*i:10*i+last_batch_amount], axis=0)
            tvec_batch = np.stack(tvec_list[10*i:10*i+last_batch_amount], axis=0)
            labels_batch = np.stack(labels_list[10*i:10*i+last_batch_amount], axis=0)

        # 输入zb
        roi_imgs_batch, point_size_batch = zb.process_zbuffer_with_rt_batch(global_imgs_batch, rvec_batch, tvec_batch,
                                                                        labels_batch)

        # 写入roi到roi_images文件夹
        # 直接遍历np矩阵就是在第0维上遍历
        # print("保存roi图像中...\n")
        for k, roi_imgs in enumerate(roi_imgs_batch):
            # print(f"roi_{idx_list[10*i+k]}")
            roi_imgs_path = os.path.join(output_root_path, "roi_images", f"roi_{idx_list[10*i+k]}")
            os.makedirs(roi_imgs_path, exist_ok=True)
            for idx, roi_img in enumerate(roi_imgs):
                roi_name = f"{idx + 1}.png"
                roi_img_path = os.path.join(roi_imgs_path, roi_name)
                cv2.imwrite(roi_img_path, roi_img)

        # 写入接收的point_size到json
        # print("写入point_size中...\n")
        for k, point_size in enumerate(point_size_batch):
            # print(f"label_{idx_list[10*i+k]}.json")
            label_path = os.path.join(output_root_path, "labels", f"label_{idx_list[10*i+k]}.json")
            with open(label_path, 'w') as label_file:
                # print(rvec_list[10*i+k].reshape(3).tolist())
                json_content = (f"{{\n"
                                f"    \"rvec\": {rvec_list[10*i+k].reshape(3).tolist()},\n"
                                f"    \"tvec\": {tvec_list[10*i+k].reshape(3).tolist()},\n"
                                f"    \"labels\": {labels_list[10*i+k].tolist()},\n"
                                f"    \"point_size\": {point_size.tolist()}\n"
                                f"}}")
                # print(json_content)
                label_file.writelines(json_content)

        # print("完成.\n")

def generate_test_global_data(raw_data_path:str, output_data_path:str, start_idx:int=0):
    idx = start_idx
    dir_list = os.listdir(raw_data_path)
    # print(dir_list)
    for dir_name in dir_list:
        path1 = os.path.join(raw_data_path, dir_name, "imageRT", "imagert1")
        dir_list2 = os.listdir(path1)
        dir_list2.sort(key=int)

        output_test_root_path = os.path.join(output_data_path, "test_datas")
        output_test_path = os.path.join(output_test_root_path, f"{idx}")

        for i in range(0, -2, -1):
            # 映射序号0->1, -1->2
            if i == 0:
                k = 1
            else:
                k = 2

            root_path = os.path.join(path1, dir_list2[i])
            image_path = os.path.join(root_path, "image.png")
            output_img_path1 = os.path.join(output_test_path, f"image_{k}.png")
            os.makedirs(output_test_path, exist_ok=True)
            shutil.copy(image_path, output_img_path1)

            # 读取txt，生成json
            # 拼接路径
            old_txt_path = os.path.join(root_path, "rt.txt")
            new_json_path = os.path.join(output_test_path, f"label_{k}.json")
            # print(new_json_path)

            # print(image_path1, old_txt_path)

            # 读取txt
            with open(old_txt_path, 'r') as txt_file:

                # 读取txt内容
                lines = txt_file.readlines()
                if not len(lines) == 3:
                    print(f"[ERROR] rt.txt 's format seems wrong.")
                rvec = [float(x.strip()) for x in lines[0].strip().split(",")]
                tvec = [float(x.strip()) for x in lines[1].strip().split(",")]
                labels = [int(x) for x in lines[2].strip().split()]
                # 转换成 1 存在， 0 不存在
                labels_convert = [0 if x == 4 else 1 for x in labels]

            # # 蓝色场地，转换成一号位从左边数起
            # labels_convert_temp = [labels_convert[2], labels_convert[1], labels_convert[0],
            #                        labels_convert[5], labels_convert[4], labels_convert[3],
            #                        labels_convert[8], labels_convert[7], labels_convert[6],
            #                        labels_convert[11], labels_convert[10], labels_convert[9]]
            # labels_convert = labels_convert_temp

            _, point_size_list = zb.process_zbuffer_with_rt(cv2.imread(image_path),
                                                            np.stack(rvec, axis=0).reshape(3, 1),
                                                            np.stack(tvec, axis=0).reshape(3, 1), labels_convert)

            # 生成json
            with open(new_json_path, 'w') as json_file:
                json_content = (f"{{\n"
                                f"    \"rvec\": {rvec},\n"
                                f"    \"tvec\": {tvec},\n"
                                f"    \"labels\": {labels_convert},\n"
                                f"    \"point_size\": {point_size_list}"
                                f"}}")
                json_file.writelines(json_content)
        idx += 1

def yolov5_detect_parse_opt(model_path:str, imgs_path:str, data_yaml:str,conf_thres:float, iou_thres:float, max_det:int):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=model_path, help="model path or triton URL")
    parser.add_argument("--source", type=str, default=imgs_path, help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=r"D:\A_myData\Pytorch\pytorch\dataset_yaml\corner8.yaml", help="(optional) dataset.yaml path")
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

def generate_corner_datasets(output_root_path:str, detect_model_path:str, data_yaml:str, conf_thres:float, iou_thres:float, max_det:int):
    """
        利用已有v5角点检测模型预测图片，生成角点数据集，同时转换txt为labelme_json
    需要结合官方yolov5-master包运行
    在总数据集的根目录下新建corner_jsons，存放json文件

    :param output_root_path: 总数据集的根目录
    :param detect_model_path: v5检测模型路径
    :param data_yaml:
    :param conf_thres: 用模型预测时的置信度阈值
    :param iou_thres:
    :param max_det:
    :return: 无
    """

    print("生成角点数据集...\n")

    # 创建json
    jsons_path = os.path.join(output_root_path, "corner_jsons")
    os.makedirs(jsons_path, exist_ok=True)
    images_path = os.path.join(output_root_path, "images")

    # 调用detct预测
    print("v5模型预测图片...\n")
    opt = yolov5_detect_parse_opt(detect_model_path, images_path, data_yaml, conf_thres, iou_thres, max_det)
    save_path = yolov5_detect.main(opt)
    # print(save_path)

    # 将txt转为json
    print("txt转换为labelme_json...\n")
    txts_path = os.path.join(save_path, "labels")
    txt_to_json(txts_path, images_path, jsons_path, ["corner", "trash"])

    print("完成\n")

if __name__ == "__main__":
    # 判断输入参数合法性
    model_test = YOLO(config.class_config.model_path)
    if model_test.task != "classify":
        raise TypeError(f"[ERROR] Not a class model: {config.class_config.model_path}")
    if not os.path.exists(config.path.raw_data_root_path):
        raise FileExistsError(f"{config.path.raw_data_root_path} not exist.")

    os.makedirs(os.path.join(config.path.output_root_path,"images"), exist_ok=True)
    os.makedirs(os.path.join(config.path.output_root_path,"labels"), exist_ok=True)
    os.makedirs(os.path.join(config.path.output_root_path,"roi_images"), exist_ok=True)

    # 生成总数据集
    if config.settings.generate_global_data:
        generate_global_dataset(config.path.raw_data_root_path, config.path.output_root_path, config.settings.start_idx)

    # 生成roi数据
    if config.settings.generate_roi_data:
        generate_roi_data(config.path.output_root_path)

    # 生成卷轴分类数据集
    if config.settings.generate_class_dataset:
        generate_11class_dataset(config.class_config.model_path, os.path.join(config.path.output_root_path,"roi_images"),
                                 config.path.output_root_path, config.class_config.start_idx, config.class_config.img_root_name,
                                 config.class_config.label_name_list, config.class_config.model_predict_output_dir)

    # 生成角点数据集
    if config.settings.generate_corner_dataset:
        generate_corner_datasets(config.path.output_root_path, config.detect_config.model_path, config.detect_config.data_yaml,
                                 config.detect_config.conf_thres, config.detect_config.iou_thres, config.detect_config.max_det)

    # 生成测试数据集
    if config.settings.generate_test_global_data:
        generate_test_global_data(config.path.raw_data_root_path, config.path.output_root_path, config.settings.test_data_start_idx)