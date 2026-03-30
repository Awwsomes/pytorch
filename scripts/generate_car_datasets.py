import os
import shutil
import json
import cv2
import numpy as np
from tqdm import tqdm
import re
from ultralytics import YOLO

import zbuffer.zb_main as zb
from convert_train_dataset_to_class_dataset import classify_dataset_by_label
from rename_data import rename_data_one_dir
from class_about.change_txt_format import change_txt_format

def generate_global_dataset(root_dir:str,output_root_path:str,start_idx:int, blue_or_red:bool):
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
    :param blue_or_red: 蓝色场地：true 红色场地：false
    :return: 无
    """
    print("拷贝并重命名文件...")

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
            if blue_or_red:
                labels_convert_temp = [labels_convert[2], labels_convert[1], labels_convert[0],
                                       labels_convert[5], labels_convert[4], labels_convert[3],
                                       labels_convert[8], labels_convert[7], labels_convert[6],
                                       labels_convert[11], labels_convert[10], labels_convert[9]]
                labels_convert = labels_convert_temp

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
    print("拷贝并重命名文件完成.")

def generate_11class_dataset(class_model_path, input_roi_img_path, output_root_dir, start_idx=0):
    """
    用已有模型预测， 从车上log数据生成分类数据集
    使用模型预测出的最高置信度的类别作为图片的类别
    默认文件名起始序号为0

    :param class_model_path: 用于预测的分类模型
    :param input_roi_img_path: 存放待预测图片的文件夹
    :param output_root_dir: 输出分类数据集的根文件夹
    :param start_idx: 文件名起始序号
    :return: 无
    """
    print("生成分类数据集中...")

    # 创建临时的检测数据集
    root_path = "temp"
    imgs_path = os.path.join(root_path, "images")
    labels_path = os.path.join(root_path, "labels_no_fix")
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(imgs_path, exist_ok=True)

    # 获取输出图片根文件名（和数据集文件夹名相同）
    img_root_name = os.path.split(output_root_dir)[1]
    img_root_name += "_"

    # 重命名并复制roi图片
    rename_data_one_dir(input_roi_img_path, imgs_path, img_root_name=img_root_name, start_idx=start_idx, list_ext=[".png", ".jpg", ".jpeg", ".bmp"])

    # 读取模型
    model = YOLO(class_model_path)
    if model.task != "classify":
        raise TypeError(f"[ERROR] Not a class model: {class_model_path}")

    # 遍历roi，给YOLO分类预测
    model_predict_output_dir = r"D:\A_myData\RC26-Vision\Pytorch\yolo11_cls\runs\classify"
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

    # 拷贝txt到临时检测数据集内（原路径 yolo11_cls\runs\classify）
    txt_list = os.listdir(os.path.join(result[0].save_dir, "labels"))
    for txt in txt_list:
        txt_path = os.path.join(result[0].save_dir, "labels", txt)
        new_txt_path = os.path.join(labels_path, txt)
        shutil.copy(txt_path, new_txt_path)

    # 转换txt格式（conf label -> label conf）
    new_labels_path = os.path.join(root_path, "labels")
    change_txt_format(labels_path, new_labels_path)

    # 生成分类数据集
    classify_dataset_by_label(root_path, output_root_dir, ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                                                                "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                                                                "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"])

    # 删除临时的检测数据集
    shutil.rmtree(root_path)

    # 保存信息：能从生成的数据集找到原始模型预测出来的txt，包含top5置信度
    raw_model_predict_path = os.path.join(output_root_dir, "raw_model_predict_path.txt")
    with open(raw_model_predict_path, 'w') as txt:
        txt.write(f"{result[0].save_dir}")
    with open(os.path.join(result[0].save_dir, "对应数据集文件夹路径.txt"), 'w') as txt:
        txt.write(f"{output_root_dir}")

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
    print("生成roi图像中...")

    # 读图像，读rt
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
    for img_name in global_img_names_list:
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
    global_imgs_batch = np.stack(global_imgs_list, axis=0)
    rvec_batch = np.stack(rvec_list, axis=0)
    tvec_batch = np.stack(tvec_list, axis=0)
    labels_batch = np.stack(labels_list, axis=0)
    # print(rvec_batch.shape)

    # 输入zb
    roi_imgs_batch, point_size_batch = zb.process_zbuffer_with_rt_batch(global_imgs_batch, rvec_batch, tvec_batch,
                                                                        labels_batch)

    # 写入roi到roi_images文件夹
    # 直接遍历np矩阵就是在第0维上遍历
    print("保存roi图像中...")
    for i, roi_imgs in enumerate(roi_imgs_batch):
        roi_imgs_path = os.path.join(output_root_path, "roi_images", f"roi_{idx_list[i]}")
        os.makedirs(roi_imgs_path, exist_ok=True)
        for idx, roi_img in enumerate(roi_imgs):
            roi_name = f"{idx + 1}.png"
            roi_img_path = os.path.join(roi_imgs_path, roi_name)
            cv2.imwrite(roi_img_path, roi_img)

    # 写入接收的point_size到json
    print("写入point_size中...")
    for i, point_size in enumerate(point_size_batch):
        label_path = os.path.join(output_root_path, "labels", f"label_{idx_list[i]}.json")
        with open(label_path, 'w') as label_file:
            json_content = (f"{{\n"
                            f"    \"rvec\": {rvec_list[i].T.tolist()},\n"
                            f"    \"tvec\": {tvec_list[i].T.tolist()},\n"
                            f"    \"labels\": {labels_list[i].tolist()},\n"
                            f"    \"point_size\": {point_size.tolist()}\n"
                            f"}}")
            label_file.writelines(json_content)

def generate_car_datasets(root_dir:str,output_root_path:str,start_idx:int, blue_or_red:bool, class_model_path:str, output_class_dataset_root_path:str):
    """
    处理车上日志数据，生成总数据集，roi数据，卷轴分类数据集
    :param root_dir: 日志数据根目录
    :param output_root_path: 输出总数据集的根目录，roi数据也在其中
    :param start_idx: 文件名起始序号，管理global images，labels 和 roi文件夹的编号，分类数据集的起始序号固定为0
    :param blue_or_red: 蓝色场地：true 红色场地：false
    :param class_model_path: 用于生成卷轴分类数据集的分类模型pt
    :param output_class_dataset_root_path: 输出分类数据集的根目录
    :return: 无
    """
    # 判断输入参数合法性
    model_test = YOLO(class_model_path)
    if model_test.task != "classify":
        raise TypeError(f"[ERROR] Not a class model: {class_model_path}")
    if not os.path.exists(root_dir):
        raise FileExistsError(f"{root_dir} not exist.")

    os.makedirs(os.path.join(output_root_path,"images"), exist_ok=True)
    os.makedirs(os.path.join(output_root_path,"labels"), exist_ok=True)
    os.makedirs(os.path.join(output_root_path,"roi_images"), exist_ok=True)

    # 生成总数据集
    generate_global_dataset(root_dir, output_root_path, start_idx, blue_or_red)

    # 生成roi数据
    generate_roi_data(output_root_path)

    # 生成卷轴分类数据集
    generate_11class_dataset(class_model_path, os.path.join(output_root_path,"roi_images"), output_class_dataset_root_path, 0)

if __name__ == "__main__":
    input_dir = r"D:\A_myData\RC26-Vision\dataset\A_car\2026_3_26_"
    output_root_dir = r"D:\A_myData\RC26-Vision\dataset\A_car\2026_3_26_output"
    start_idx = 1
    yolo11cls_path = "D:\A_myData\RC26-Vision\Pytorch\pytorch\src\weights\hou_li_11cls_0113_09.pt"
    output_cls_dataset_root_path = r"D:\A_myData\RC26-Vision\dataset\juanZhou_car2"

    # generate_car_datasets(input_dir, output_root_dir, start_idx, True, yolo11cls_path, output_cls_dataset_root_path)

    generate_11class_dataset(yolo11cls_path, r"D:\A_myData\RC26-Vision\dataset\A_car\2026_3_28\roi_images", output_cls_dataset_root_path, 1056)