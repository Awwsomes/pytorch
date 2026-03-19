import os
import json
import cv2
from tqdm import tqdm

# 仅支持正矩形
def txt_to_json(txt_path:str,image_path:str, output_json_path:str, label_idx_list:list):
    """
    将yolo标准格式的txt文件转换为labelme格式的json文件

    :param txt_path: 存放要转换的txt
    :param image_path: 对应的图像路径
    :param output_json_path: 输出json路径
    :param label_idx_list: 类别映射列表，映射txt里的序号和对应的标签名
    :return: 无

    txt文件必须是yolo标准格式：x_center y_center width height
    转换成labelme格式下的正矩形框：左上点 右下点
    """
    if not os.path.exists(txt_path):
        print(f"[Warn]: {txt_path} not exists!")
        return -1
    os.makedirs(output_json_path,exist_ok=True)

    # 读取txt
    list_txt = os.listdir(txt_path)
    list_txt = [x for x in list_txt if os.path.isfile(os.path.join(txt_path,x))]
    list_txt = [x for x in list_txt if x.endswith(".txt")]

    for txt in tqdm(list_txt):
        # 拼接路径
        file_root_name = os.path.splitext(txt)[0]
        path_txt = os.path.join(txt_path,txt)
        path_json = os.path.join(output_json_path,f"{file_root_name}.json")
        path_image = ""
        for i,ext in enumerate([".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]):
            path_image = os.path.join(image_path,f"{file_root_name}{ext}")
            if os.path.exists(path_image):
                break
            else:
                path_image = ""
        if path_image == "":
            print(f"[Warn]: {txt} 没有相对应的图像，跳过")
            continue

        # 获取图像尺寸
        img = cv2.imread(path_image)
        img_height, img_width, _ = img.shape  # H,W,C
        # print(img_width)

        shapes = []
        with open(path_txt,'r') as txt_file:
            # 遍历每一行（一个框）
            for idx, line in enumerate(txt_file):
                # 拿取原数据
                list_raw_data = []  # 格式：label_idx, x_center, y_center, width, height
                for data in line.strip().split():
                    list_raw_data.append(float(data))

                # 判断是否为四边形
                if len(list_raw_data) > 5:
                    print(f"[Warn]: {txt} line {idx + 1} 's shape does not support, skip this line...")
                    continue

                # 判断是否有该标签
                try:
                    label_idx = label_idx_list[int(list_raw_data[0])]
                except:
                    print(f"[Warn]: {txt} 行 {idx + 1} 的标签不在程序输入的标签映射列表中，跳过该行")
                    continue

                # 转换为左上右下点,反归一化
                point_left_up = [(list_raw_data[1] - list_raw_data[3] / 2) * img_width, (list_raw_data[2] - list_raw_data[4] / 2) * img_height]
                point_right_down = [(list_raw_data[1] + list_raw_data[3] / 2) * img_width, (list_raw_data[2] + list_raw_data[4] / 2) * img_height]
                points = [point_left_up, point_right_down]

                # 格式化输出数据
                shape:dict = {
                    "label": label_idx,
                    "points": points,
                    "shape_type": "rectangle",
                    "flags": {}
                }
                shapes.append(shape)

        # print(1)

        # 格式化输出数据
        json_data = {
            "version": "5.1.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": f"{path_image}",
            "imageData": generate_imageData(path_image),
            "imageHeight": img_height,
            "imageWidth": img_width
        }

        # 写入
        with open(path_json,'w',encoding="utf-8") as json_file:
            json.dump(json_data,json_file,indent=4)
            # print(f"{os.path.splitext(txt)[0]}.json write.")

import base64
def generate_imageData(image_path):
    """
    读取图片文件，生成labelme格式的imageData字符串
    :param image_path: 图片文件路径（支持jpg/png等）
    :return: base64编码的imageData字符串
    """
    # 以二进制模式读取图片
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        # 进行base64编码（注意：labelme用的是标准base64，无urlsafe转换）
        image_data = base64.b64encode(image_bytes).decode('utf-8')
    return image_data

if __name__ == "__main__":
    input_txt_path = r"D:\A_myData\RC26-Vision\dataset\corner4\txts"
    input_image_path = r"D:\A_myData\RC26-Vision\dataset\corner4\images"
    output_json_path = r"D:\A_myData\RC26-Vision\dataset\corner4\jsons_test"
    label_idx_list = ["corner"]
    txt_to_json(input_txt_path,input_image_path,output_json_path,label_idx_list)