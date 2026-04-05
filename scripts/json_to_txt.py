import json
import os
from tqdm import tqdm
import numpy

def convert_label(json_dir, save_dir, classes: list):
    json_paths = os.listdir(json_dir)

    # 循环处理所有json
    for json_path in tqdm(json_paths):
        path = os.path.join(json_dir, json_path)

        # 打开文件
        with open(path, 'r') as load_f:
            json_dict = json.load(load_f)

        # 获取图像宽高
        h, w = json_dict['imageHeight'], json_dict['imageWidth']

        # 拼接输出txt路径
        txt_path = os.path.join(save_dir, json_path.replace('json', 'txt'))
        txt_file = open(txt_path, 'w')

        # 遍历json下的所有框（一个shape就是一个框）
        for shape_dict in json_dict['shapes']:
            # 尝试匹配json里的标签和输入标签映射
            label = shape_dict['label']
            try:
                label_index = classes.index(label)
            except ValueError as e:
                print(f"Warning: {e}, skipping {json_path}...")
                continue

            shape_type = shape_dict["shape_type"]
            # 正矩形框（labelme_json：左上右下点 -> yolo_txt：x_center y_center width height）
            if shape_type == "rectangle":
                # 点归一化
                points_normalize = numpy.zeros((2,2))
                for idx,point in enumerate(shape_dict["points"]):
                    points_normalize[idx, 0] = float(point[0]) / w
                    points_normalize[idx, 1] = float(point[1]) / h
                # 计算：x_center y_center width height
                # 逻辑待补全：必须是大减小
                width = points_normalize[1,0] - points_normalize[0,0]
                height = points_normalize[1,1] - points_normalize[0,1]
                x_center = points_normalize[0,0] + width / 2
                y_center = points_normalize[0,1] + height / 2
                # 写入一行
                txt_file.write(f"{label_index} {x_center} {y_center} {width} {height}\n")

            # 多边形（仅支持四边形）
            elif shape_type == "polygon":
                # 判断是不是四边形
                if not len(shape_dict["points"]) == 4:
                    print(f"[Warn]: 不是四边形, 跳过 {json_path}...")
                    continue
                # 点归一化
                points_normalize = numpy.zeros((4,2))
                for idx,point in enumerate(shape_dict["points"]):
                    points_normalize[idx, 0] = float(point[0]) / w
                    points_normalize[idx, 1] = float(point[1]) / h
                # 写入一行
                txt_file.write(f"{label_index} "
                               f"{points_normalize[0, 0]} {points_normalize[0, 1]} "
                               f"{points_normalize[1, 0]} {points_normalize[1, 1]} "
                               f"{points_normalize[2, 0]} {points_normalize[2, 1]} "
                               f"{points_normalize[3, 0]} {points_normalize[3, 1]}\n")

            else:
                print(f"[Warn]: {shape_type} not support. ")
                continue

if __name__ == "__main__":
    json_dir = r"D:\A_myData\RC26-Vision\dataset\corner9\jsons"
    save_dir = r"D:\A_myData\RC26-Vision\dataset\corner9\txts"
    classes = ["corner", "trash"]

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    convert_label(json_dir, save_dir, classes)