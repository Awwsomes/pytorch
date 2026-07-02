import json
import os
from tqdm import tqdm
import numpy as np


def convert_label(json_dir, save_dir, classes: list):
    json_paths = os.listdir(json_dir)

    # 统计信息
    total_jsons = 0
    converted_jsons = 0
    skipped_empty_shapes = 0  # 新增：统计shapes为空的文件
    skipped_unsupported_shape = 0
    skipped_unknown_label = 0

    # 循环处理所有json
    for json_path in tqdm(json_paths):
        # 只处理.json文件
        if not json_path.lower().endswith('.json'):
            continue

        total_jsons += 1
        path = os.path.join(json_dir, json_path)

        # 打开文件
        with open(path, 'r', encoding="utf-8") as load_f:
            try:
                json_dict = json.load(load_f)
            except json.JSONDecodeError:
                print(f"\n[ERROR]: JSON格式错误，跳过 {json_path}")
                continue

        # 关键修复：提前检查shapes字段是否为空
        if "shapes" not in json_dict or len(json_dict["shapes"]) == 0:
            print(f"\n[INFO]: 空标签文件（无标注），跳过 {json_path}")
            skipped_empty_shapes += 1
            continue  # 不生成空的txt文件

        # 获取图像宽高
        h, w = json_dict['imageHeight'], json_dict['imageWidth']

        # 拼接输出txt路径
        txt_path = os.path.join(save_dir, os.path.splitext(json_path)[0] + '.txt')

        # 使用with语句自动管理文件句柄
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            # 遍历json下的所有框（一个shape就是一个框）
            for shape_dict in json_dict['shapes']:
                # 尝试匹配json里的标签和输入标签映射
                label = shape_dict['label']
                try:
                    label_index = classes.index(label)
                except ValueError:
                    print(f"\n[WARN]: 未知标签 '{label}'，跳过该标注 {json_path}")
                    skipped_unknown_label += 1
                    continue

                shape_type = shape_dict["shape_type"]

                # 正矩形框（labelme_json：左上右下点 -> yolo_txt：x_center y_center width height）
                if shape_type == "rectangle":
                    points = np.array(shape_dict["points"], dtype=np.float32)

                    # 修复：确保坐标是正确的（无论用户从哪个方向画框）
                    x_min, y_min = np.min(points, axis=0)
                    x_max, y_max = np.max(points, axis=0)

                    # 归一化
                    x_min /= w
                    y_min /= h
                    x_max /= w
                    y_max /= h

                    # 计算YOLO格式
                    width = x_max - x_min
                    height = y_max - y_min
                    x_center = x_min + width / 2
                    y_center = y_min + height / 2

                    # 写入一行
                    txt_file.write(f"{label_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                # 多边形（仅支持四边形，用于旋转目标检测OBB）
                elif shape_type == "polygon":
                    points = np.array(shape_dict["points"], dtype=np.float32)

                    # 判断是不是四边形
                    if len(points) != 4:
                        print(f"\n[WARN]: 不支持{len(points)}边形，跳过该标注 {json_path}")
                        skipped_unsupported_shape += 1
                        continue

                    # 点归一化
                    points[:, 0] /= w
                    points[:, 1] /= h

                    # 写入一行（YOLO OBB格式：class x1 y1 x2 y2 x3 y3 x4 y4）
                    txt_file.write(f"{label_index} "
                                   f"{points[0, 0]:.6f} {points[0, 1]:.6f} "
                                   f"{points[1, 0]:.6f} {points[1, 1]:.6f} "
                                   f"{points[2, 0]:.6f} {points[2, 1]:.6f} "
                                   f"{points[3, 0]:.6f} {points[3, 1]:.6f}\n")

                else:
                    print(f"\n[WARN]: 不支持的形状类型 '{shape_type}'，跳过该标注 {json_path}")
                    skipped_unsupported_shape += 1
                    continue

        converted_jsons += 1

    # 输出最终统计信息
    print("\n" + "=" * 60)
    print("转换完成！统计信息：")
    print(f"总JSON文件数: {total_jsons}")
    print(f"成功转换: {converted_jsons}")
    print(f"跳过（空标签/无标注）: {skipped_empty_shapes}")
    print(f"跳过（未知标签）: {skipped_unknown_label}")
    print(f"跳过（不支持的形状）: {skipped_unsupported_shape}")
    print("=" * 60)


if __name__ == "__main__":
    json_dir = r"F:\RC2026_OFFLINE\datasets\juanZhou_det_mix5\jsons"
    save_dir = r"F:\RC2026_OFFLINE\datasets\juanZhou_det_mix5\labels"
    classes = ["red", "blue"]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    convert_label(json_dir, save_dir, classes)