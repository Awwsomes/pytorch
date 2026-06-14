import os
import json
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

def crop_bbox_image(image, points):
    """
    根据标注点裁剪图像外接矩形
    支持2点(rectangle)和多点(polygon)格式
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min = int(max(0, min(x_coords)))
    y_min = int(max(0, min(y_coords)))
    x_max = int(min(image.shape[1], max(x_coords)))
    y_max = int(min(image.shape[0], max(y_coords)))

    if x_max <= x_min or y_max <= y_min:
        return None
    return image[y_min:y_max, x_min:x_max]


def load_classification_model(model_path, class_names=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    加载Ultralytics分类模型
    class_names传入则覆盖模型自带类别，否则使用模型内置类别
    """
    model = YOLO(model_path)
    if class_names is None:
        class_names = model.names
    return model, class_names, device


def predict_class(crop_img_bgr, model, class_names, device):
    """
    对裁剪的BGR图像进行分类推理，返回最高置信度类别与置信度
    修复：使用传入的class_names映射类别，而非模型自带names
    """
    if crop_img_bgr is None or crop_img_bgr.size == 0:
        return None, 0.0

    try:
        results = model(crop_img_bgr, verbose=False, device=device)[0]
        if results.probs is None or len(results.probs) == 0:
            return None, 0.0

        top1_idx = int(results.probs.top1)
        confidence = float(results.probs.top1conf)
        # 使用传入的类别列表映射，确保和用户定义一致
        pred_class = class_names[top1_idx]
        return pred_class, confidence
    except Exception as e:
        print(f"    推理异常: {str(e)}")
        return None, 0.0


def update_labelme_json(
        img_folder,
        json_in_folder,
        json_out_folder,
        model_path,
        class_names=None
):
    """
    主函数：批量处理labelme json，用分类模型更新每个检测框的类别
    图片匹配规则：按文件名与json一一对应，忽略json内部的imagePath
    兼容 rectangle 和 polygon 两种标注类型
    """
    os.makedirs(json_out_folder, exist_ok=True)

    # 加载模型
    model, class_names, device = load_classification_model(model_path, class_names)
    print(f"使用设备: {device}")
    print(f"类别数量: {len(class_names)}")
    print(f"类别列表: {class_names}\n")

    # 遍历所有json文件
    json_files = list(Path(json_in_folder).glob("*.json"))
    total = len(json_files)
    print(f"找到 {total} 个json文件待处理\n")

    for idx, json_path in enumerate(json_files, 1):
        json_name = json_path.name
        img_base_name = json_path.stem
        print(f"[{idx}/{total}] 处理: {json_name}")

        # 读取json
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  读取json失败: {str(e)}，跳过")
            continue

        # 在图片文件夹中查找同名图片，兼容常见格式
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.bmp', '.JPG', '.PNG']:
            candidate = os.path.join(img_folder, img_base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"  警告: 图片文件夹中找不到 {img_base_name} 对应的图片，跳过")
            continue

        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            print(f"  警告: 图片读取失败 {img_path}，跳过")
            continue

        # 遍历每个标注框
        shapes = data.get('shapes', [])
        if len(shapes) == 0:
            print(f"  警告: 当前json无任何标注框")

        update_count = 0
        skip_count = 0
        for i, shape in enumerate(shapes):
            shape_type = shape.get('shape_type', 'unknown')
            points = shape.get('points', [])

            # 兼容矩形和多边形，只要有坐标点就处理
            if shape_type not in ['rectangle', 'polygon']:
                print(f"    跳过第{i}个标注：类型为{shape_type}，不支持")
                skip_count += 1
                continue

            if len(points) < 2:
                print(f"    跳过第{i}个标注：坐标点数量不足({len(points)}个)")
                skip_count += 1
                continue

            # 裁剪外接矩形
            crop_img = crop_bbox_image(image, points)
            if crop_img is None:
                print(f"    跳过第{i}个标注：裁剪区域为空")
                skip_count += 1
                continue

            # 推理分类
            pred_class, conf = predict_class(crop_img, model, class_names, device)
            if pred_class is not None:
                shape['old_label'] = shape.get('label', '')
                shape['label'] = pred_class
                shape['classification_confidence'] = round(conf, 4)
                update_count += 1
            else:
                print(f"    跳过第{i}个标注：推理无结果")
                skip_count += 1

        # 写入新json
        out_path = os.path.join(json_out_folder, json_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"  更新 {update_count} 个标注框，跳过 {skip_count} 个 -> {out_path}")

    print("\n全部处理完成！")


# ===================== 使用示例 =====================
if __name__ == "__main__":
    # ====== 请根据你的实际情况修改以下配置 ======
    IMG_FOLDER = r"D:\A_myData\RC26-Vision\dataset\juanZhou_det1\image1"  # 图片文件夹
    JSON_IN_FOLDER = r"D:\A_myData\RC26-Vision\dataset\juanZhou_det1\jsons"  # 输入labelme json文件夹
    JSON_OUT_FOLDER = r"D:\A_myData\RC26-Vision\dataset\juanZhou_det1\jsons_fix"  # 输出json文件夹
    MODEL_PATH = r"D:\A_myData\RC26-Vision\Pytorch\yolo11\runs\classify\卷轴分类3_仿真带旋转+现实第一批_32类_不带筛空功能\weights\best.pt"  # 分类模型pt文件路径

    # 分类模型的类别列表，索引必须与模型输出顺序一致
    CLASS_NAMES = ["1", "10", "11", "12", "13", "14", "15", "16", "17", "18",
                   "19", "2", "20", "21", "22", "23", "24", "25", "26", "27",
                   "28", "29", "3", "30", "31", "32", "4", "5", "6", "7", "8", "9"]
    # ============================================

    update_labelme_json(
        img_folder=IMG_FOLDER,
        json_in_folder=JSON_IN_FOLDER,
        json_out_folder=JSON_OUT_FOLDER,
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES
    )